"""Server-side orchestration — synchronous (FedServer) and asynchronous (AsyncFedServer).

Sync mode  (federation.mode = "sync")
--------------------------------------
Classic FedAvg barrier.  Every round:
  1. Broadcast current global model to all nodes.
  2. All nodes train independently for local_epochs.
  3. Server WAITS until every node has finished (pool.map barrier).
  4. Aggregate all updates at once, save checkpoint, advance to next round.

  Pro:  simple, theoretically well-studied, deterministic.
  Con:  the slowest node in each round blocks everyone else.
        A node with 10× more data than others stalls the whole federation.

Async mode  (federation.mode = "async")
-----------------------------------------
True asynchronous federation — no barrier, no rounds.  Instead:
  1. Every node runs independently in its own thread, cycling:
       pull current global model (snapshot + version number)
       → train locally for local_epochs
       → push update to server immediately (no waiting for siblings)
  2. Server aggregates each push the moment it arrives, under a lock.
  3. Global model VERSION increments by 1 after each single-node push.
     Total version count = n_nodes × async_node_cycles.
  4. A checkpoint is saved after every push so the full learning trajectory
     is recorded (global_v{N}_{node_name}.pt).

  Staleness handling
  ------------------
  Between the moment a node pulled the model (at version V_pull) and the
  moment it pushes its update, other nodes may have already submitted and
  advanced the global model to version V_now.  The "staleness" of this
  submission is:

      staleness = V_now − V_pull

  The server discounts the update proportionally:

      w = 1 / (1 + staleness_alpha × staleness)

  so a node that trained on a 5-version-old model contributes less than one
  that trained on the current model.

  Aggregation uses a DELTA approach rather than full re-averaging:

      global_new[k] = global_old[k] + w × (node_sd[k] − pulled_sd[k])

  The delta (node_sd − pulled_sd) is the implicit parameter update the node
  learned.  This correctly handles the case where the global model has been
  updated by other nodes between pull and push, since we're applying the
  learned improvement ON TOP of the current global rather than overwriting it.

  For per-class classification weights the delta is applied channel-by-channel,
  only for classes this node owns — exactly the same class-conditional logic as
  the sync aggregation.

  Pro:  fast nodes don't wait for slow ones; the global model progresses
        continuously; better wall-clock utilisation on heterogeneous hardware.
  Con:  slightly less stable than sync early in training (stale updates from
        very slow nodes can temporarily pull the model in an outdated direction);
        mitigated by staleness_alpha and the warmup schedule.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

# Each round ships a ~350-tensor state_dict to every node process and back.
# Torch's default "file_descriptor" CPU tensor sharing strategy opens one fd
# per tensor for this kind of IPC and can exhaust the process's fd limit
# over many rounds; "file_system" avoids that at a small disk-temp-file cost.
mp.set_sharing_strategy("file_system")

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel

from .client import NodeRoundResult, run_node_round
from .config import FedYoloConfig, NodeConfig
from .model import aggregate, build_model, per_class_param_names

#logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fedyolo.server")
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Shared: evaluation helper + mAP history display
# ─────────────────────────────────────────────────────────────────────────────

def _run_eval(cfg: FedYoloConfig, state_dict: dict, label: str) -> dict | None:
    """Run silent evaluation; return a metrics summary dict or None on failure.
    Also writes {output_dir}/live_map.json so in-progress node threads can
    display the latest known mAP in their progress bar without waiting for
    the full run to complete."""
    try:
        from .evaluate import evaluate_state_dict, per_class_map50
        metrics = evaluate_state_dict(cfg, state_dict, name=label, verbose=False)
        result = {
            "label":    label,
            "map50":    float(metrics.box.map50),
            "map5095":  float(metrics.box.map),
            "per_class": per_class_map50(metrics),
        }
        # Documented but previously never actually written -- this is what
        # client.py's _read_latest_map() polls to show live mAP in each
        # node's training progress bar. Best-effort: a write failure here
        # shouldn't fail the run.
        try:
            Path(cfg.output_dir).joinpath("live_map.json").write_text(json.dumps(result))
        except OSError as exc:
            log.warning(f"could not write live_map.json: {exc}")
        return result
    except Exception as exc:
        log.warning(f"evaluation failed ({label}): {exc}")
        return None


def _print_map_table(history: list[dict], global_classes: list[str]) -> None:
    """Print the mAP-over-time history to the terminal.

    Only Step/mAP50/mAP50-95 are columns: with >5-10 classes, one column per
    class doesn't fit any real terminal width, and fits *no* width at all when
    stdout is redirected to a file (Rich has no real size to measure against
    and silently collapses columns to blank). Full per-class numbers are never
    lost -- they're in every history entry and in the final summary.json --
    this just stops trying to cram them into a live table.
    """
    if not history:
        return

    table = Table(title="📈  DAFYOLO — Live mAP History", show_lines=True)
    table.add_column("Step",      style="cyan",  no_wrap=True)
    table.add_column("mAP50",     style="bold green")
    table.add_column("mAP50-95",  style="bold yellow")
    table.add_column("Best class (mAP50)", style="dim white")
    table.add_column("Worst class (mAP50)", style="dim white")

    best_map50 = max(h["map50"] for h in history)

    for h in history:
        is_best = h["map50"] == best_map50
        map50_str    = f"[bold magenta]{h['map50']:.4f} ★[/bold magenta]" if is_best else f"{h['map50']:.4f}"
        map5095_str  = f"{h['map5095']:.4f}"
        per_class = h.get("per_class") or {}
        if per_class:
            best_c  = max(per_class, key=per_class.get)
            worst_c = min(per_class, key=per_class.get)
            best_str  = f"{best_c} ({per_class[best_c]:.4f})"
            worst_str = f"{worst_c} ({per_class[worst_c]:.4f})"
        else:
            best_str = worst_str = "-"
        table.add_row(h["label"], map50_str, map5095_str, best_str, worst_str)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Sync server
# ─────────────────────────────────────────────────────────────────────────────
def _node_worker(args):
    node, cfg, global_sd, round_idx, teacher_sd, optimizer_state = args
    return run_node_round(node, cfg, global_sd, round_idx, teacher_sd, optimizer_state)


class FedServer:
    """Synchronous federated server — barrier aggregation every round."""

    def __init__(self, cfg: FedYoloConfig):
        self.cfg = cfg
        self.model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained, class_names=cfg.global_classes)
        self.global_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._map_history: list[dict] = []
        # None disables the diagnostic entirely (no extra work in aggregate()).
        self._merge_diag: list | None = [] if cfg.federation.merge_diagnostics else None
        # Per-node SGD momentum buffers, carried forward round-to-round instead
        # of restarting cold every round (see run_node_round's optimizer_state
        # param). Model WEIGHTS still come from the freshly-aggregated global
        # state each round -- only the gradient-momentum statistics persist.
        self._node_optimizer_state: dict[str, dict] = {}
        # See client.py's run_node_round for why random/np.random are seeded
        # here too, not just torch -- ultralytics' augmentation pipeline
        # reads from those, not torch's RNG.
        random.seed(cfg.seed)
        np.random.seed(cfg.seed % (2**32))
        torch.manual_seed(cfg.seed)

    def run(self) -> dict:
        previous_round_sd = None
        # Nodes within a round are independent (the barrier already waits for
        # all of them before aggregating), so up to max_concurrent_nodes can
        # train in parallel worker processes instead of strictly one-at-a-time.
        max_workers = max(1, min(self.cfg.federation.max_concurrent_nodes, len(self.cfg.nodes)))
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            for round_idx in range(self.cfg.federation.rounds):
                log.info(f"=== round {round_idx + 1}/{self.cfg.federation.rounds} ===")

                tasks = [
                    (node, self.cfg, self.global_state_dict, round_idx, previous_round_sd,
                     self._node_optimizer_state.get(node.name))
                    for node in self.cfg.nodes
                ]
                results: list[NodeRoundResult] = list(pool.map(_node_worker, tasks))

                for r in results:
                    owned = [self.cfg.global_classes[c] for c in r.owned_global_ids]
                    log.info(f"  node={r.name:10s} images={r.num_images:4d} owns={owned}")
                    self._node_optimizer_state[r.name] = r.optimizer_state

                previous_round_sd = {k: v.clone() for k, v in self.global_state_dict.items()}
                log.info("🧠 Aggregating node weights...")

                self.global_state_dict = aggregate(
                    node_state_dicts=[r.state_dict for r in results],
                    node_num_images=[r.num_images for r in results],
                    node_owned_global_ids=[r.owned_global_ids for r in results],
                    node_class_counts=[r.class_counts for r in results],
                    model=self.model,
                    nc=self.cfg.nc,
                    # previous_round_sd is this round's pre-aggregation global,
                    # i.e. exactly the snapshot every node started from -- the
                    # baseline the merge rules form their deltas against.
                    prev_global=previous_round_sd,
                    rule=self.cfg.federation.merge_rule,
                    trim=self.cfg.federation.merge_trim,
                    diag_out=self._merge_diag,
                    node_steps=([r.num_steps for r in results]
                                if self.cfg.federation.normalize_steps else None),
                    weight_basis=self.cfg.federation.weight_basis,
                    node_instance_totals=[sum(r.class_counts.values()) for r in results],
                )
                if self._merge_diag is not None:
                    (self.out_dir / "merge_diagnostics.json").write_text(
                        json.dumps(self._merge_diag, indent=2))

                ckpt_path = self.out_dir / f"global_round{round_idx + 1:03d}.pt"
                torch.save(
                    {"state_dict": self.global_state_dict,
                     "global_classes": self.cfg.global_classes},
                    ckpt_path,
                )
                # Keep only the last `keep_checkpoints` rounds on disk -- every
                # one of these is a full model (up to ~350MB for yolo26x).
                # Saving all of them for a 10-round run cost a real "No space
                # left on device" crash on this machine once model sizes grew
                # past yolov8s. Keeping more than one exists so a finished run
                # can be re-scored on a *different* eval set at several
                # late rounds rather than only its last: this config's
                # round-to-round mAP swings by up to 0.10 on the small
                # balanced-fraction eval set, so a single final-round number
                # is a noisy way to compare two runs. Delete only AFTER the
                # new one is safely written, so a mid-save crash never leaves
                # zero checkpoints on disk.
                stale = round_idx + 1 - self.cfg.federation.keep_checkpoints
                if stale > 0:
                    (self.out_dir / f"global_round{stale:03d}.pt").unlink(missing_ok=True)

                # ── live mAP evaluation ──────────────────────────────────────
                is_last = (round_idx + 1 == self.cfg.federation.rounds)
                due     = ((round_idx + 1) % self.cfg.federation.eval_interval == 0)
                if due or is_last:
                    label = f"round {round_idx + 1:03d}"
                    console.print(f"\n[bold yellow]⚡ Evaluating after {label}…[/bold yellow]")
                    result = _run_eval(self.cfg, self.global_state_dict, label)
                    if result:
                        self._map_history.append(result)
                        # Persisted so a later comparison can average the last
                        # few rounds instead of trusting a single final number;
                        # previously this history only ever reached the
                        # terminal, and was lost once the run ended.
                        (self.out_dir / "map_history.json").write_text(
                            json.dumps(self._map_history, indent=2))
                        _print_map_table(self._map_history, self.cfg.global_classes)

        final_path = self.out_dir / "global_final.pt"
        torch.save(
            {"state_dict": self.global_state_dict, 
             "global_classes": self.cfg.global_classes},
            final_path,
        )
        log.info(f"federation complete -> {final_path}")
        return self.global_state_dict


# ─────────────────────────────────────────────────────────────────────────────
# Async server
# ─────────────────────────────────────────────────────────────────────────────

class AsyncFedServer:
    """Asynchronous federated server — immediate delta aggregation, no barrier.

    Each node runs in its own thread cycling pull → train → push independently.
    The global model is updated after EVERY single-node push.
    """

    def __init__(
        self,
        cfg: FedYoloConfig,
        target_map: float | None = None,
        target_metric: str = "map50",
        patience: int | None = None,
        min_improvement: float = 0.002,
    ):
        """target_map/target_metric/patience/min_improvement are opt-in (all
        default to no-op) -- used by simulate_to_target.py to stop a node's
        cycle loop early once the target is reached or progress has plateaued,
        without changing behavior for the standard fixed-cycle-count run()."""
        self.cfg = cfg
        self.model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained, class_names=cfg.global_classes)

        # Shared global state — protected by _lock for all reads/writes
        self._global_sd: dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._version: int = 0          # increments after every push
        self._submission_count: int = 0
        self._last_ckpt_path: Path | None = None   # for pruning, see push()

        self._lock = threading.Lock()   # guards _global_sd, _version, _submission_count

        self.target_map = target_map
        self.target_metric = target_metric
        self.patience = patience
        self.min_improvement = min_improvement
        self._best_metric = -1.0
        self._evals_since_improvement = 0
        self.target_reached = False
        self.stopped_early = False

        # Per-node SGD momentum buffers, carried forward cycle-to-cycle instead
        # of restarting cold every cycle. Each key is only ever touched by its
        # own node's thread, so no lock is needed here.
        self._node_optimizer_state: dict[str, dict] = {}

        self._total_submissions = len(cfg.nodes) * cfg.federation.async_node_cycles

        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._map_history: list[dict] = []

        # Rich progress (set in run() before threads start)
        self._progress: Progress | None = None
        self._progress_task = None
        self._pending_eval: tuple | None = None   # set under lock in push(), consumed outside it
        
        # Limits how many nodes train simultaneously.
        # Default 1 = sequential execution with async semantics (safe for any machine).
        # Increase only if you have enough RAM for N concurrent models.
        self._train_semaphore = threading.Semaphore(cfg.federation.max_concurrent_nodes)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed % (2**32))
        torch.manual_seed(cfg.seed)

    # ── public interface used by node threads ──────────────────────────────

    def pull(self) -> tuple[dict, int]:
        """Return a deep copy of the current global model and its version."""
        with self._lock:
            sd_copy = {k: v.clone() for k, v in self._global_sd.items()}
            version = self._version
        return sd_copy, version

    def push(self, result: NodeRoundResult, pulled_sd: dict, pulled_version: int) -> None:
        """Immediately aggregate one node's update into the global model."""
        with self._lock:
            staleness = self._version - pulled_version
            self._aggregate_async(result, pulled_sd, staleness)
            self._version += 1
            self._submission_count += 1
            sub = self._submission_count

            owned_names = [self.cfg.global_classes[c] for c in result.owned_global_ids]
            log.info(
                f"[async] sub {sub:3d}/{self._total_submissions} | "
                f"node={result.name:10s} | staleness={staleness:2d} | "
                f"owns={owned_names} | global_v={self._version}"
            )

            # Per-submission checkpoint. Only the latest is kept on disk --
            # async saves one of these per submission (4 nodes x
            # async_node_cycles, e.g. 40 for the 20-epoch/node generation),
            # and at up to ~350MB each (yolo26x) that's easily 10GB+ for one
            # run if every version is retained. Nothing downstream reads any
            # version but the latest, and keeping all of them caused a real
            # "No space left on device" crash on this machine. Delete the
            # PREVIOUS checkpoint only after the new one is safely written,
            # so a mid-save crash never leaves zero checkpoints on disk.
            ckpt_path = self.out_dir / f"global_v{self._version:04d}_{result.name}.pt"
            torch.save(
                {
                    "state_dict": {k: v.clone() for k, v in self._global_sd.items()},
                    "global_classes": self.cfg.global_classes,
                    "version": self._version,
                    "from_node": result.name,
                    "staleness": staleness,
                    "submission": sub,
                },
                ckpt_path,
            )
            if self._last_ckpt_path is not None:
                self._last_ckpt_path.unlink(missing_ok=True)
            self._last_ckpt_path = ckpt_path

            if self._progress is not None:
                self._progress.advance(self._progress_task)
                
            # ── live mAP evaluation ──────────────────────────────────────────
            is_last = (sub == self._total_submissions)
            due     = (sub % self.cfg.federation.eval_interval == 0)
            if due or is_last:
                label = f"sub {sub:04d} v{self._version}"
                sd_snap = {k: v.clone() for k, v in self._global_sd.items()}
                # Evaluation runs outside the lock to avoid blocking pushes
                # from other nodes while val runs. We snapshot the state_dict
                # under the lock above and release before calling eval.
                self._pending_eval = (label, sd_snap)
                
        # ── run evaluation outside the lock ───────────────────────────────
        if hasattr(self, "_pending_eval") and self._pending_eval:
            label, sd_snap = self._pending_eval
            self._pending_eval = None
            console.print(f"\n[bold yellow]⚡ Evaluating at {label}…[/bold yellow]")
            result_eval = _run_eval(self.cfg, sd_snap, label)
            if result_eval:
                with self._lock:
                    self._map_history.append(result_eval)
                    history_snap = list(self._map_history)
                    if self.target_map is not None:
                        metric_val = (
                            result_eval["map50"] if self.target_metric == "map50"
                            else result_eval["map5095"]
                        )
                        if metric_val >= self.target_map:
                            self.target_reached = True
                        if metric_val > self._best_metric + self.min_improvement:
                            self._best_metric = metric_val
                            self._evals_since_improvement = 0
                        else:
                            self._evals_since_improvement += 1
                        if self.patience is not None and self._evals_since_improvement >= self.patience:
                            self.stopped_early = True
                _print_map_table(history_snap, self.cfg.global_classes)

    # ── internal aggregation ──────────────────────────────────────────────

    def _aggregate_async(self, result: NodeRoundResult,
                         pulled_sd: dict, staleness: int) -> None:
        """Delta-based incremental update for one node's submission.

        For shared weights (backbone, neck, box head):
            global += w × (node_trained − pulled_snapshot)

        For per-class weights (cv3.{i}.2):
            global[c] += w × (node_trained[c] − pulled_snapshot[c])
            only for c in result.owned_global_ids

        w = 1 / (1 + staleness_alpha × staleness)
        """
        alpha = self.cfg.federation.staleness_alpha
        w = 1.0 / (1.0 + alpha * staleness)

        per_class_keys = {k for pair in per_class_param_names(self.model) for k in pair}

        with torch.no_grad():
            # Shared layers: full delta
            for k, g_val in self._global_sd.items():
                if k in per_class_keys or not torch.is_floating_point(g_val):
                    continue
                delta = (result.state_dict[k].to(torch.float32)
                         - pulled_sd[k].to(torch.float32))
                self._global_sd[k] = (g_val.to(torch.float32) + w * delta).to(g_val.dtype)

            # Per-class classification head: delta only for owned channels
            for wk, bk in per_class_param_names(self.model):
                gw = self._global_sd[wk].to(torch.float32)
                gb = self._global_sd[bk].to(torch.float32)
                for c in result.owned_global_ids:
                    gw[c] += w * (result.state_dict[wk][c].to(torch.float32)
                                  - pulled_sd[wk][c].to(torch.float32))
                    gb[c] += w * (result.state_dict[bk][c].to(torch.float32)
                                  - pulled_sd[bk][c].to(torch.float32))
                self._global_sd[wk] = gw.to(self._global_sd[wk].dtype)
                self._global_sd[bk] = gb.to(self._global_sd[bk].dtype)
                
    # ── dataset cache pre-warming ─────────────────────────────────────────

    def _prewarm_caches(self) -> None:
        """Build every node's train dataset sequentially BEFORE any thread
        starts. This creates the Ultralytics .cache files one at a time so
        concurrent threads never race to write the same file simultaneously.

        Without this, launching N threads at once causes:
          • N processes all scanning the same large label directory concurrently
          • repeated / corrupted .cache writes
          • a large simultaneous RAM spike (all datasets in memory at once)
          • the OS OOM-killer terminating the process ("Ucciso")
        """
        from .data import build_node_dataset
        log.info("[async] pre-warming dataset caches (sequential, one node at a time)...")
        for node in self.cfg.nodes:
            log.info(f"[async]   caching {node.name} train labels...")
            build_node_dataset(node, self.cfg, split="train")
        log.info("[async] all caches ready — launching node threads")

    # ── per-node thread function ──────────────────────────────────────────

    def _node_loop(self, node: NodeConfig) -> None:
        """Each node thread cycles: [acquire semaphore] → pull → train → push.

        pull() happens INSIDE the semaphore, right before training starts, so
        `staleness` measures genuine drift of the global model during this
        node's own training window. Pulling BEFORE the semaphore (the previous
        ordering) let every node grab a version snapshot the instant it was
        scheduled, then sit queued behind the other nodes -- so by push time it
        was always ~(n_nodes - max_concurrent_nodes) versions stale regardless
        of actual concurrency. At the documented "safe" default of
        max_concurrent_nodes=1 with several nodes, that steady-state staleness
        silently discounted nearly every submission's learning signal via the
        staleness weight (e.g. alpha=0.5, staleness=3 -> w=0.4) even though
        nothing was genuinely running concurrently and true staleness should
        have been 0 throughout.

        The semaphore limits how many nodes load a full model + dataset into
        RAM and run forward/backward simultaneously. With max_concurrent_nodes=1
        (the default) nodes execute one at a time, which is safe on any machine.
        The async semantics (immediate aggregation, staleness tracking) are
        fully preserved even with sequential execution.
        """
        pl_cfg = self.cfg.federation.pseudo_label
        for cycle in range(self.cfg.federation.async_node_cycles):
            if self.target_reached or self.stopped_early:
                break
            # Acquire semaphore before pulling AND before the heavy work (model
            # allocation + training), so the pulled snapshot reflects the global
            # model as of when this node actually starts working, not when it
            # was merely scheduled. Released automatically when the `with`
            # block exits, allowing the next waiting node thread to proceed.
            with self._train_semaphore:
                pulled_sd, pulled_version = self.pull()
                teacher_sd = (
                    pulled_sd
                    if pl_cfg.enabled and cycle >= pl_cfg.start_round
                    else None
                )
                result = run_node_round(
                    node, self.cfg, pulled_sd, cycle, teacher_sd,
                    self._node_optimizer_state.get(node.name),
                )
                self._node_optimizer_state[node.name] = result.optimizer_state
            # push() is called outside the semaphore so aggregation (fast, CPU-only)
            # never blocks other nodes from starting their training.
            self.push(result, pulled_sd, pulled_version)

    # ── orchestration ─────────────────────────────────────────────────────

    def run(self) -> dict:
        log.info(
            f"[async] {len(self.cfg.nodes)} nodes × "
            f"{self.cfg.federation.async_node_cycles} cycles = "
            f"{self._total_submissions} total submissions | "
            f"eval every {self.cfg.federation.eval_interval} submission(s) | "
            f"max {self.cfg.federation.max_concurrent_nodes} node(s) training concurrently"
        )
        
        # Pre-warm caches sequentially before any thread starts.
        # This prevents simultaneous .cache writes and the associated OOM spike.
        self._prewarm_caches()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[green]elapsed[/green]"),
            TimeElapsedColumn(),
            TextColumn("[yellow]left[/yellow]"),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            self._progress = progress
            self._progress_task = progress.add_task(
                "[cyan]async submissions[/cyan]",
                total=self._total_submissions,
            )

            # One thread per node — they run truly concurrently on multi-core
            # machines; on single-core / single-GPU they interleave but the
            # async semantics (no barrier, staleness tracking) still apply.
            with ThreadPoolExecutor(max_workers=len(self.cfg.nodes)) as executor:
                futures = {
                    executor.submit(self._node_loop, node): node.name
                    for node in self.cfg.nodes
                }
                for future in as_completed(futures):
                    node_name = futures[future]
                    try:
                        future.result()
                        log.info(f"[async] node {node_name} thread finished")
                    except Exception as exc:
                        log.error(f"[async] node {node_name} raised: {exc}")
                        raise

        final_sd = {k: v.clone() for k, v in self._global_sd.items()}
        final_path = self.out_dir / "global_final.pt"
        torch.save(
            {"state_dict": final_sd, "global_classes": self.cfg.global_classes,
             "version": self._version},
            final_path,
        )
        if self.target_map is not None:
            reason = (
                "TARGET REACHED" if self.target_reached else
                "stopped early (no improvement -- patience exhausted)" if self.stopped_early else
                "max cycles reached without hitting target"
            )
            log.info(
                f"[async] federation complete | {reason} | "
                f"best {self.target_metric}={self._best_metric:.4f} (target={self.target_map}) | "
                f"submissions={self._submission_count} | -> {final_path}"
            )
        else:
            log.info(
                f"[async] federation complete | "
                f"final global version: {self._version} | -> {final_path}"
            )
        return final_sd
