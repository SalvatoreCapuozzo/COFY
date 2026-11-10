"""Alternative CLI entry point (does not touch simulate.py): train the
centralized baseline until it reaches a target mAP, then attempt to reach the
SAME target with the federated (async) approach, to measure whether it can,
and how much longer it takes on an equal footing.

    python -m cofy.simulate_to_target --config configs/kfm_250423_target_map.yaml

Config: everything simulate.py reads (global_classes/model/nodes/federation),
plus an extra top-level `target_run:` block (see TargetRunConfig below) that
only this entry point reads.

Both phases are bounded (never literally indefinite):
  - centralized: capped at `centralized_max_epochs` epochs.
  - federated:   capped at `federated_max_cycles` cycles/node, AND stopped
                 early if `federated_patience` consecutive evaluations pass
                 with less than `min_improvement` gain (plateau detection).
Either phase also stops immediately once its metric >= target_map.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from .client import _class_mask, _read_latest_map
from .config import FedYoloConfig, load_config
from .data import build_node_dataloader
from .model import build_model
from .optim_utils import set_warmup_lr
from .server import AsyncFedServer, _print_map_table, _run_eval
from .stats import print_class_counts

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("cofy.simulate_to_target")
console = Console()


@dataclass
class TargetRunConfig:
    target_map: float = 0.80
    target_metric: str = "map50"          # "map50" | "map50-95"
    eval_every_epochs: int = 1            # centralized: evaluate after every N epochs
    eval_every_submissions: int = 1       # federated: evaluate after every N submissions
    centralized_max_epochs: int = 40      # safety cap, phase 1
    federated_max_cycles: int = 40        # safety cap per node, phase 2
    federated_patience: int = 8           # stop federated early after this many
                                          # evals with no meaningful improvement
    min_improvement: float = 0.002        # smallest change that counts as progress


def load_target_run_config(path: str) -> TargetRunConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return TargetRunConfig(**raw.get("target_run", {}))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: centralized, trained to target
# ─────────────────────────────────────────────────────────────────────────────

def train_centralized_to_target(cfg: FedYoloConfig, trc: TargetRunConfig) -> tuple[dict, dict]:
    device = torch.device(cfg.federation.device)
    # See client.py's run_node_round for why random/np.random are seeded
    # here too, not just torch.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32))
    torch.manual_seed(cfg.seed)

    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained, class_names=cfg.global_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.federation.lr0, momentum=0.9, weight_decay=5e-4)

    node_loaders = [
        (node, node.owned_global_ids(cfg.global_classes), build_node_dataloader(node, cfg, split="train"))
        for node in cfg.nodes
    ]

    history: list[dict] = []
    best_metric = -1.0
    evals_since_improvement = 0
    target_reached = False
    stopped_early = False
    step = 0
    node_pass_count = 0  # increments once per node's pass, not once per epoch --
                         # eval_every_epochs is applied at this finer granularity
                         # (4x more frequent than before, one node = one unit)

    for epoch in range(trc.centralized_max_epochs):
        if target_reached or stopped_early:
            break
        for node, owned_ids, loader in node_loaders:
            model.class_weights = _class_mask(cfg.nc, owned_ids, device)
            # See centralized.py's identical fix: `del`, not `= None` --
            # YOLOEModel.loss() checks hasattr(), which `= None` doesn't clear.
            if hasattr(model, "criterion"):
                del model.criterion
            model.train()

            map_str = _read_latest_map(cfg.output_dir)
            # force_terminal=True: without it, Rich detects stdout isn't a real
            # tty (true whenever this runs redirected to a log file, which is
            # always, given how these long runs are launched) and suppresses
            # live incremental rendering -- the bar silently stops updating the
            # log even though training is genuinely progressing underneath it.
            bar_console = Console(force_terminal=True, width=200)
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), MofNCompleteColumn(),
                TextColumn("[green]Passed:[/green]"), TimeElapsedColumn(),
                TextColumn("[yellow]Left:[/yellow]"), TimeRemainingColumn(),
                transient=False,
                console=bar_console,
            ) as progress:
                task = progress.add_task(
                    f"[magenta]centralized[/magenta] epoch {epoch + 1}/{trc.centralized_max_epochs} {node.name}",
                    total=len(loader),
                )
                for batch in loader:
                    set_warmup_lr(optimizer, step, cfg.federation.warmup_steps, cfg.federation.lr0)
                    step += 1
                    batch["img"] = batch["img"].to(device).float() / 255.0
                    optimizer.zero_grad()
                    loss, _ = model.loss(batch)
                    loss = loss.sum()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.federation.grad_clip)
                    optimizer.step()
                    progress.update(
                        task, advance=1,
                        description=(
                            f"[magenta]centralized[/magenta] epoch {epoch + 1}/{trc.centralized_max_epochs} "
                            f"{node.name} [white]Loss: {loss.item():.4f}[/white]{map_str}"
                        ),
                    )

            node_pass_count += 1
            is_last_pass = (epoch + 1 == trc.centralized_max_epochs) and (node is node_loaders[-1][0])
            due = (node_pass_count % trc.eval_every_epochs == 0) or is_last_pass
            if due:
                sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                label = f"centralized epoch {epoch + 1:03d} {node.name}"
                console.print(f"\n[bold yellow]⚡ Evaluating at {label}…[/bold yellow]")
                result = _run_eval(cfg, sd, label)
                if result:
                    history.append(result)
                    _print_map_table(history, cfg.global_classes)
                    metric_val = result["map50"] if trc.target_metric == "map50" else result["map5095"]
                    if metric_val >= trc.target_map:
                        target_reached = True
                    if metric_val > best_metric + trc.min_improvement:
                        best_metric = metric_val
                        evals_since_improvement = 0
                    else:
                        evals_since_improvement += 1
                    # Incremental checkpoint -- overwritten every eval, so this
                    # never falls more than one eval interval behind. Unlike
                    # the federated phase (which checkpoints every submission),
                    # centralized previously only saved once at full phase
                    # completion -- a crash or interruption at ANY point before
                    # that lost the entire run's progress, weights included,
                    # even after reaching the epoch cap with strong results.
                    torch.save(
                        {
                            "state_dict": sd,
                            "global_classes": cfg.global_classes,
                            "epoch": epoch + 1,
                            "node": node.name,
                            "map50": result["map50"],
                            "map5095": result["map5095"],
                        },
                        Path(cfg.output_dir) / "centralized_latest.pt",
                    )
            if target_reached:
                break

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    reason = "TARGET REACHED" if target_reached else "max epochs reached without hitting target"
    outcome = {
        "phase": "centralized",
        "reason": reason,
        "target_reached": target_reached,
        "epochs_run": epoch + 1,
        "best_metric": best_metric,
        "target_metric": trc.target_metric,
        "target_map": trc.target_map,
        "history": history,
    }
    log.info(f"[magenta]centralized phase done[/magenta] | {reason} | best {trc.target_metric}={best_metric:.4f} "
             f"| epochs={epoch + 1}/{trc.centralized_max_epochs}")
    return sd, outcome


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: federated, attempted to the same target
# ─────────────────────────────────────────────────────────────────────────────

def train_federated_to_target(cfg: FedYoloConfig, trc: TargetRunConfig) -> tuple[dict, dict]:
    cfg.federation.async_node_cycles = trc.federated_max_cycles
    cfg.federation.eval_interval = trc.eval_every_submissions

    server = AsyncFedServer(
        cfg,
        target_map=trc.target_map,
        target_metric=trc.target_metric,
        patience=trc.federated_patience,
        min_improvement=trc.min_improvement,
    )
    sd = server.run()

    reason = (
        "TARGET REACHED" if server.target_reached else
        "stopped early (plateau -- patience exhausted)" if server.stopped_early else
        "max cycles reached without hitting target"
    )
    outcome = {
        "phase": "federated",
        "reason": reason,
        "target_reached": server.target_reached,
        "stopped_early": server.stopped_early,
        "submissions_run": server._submission_count,
        "max_possible_submissions": len(cfg.nodes) * trc.federated_max_cycles,
        "best_metric": server._best_metric,
        "target_metric": trc.target_metric,
        "target_map": trc.target_map,
        "history": server._map_history,
    }
    log.info(f"[cyan]federated phase done[/cyan] | {reason} | best {trc.target_metric}={server._best_metric:.4f} "
             f"| submissions={server._submission_count}/{len(cfg.nodes) * trc.federated_max_cycles}")
    return sd, outcome


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train centralized to a target mAP, then attempt federated to the same target")
    parser.add_argument("--config", required=True)
    parser.add_argument("--session", default=None)
    parser.add_argument("--no-session", action="store_true")
    parser.add_argument(
        "--resume-centralized-from", default=None,
        help="path to a previous session dir containing centralized_final.pt + "
             "centralized_outcome.json -- skip phase 1 entirely and reuse that "
             "result (e.g. after an interrupted run) instead of retraining it. "
             "Must be from a compatible config (same arch/nc) or loading will fail.",
    )
    parser.add_argument(
        "--federated-only", action="store_true",
        help="skip phase 1 (centralized) entirely -- no prior result needed. "
             "Runs only the federated phase, starting from a fresh model. No "
             "centralized-vs-federated comparison is produced in this mode.",
    )
    parser.add_argument(
        "--auto-cleanup-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="before starting: kill any lingering cofy process from a previous "
             "crashed run and clear stale GPU/shm state. If THIS run crashes: do the "
             "same cleanup before exiting. On by default; use --no-auto-cleanup-oom "
             "to disable.",
    )
    args = parser.parse_args()

    if args.auto_cleanup_oom:
        from .gpu_cleanup import preflight_cleanup
        preflight_cleanup()

    cfg = load_config(args.config)
    trc = load_target_run_config(args.config)

    base_dir = Path(cfg.output_dir)
    if not args.no_session:
        session_name = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = base_dir / session_name
        base_dir.mkdir(parents=True, exist_ok=True)
        latest_link = base_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(session_name, target_is_directory=True)
        except (OSError, NotImplementedError):
            pass
    else:
        session_dir = base_dir
    session_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold cyan]Target:[/bold cyan] {trc.target_metric} >= {trc.target_map}\n"
        f"[bold cyan]Centralized cap:[/bold cyan] {trc.centralized_max_epochs} epochs\n"
        f"[bold cyan]Federated cap:[/bold cyan] {trc.federated_max_cycles} cycles/node, "
        f"patience={trc.federated_patience} evals\n"
        f"[bold cyan]Global classes ({cfg.nc}):[/bold cyan] {cfg.global_classes}\n"
        f"[bold cyan]Session:[/bold cyan] {session_dir}",
        title="[bold white]DAFYOLO — Train-to-Target[/bold white]",
    ))

    # ── report images/objects per class before starting, as requested ──────
    stats = print_class_counts(cfg, console)
    (session_dir / "class_counts.json").write_text(json.dumps(stats, indent=2))

    # ── phase 1: centralized (skipped entirely if --federated-only) ────────
    central_sd = central_outcome = None
    if args.federated_only:
        log.info("[bold yellow]Phase 1/2: SKIPPED (--federated-only)[/bold yellow]")
    elif args.resume_centralized_from:
        prev = Path(args.resume_centralized_from)
        final_ckpt = prev / "centralized_final.pt"
        latest_ckpt = prev / "centralized" / "centralized_latest.pt"
        if final_ckpt.exists():
            log.info(f"[bold green]Phase 1/2: reusing COMPLETED centralized result from {prev}[/bold green]")
            central_sd = torch.load(final_ckpt, map_location="cpu")["state_dict"]
            central_outcome = json.loads((prev / "centralized_outcome.json").read_text())
        elif latest_ckpt.exists():
            # Interrupted run, never reached its final save -- recover from
            # the incremental checkpoint instead (at most one eval interval
            # behind wherever it actually stopped).
            ckpt = torch.load(latest_ckpt, map_location="cpu")
            central_sd = ckpt["state_dict"]
            central_outcome = {
                "phase": "centralized",
                "reason": f"RECOVERED from interrupted run (last checkpoint: epoch {ckpt['epoch']}, node {ckpt['node']})",
                "target_reached": False,
                "epochs_run": ckpt["epoch"],
                "best_metric": ckpt["map50"] if trc.target_metric == "map50" else ckpt["map5095"],
                "target_metric": trc.target_metric,
                "target_map": trc.target_map,
                "history": [],
            }
            log.warning(
                f"[bold yellow]Phase 1/2: {prev} never finished -- recovering from its last "
                f"incremental checkpoint (epoch {ckpt['epoch']}, node {ckpt['node']}, "
                f"map50={ckpt['map50']:.4f}) instead of a fully-completed result[/bold yellow]"
            )
        else:
            raise FileNotFoundError(
                f"no centralized_final.pt or centralized/centralized_latest.pt found under {prev}"
            )
        torch.save({"state_dict": central_sd, "global_classes": cfg.global_classes}, session_dir / "centralized_final.pt")
        (session_dir / "centralized_outcome.json").write_text(json.dumps(central_outcome, indent=2))
        log.info(
            f"[magenta]centralized (reused)[/magenta] | {central_outcome['reason']} | "
            f"best {trc.target_metric}={central_outcome['best_metric']:.4f} | "
            f"epochs={central_outcome['epochs_run']}"
        )
    else:
        cfg.output_dir = str(session_dir / "centralized")
        log.info("[bold green]Phase 1/2: centralized -> target[/bold green]")
        try:
            central_sd, central_outcome = train_centralized_to_target(cfg, trc)
        except Exception:
            if args.auto_cleanup_oom:
                from .gpu_cleanup import cleanup_after_crash
                cleanup_after_crash()
            raise
        torch.save({"state_dict": central_sd, "global_classes": cfg.global_classes}, session_dir / "centralized_final.pt")
        (session_dir / "centralized_outcome.json").write_text(json.dumps(central_outcome, indent=2))

    # ── phase 2: federated ──────────────────────────────────────────────────
    cfg.output_dir = str(session_dir / "federated")
    log.info("[bold green]Phase 2/2: federated -> same target[/bold green]")
    try:
        fed_sd, fed_outcome = train_federated_to_target(cfg, trc)
    except Exception:
        if args.auto_cleanup_oom:
            from .gpu_cleanup import cleanup_after_crash
            cleanup_after_crash()
        raise
    (session_dir / "federated_outcome.json").write_text(json.dumps(fed_outcome, indent=2))

    # ── final comparison ────────────────────────────────────────────────────
    summary = {"centralized": central_outcome, "federated": fed_outcome}
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    central_line = (
        f"[bold]Centralized[/bold]: {central_outcome['reason']} | "
        f"best {trc.target_metric}={central_outcome['best_metric']:.4f} | "
        f"epochs={central_outcome['epochs_run']}\n"
        if central_outcome is not None else
        "[bold]Centralized[/bold]: skipped (--federated-only)\n"
    )
    console.print(Panel(
        central_line +
        f"[bold]Federated[/bold]: {fed_outcome['reason']} | "
        f"best {trc.target_metric}={fed_outcome['best_metric']:.4f} | "
        f"submissions={fed_outcome['submissions_run']}/{fed_outcome['max_possible_submissions']}",
        title="[bold white]Train-to-Target — Final Result[/bold white]",
    ))
    log.info(f"Wrote summary -> {session_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
