"""CLI entry point.

    python -m fedyolo.simulate --config configs/your_federation.yaml
    python -m fedyolo.simulate --config configs/your_federation.yaml --mode sync
    python -m fedyolo.simulate --config configs/your_federation.yaml --mode async
    python -m fedyolo.simulate --config configs/your_federation.yaml --also-centralized
    python -m fedyolo.simulate --config configs/your_federation.yaml --also-centralized --also-oracle data/your_oracle_node/data.yaml

--mode controls how many federated trainings run in this one invocation:
    sync-async (default) — train BOTH sync and async from the same config,
        one after another. --also-centralized/--also-oracle, if given, each
        train ONCE and are shared by both (neither centralized.py nor
        oracle.py actually reads federation.mode for anything except
        matching the total local-epoch budget, which rounds*local_epochs
        and async_node_cycles*local_epochs are designed to equal anyway --
        so training them twice, once per sibling sync/async config file,
        was pure redundant compute; this is the fix).
    sync  — only the sync (round-barrier FedAvg) training.
    async — only the async (staleness-discounted delta) training.
--mode always overrides federation.mode in the config file (which still
sets sync-specific/async-specific defaults like rounds/staleness_alpha,
just not which mode(s) actually run).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

# Import rich for terminal UI
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .config import FedYoloConfig, load_config
from .evaluate import evaluate_state_dict, per_class_map50
from .server import AsyncFedServer, FedServer

# 1. Upgrade logging to use RichHandler for beautiful terminal logs
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

log = logging.getLogger("fedyolo.simulate")
console = Console()


def _run_one_federated_mode(cfg: FedYoloConfig, mode: str, args, console, log) -> dict:
    """Train and evaluate ONE federated mode (sync or async), writing its
    checkpoints/live logs into their own `{cfg.output_dir}/{mode}/`
    subfolder so a combined --mode sync-async run's two servers never
    collide on shared filenames (both FedServer and AsyncFedServer write
    `global_final.pt`, for instance). Returns the small metrics dict that
    goes under summary["federated_sync"] / summary["federated_async"].
    """
    sub_cfg = dataclasses.replace(cfg, output_dir=str(Path(cfg.output_dir) / mode),
                                  federation=dataclasses.replace(cfg.federation, mode=mode))
    Path(sub_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if mode == "async":
        mode_detail = (
            f"[bold magenta]ASYNC[/bold magenta] | "
            f"cycles/node={sub_cfg.federation.async_node_cycles} | "
            f"staleness_alpha={sub_cfg.federation.staleness_alpha} | "
            f"total submissions={len(sub_cfg.nodes) * sub_cfg.federation.async_node_cycles}"
        )
        server = AsyncFedServer(sub_cfg)
        log.info("[bold green]🚀 Starting ASYNC Federated Training...[/bold green]")
    else:
        mode_detail = (
            f"[bold blue]SYNC[/bold blue] | "
            f"rounds={sub_cfg.federation.rounds} | "
            f"local_epochs={sub_cfg.federation.local_epochs}"
        )
        server = FedServer(sub_cfg)
        log.info("[bold green]🚀 Starting SYNC Federated Training...[/bold green]")
    console.print(Panel(mode_detail, title=f"[bold white]{mode} federated run[/bold white]"))

    try:
        federated_sd = server.run()
    except Exception:
        if args.auto_cleanup_oom:
            from .gpu_cleanup import cleanup_after_crash
            cleanup_after_crash()
        raise
    log.info(f"[bold green]✔ {mode} federated training complete![/bold green]")

    # The server already evaluated after the last round/submission and printed
    # the live table. Reuse that result instead of re-running a full
    # validation pass.
    if server._map_history:
        final_fed = server._map_history[-1]
        result = {
            "map50":    final_fed["map50"],
            "map50-95": final_fed["map5095"],
            "per_class_map50": final_fed["per_class"],
        }
        log.info(
            f"[bold green]Final {mode} federated mAP50=[/bold green][bold magenta]{final_fed['map50']:.4f}[/bold magenta]  "
            f"[bold green]mAP50-95=[/bold green][bold magenta]{final_fed['map5095']:.4f}[/bold magenta]"
        )
    else:
        # Fallback: eval_interval was set so high that no mid-training eval ran.
        # Re-materializes this mode's own pooled val set under its subfolder --
        # a known, minor redundancy in --mode sync-async (each mode's fallback
        # eval pools separately rather than sharing one cache), not worth the
        # extra plumbing to dedupe given how rarely this fallback path fires.
        with console.status(f"[bold yellow]Evaluating {mode} federated model...", spinner="bouncingBar"):
            fed_metrics = evaluate_state_dict(sub_cfg, federated_sd, name=f"federated_{mode}", verbose=True)
        result = _summarize(fed_metrics)
    return result


def main():
    parser = argparse.ArgumentParser(description="Federated object detection with heterogeneous class sets")
    parser.add_argument("--config", required=True, help="path to federation YAML config")
    parser.add_argument(
        "--mode",
        choices=["sync-async", "sync", "async"],
        default="sync-async",
        help="which federated training(s) to run in this invocation. "
             "'sync-async' (default) trains both from the same config and "
             "shares one --also-centralized/--also-oracle run across both "
             "instead of retraining them per sibling config. Overrides "
             "federation.mode in the config file.",
    )
    parser.add_argument(
        "--also-centralized",
        action="store_true",
        help="also train the non-federated centralized baseline for comparison (slower)",
    )
    parser.add_argument(
        "--also-oracle",
        default=None,
        metavar="ORACLE_DATA_YAML",
        help="also train the true full-annotation oracle baseline (one 'node' with every "
             "class labeled on every image, e.g. data/kfm_250423_oracle/oracle/data.yaml from "
             "`split_dataset_into_nodes.py --n-nodes 1`) -- distinct from --also-centralized, "
             "which still respects each node's own partial labels. Slowest option: this "
             "dataset is much bigger than any one node's.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="name this run's output subfolder (default: auto timestamp) so repeated "
             "runs of the same config never overwrite each other's checkpoints",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="write directly into config.output_dir instead of a session subfolder "
             "(old behaviour — reruns will overwrite previous checkpoints there)",
    )
    parser.add_argument(
        "--auto-cleanup-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="before starting: kill any lingering fedyolo process from a previous "
             "crashed run and clear stale GPU/shm state. If THIS run crashes: do the "
             "same cleanup before exiting, so the next launch doesn't inherit a dirty "
             "state. On by default; use --no-auto-cleanup-oom to disable.",
    )
    args = parser.parse_args()

    if args.auto_cleanup_oom:
        from .gpu_cleanup import preflight_cleanup
        preflight_cleanup()

    cfg = load_config(args.config)

    # ── session directory ────────────────────────────────────────────────────
    # config.output_dir is treated as a FAMILY folder; each invocation writes
    # into its own session subfolder so re-running the same config (or several
    # concurrent runs of it) never clobbers a previous run's checkpoints.
    if not args.no_session:
        base_dir = Path(cfg.output_dir)
        session_name = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = str(base_dir / session_name)
        base_dir.mkdir(parents=True, exist_ok=True)
        latest_link = base_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(session_name, target_is_directory=True)
        except (OSError, NotImplementedError):
            pass  # symlinks unsupported (e.g. some Windows setups) — non-fatal
    log.info(f"[bold cyan]Session output directory:[/bold cyan] {cfg.output_dir}")

    console.print(Panel(
        f"[bold cyan]--mode:[/bold cyan] {args.mode}\n"
        f"[bold cyan]Global classes ({cfg.nc}):[/bold cyan] {cfg.global_classes}\n"
        f"[bold cyan]Nodes:[/bold cyan] {[n.name for n in cfg.nodes]}",
        title="[bold white]DAFYOLO — Federation Setup[/bold white]",
    ))
    for node in cfg.nodes:
        log.info(f"[bold blue]Node {node.name}[/bold blue] owns: {node.owned_classes}")

    # ── federated training: one or both modes, per --mode ───────────────────
    # Neither server class reads cfg.federation.mode internally (confirmed --
    # it's purely a simulate.py-level dispatch decision), so the same cfg's
    # rounds/async_node_cycles/etc. work for either server without cloning.
    # Each mode gets its own output subfolder so a combined sync-async run's
    # two servers never collide on shared checkpoint filenames like
    # global_final.pt.
    modes_to_run = ["sync", "async"] if args.mode == "sync-async" else [args.mode]
    summary: dict = {}
    for mode in modes_to_run:
        summary[f"federated_{mode}"] = _run_one_federated_mode(cfg, mode, args, console, log)
    log.info("[bold green]✔ Evaluation complete![/bold green]")

    if args.also_centralized:
        from .centralized import train_centralized_baseline

        with console.status("[bold magenta]Training centralized baseline (same total local epochs)...", spinner="aesthetic"):
            try:
                central_sd = train_centralized_baseline(cfg)
            except Exception:
                if args.auto_cleanup_oom:
                    from .gpu_cleanup import cleanup_after_crash
                    cleanup_after_crash()
                raise
            torch.save(
                {"state_dict": central_sd, "global_classes": cfg.global_classes},
                Path(cfg.output_dir) / "centralized_baseline.pt",
            )
        log.info("[bold green]✔ Centralized baseline training complete![/bold green]")
        
        with console.status("[bold yellow]Evaluating centralized baseline...", spinner="bouncingBar"):
            central_metrics = evaluate_state_dict(cfg, central_sd, name="centralized")
            summary["centralized"] = _summarize(central_metrics)
        log.info("[bold green]✔ Centralized evaluation complete![/bold green]")

    if args.also_oracle:
        from .oracle import train_oracle_baseline

        with console.status("[bold magenta]Training full-annotation oracle baseline (same total local epochs)...", spinner="aesthetic"):
            try:
                oracle_sd = train_oracle_baseline(cfg, args.also_oracle)
            except Exception:
                if args.auto_cleanup_oom:
                    from .gpu_cleanup import cleanup_after_crash
                    cleanup_after_crash()
                raise
            torch.save(
                {"state_dict": oracle_sd, "global_classes": cfg.global_classes},
                Path(cfg.output_dir) / "oracle_baseline.pt",
            )
        log.info("[bold green]✔ Oracle baseline training complete![/bold green]")

        with console.status("[bold yellow]Evaluating oracle baseline...", spinner="bouncingBar"):
            oracle_metrics = evaluate_state_dict(cfg, oracle_sd, name="oracle")
            summary["oracle"] = _summarize(oracle_metrics)
        log.info("[bold green]✔ Oracle evaluation complete![/bold green]")

    summary_path = Path(cfg.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    log.info(f"Wrote comparison summary -> [bold underline]{summary_path}[/bold underline]")
    
    # 4. Print the final JSON beautifully
    console.print_json(data=summary)


def _summarize(metrics) -> dict:
    return {
        "map50-95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "per_class_map50": per_class_map50(metrics),
    }


if __name__ == "__main__":
    main()
