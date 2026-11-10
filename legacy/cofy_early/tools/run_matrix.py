#!/usr/bin/env python3
"""Resumable orchestrator for the reseed matrix + new-model + new-dataset
runs. Safe to interrupt (Ctrl-C, VS Code close, anything) and safe to
re-invoke with the exact same command at any time -- it always converges
toward "everything currently possible is done" by checking each run's own
summary.json before launching it, rather than tracking its own separate
"what have I done" state that could drift from reality.

    python3 tools/run_matrix.py            # run everything not yet done
    python3 tools/run_matrix.py --dry-run  # show the plan, launch nothing
    python3 tools/run_matrix.py --only rtdetr   # substring filter on session name
    python3 tools/run_matrix.py --list-deformabledetr  # see why those are skipped

Each entry in MATRIX is one cofy.simulate invocation: a config file,
a session name (also used to check completion), and the CLI flags that
invocation needs. "Done" means runs/<output_dir>/<session>/summary.json
exists AND contains real keys for what was asked for (federated_sync or
federated_async, plus centralized/oracle if those were requested) --
matching this project's own established rule of verifying real result
data, not trusting a directory's mere existence or a process exit code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

FEDYOLO_ROOT = Path(__file__).resolve().parent.parent
ORACLE_YAML = "data/kfm_250423_oracle/node_A/data.yaml"
WBC_ORACLE_YAML = "data/wbc_detection_oracle/node_A/data.yaml"

# Minimum free disk space (GB) to keep launching new runs. This project
# has hit real "No space left on device" crashes from unbounded
# checkpoint/run accumulation before -- pause the WHOLE matrix rather
# than let one run fail partway through and leave a corrupt result.
MIN_FREE_GB = 5.0


@dataclass
class Run:
    config: str          # configs/<config>.yaml
    session: str          # --session; also the completion-check name
    mode: str             # sync | async
    output_dir: str        # runs/<output_dir>/<session>/summary.json
    also_centralized: bool = True
    also_oracle: str | None = ORACLE_YAML
    note: str = ""


# Deliberately NOT trying to auto-detect each config's pre-existing seed=0
# result: this project's session naming drifted a lot across a very long
# investigation (some got a "_retry" or "_e2efix" suffix after a real bug
# fix superseded the original; some pre-date the federated_sync/
# federated_async key split and use a bare "federated" key with no
# "oracle" key at all -- found by direct inspection, not assumed, and
# risky to reverse-engineer a general rule for without risking silently
# treating a stale pre-fix result as "done"). Those seed=0 results are
# already correct and already on the leaderboard -- nothing to discover.
# The matrix below owns exactly what's actually new: seeds 1 and 2 for
# every existing config (seed=0 is out of scope, already done, always),
# plus a full 3-seed set for the one leg that never existed at all
# (yolov8m@m480 sync, passed with a distinct `full_backfill=True`).
def _reseed_pair(stem: str, output_dir: str, mode: str, full_backfill: bool = False,
                  seeds: tuple[int, ...] | None = None) -> list[Run]:
    if seeds is None:
        seeds = (0, 1, 2) if full_backfill else (1, 2)
    runs = []
    for seed in seeds:
        suffix = "" if seed == 0 else f"_seed{seed}"
        cfg = f"{stem}{suffix}"
        runs.append(Run(config=cfg, session=cfg, mode=mode, output_dir=output_dir))
    return runs


def build_matrix() -> list[Run]:
    m: list[Run] = []
    od_kfm = "kfm_250423_clean12_federation"
    od_kfm_yolo26 = "kfm_250423_clean12_federation_yolo26"
    od_kfm_yolo26e2e = "kfm_250423_clean12_federation_yolo26e2e"
    od_kfm_rtdetr = "kfm_250423_clean12_federation_rtdetr"
    od_wbc = "wbc_detection_federation"

    # --- reseed matrix (existing dataset, 9 main-table configs -> 3 seeds/mode) ---
    m += _reseed_pair("kfm12_yolo26_sync_l640_20ep", od_kfm_yolo26, "sync")
    m += _reseed_pair("kfm12_yolo26_async_l640_20ep", od_kfm_yolo26, "async")
    
    # --- RT-DETR, existing dataset, 2 scales x both modes -- seed 0 ONLY
    # here, by request: RT-DETR is a brand-new architecture this session,
    # never trained on this dataset before, so its seed 0 result is worth
    # a real look (and, separately, a real eval-path bug was just found
    # and fixed -- see evaluate.py) before spending 2 more multi-hour
    # seeds per config on it blind. seed1/seed2 are queued further down,
    # positioned AFTER this section and the WBC section below -- since
    # run_matrix.py processes its queue strictly in order, one run at a
    # time, that ordering alone guarantees the reseed pairs never launch
    # until every RT-DETR seed0 leg AND every WBC leg has already
    # produced a real summary.json (or been skipped as already done).
    #m += _reseed_pair("kfm12_rtdetr_sync_l640_20ep", od_kfm_rtdetr, "sync", seeds=(0,))
    #m += _reseed_pair("kfm12_rtdetr_async_l640_20ep", od_kfm_rtdetr, "async", seeds=(0,))
    #m += _reseed_pair("kfm12_rtdetr_sync_x640_20ep", od_kfm_rtdetr, "sync", seeds=(0,))
    #m += _reseed_pair("kfm12_rtdetr_async_x640_20ep", od_kfm_rtdetr, "async", seeds=(0,))

    # --- WBC dataset: yolo26-classic@l640, single seed for now (this
    # dataset just landed -- reseed it the same way once these first
    # results are in and worth trusting) ---
    for stem in ["wbc_yolo26classic_sync_l640_20ep"]:#, "wbc_rtdetr_sync_l640_20ep"]:
        m.append(Run(config=stem, session=stem, mode="sync", output_dir=od_wbc, also_oracle=WBC_ORACLE_YAML))
    for stem in ["wbc_yolo26classic_async_l640_20ep"]:#, "wbc_rtdetr_async_l640_20ep"]:
        m.append(Run(config=stem, session=stem, mode="async", output_dir=od_wbc, also_oracle=WBC_ORACLE_YAML))

    # --- RT-DETR reseed pair (seed1, seed2) -- deliberately placed here,
    # after every RT-DETR seed0 leg AND every WBC leg above, per explicit
    # request: only start spending compute on RT-DETR's extra seeds once
    # its first real result (and WBC's) is already in hand. ---
    #m += _reseed_pair("kfm12_rtdetr_sync_l640_20ep", od_kfm_rtdetr, "sync", seeds=(1, 2))
    #m += _reseed_pair("kfm12_rtdetr_async_l640_20ep", od_kfm_rtdetr, "async", seeds=(1, 2))
    #m += _reseed_pair("kfm12_rtdetr_sync_x640_20ep", od_kfm_rtdetr, "sync", seeds=(1, 2))
    #m += _reseed_pair("kfm12_rtdetr_async_x640_20ep", od_kfm_rtdetr, "async", seeds=(1, 2))

    # --- WBC dataset: 3 new config families, seed 0 only, right after the
    # existing WBC l640 yolo26-classic legs above -- these are genuinely
    # NEW work (new arch/scale combos never trained on WBC before), not a
    # reorder of anything pre-existing. Mirrors kfm12's own arch/batch_size
    # convention exactly: yolo26e2e keeps the stock yolo26{l,x}.yaml arch
    # string (ultralytics' own filename-scale guessing), yolo26-classic@x640
    # uses the same configs/arch/yolo26_classic_x.yaml custom-arch path
    # kfm12_yolo26_sync_x640_20ep.yaml already uses (_resolve_arch_cfg()
    # strips the _x suffix and loads yolo26_classic.yaml with scale='x' by
    # hand -- confirmed directly: _resolve_arch_cfg() returns a dict with
    # scale='x', _pretrained_ckpt_name() maps it to yolo26x.pt). ---
    for stem in ["wbc_yolo26e2e_sync_l640_20ep", "wbc_yolo26e2e_sync_x640_20ep",
                 "wbc_yolo26classic_sync_x640_20ep"]:
        m.append(Run(config=stem, session=stem, mode="sync", output_dir=od_wbc, also_oracle=WBC_ORACLE_YAML))
    for stem in ["wbc_yolo26e2e_async_l640_20ep", "wbc_yolo26e2e_async_x640_20ep",
                 "wbc_yolo26classic_async_x640_20ep"]:
        m.append(Run(config=stem, session=stem, mode="async", output_dir=od_wbc, also_oracle=WBC_ORACLE_YAML))

    # --- WBC reseed (seed1, seed2), ALL WBC config families -- placed
    # here, after every WBC seed0 leg above (both the original l640
    # yolo26-classic legs and the 3 new families just added), per explicit
    # request: only start spending compute on WBC's extra seeds once every
    # WBC config's first real result is in hand. Uses the same seed1/seed2
    # config files as everywhere else (only `seed:` differs from the base
    # config). ---
    for stem in ["wbc_yolo26classic_sync_l640_20ep", "wbc_yolo26e2e_sync_l640_20ep",
                 "wbc_yolo26e2e_sync_x640_20ep", "wbc_yolo26classic_sync_x640_20ep"]:
        for seed in (1, 2):
            cfg = f"{stem}_seed{seed}"
            m.append(Run(config=cfg, session=cfg, mode="sync", output_dir=od_wbc, also_oracle=WBC_ORACLE_YAML))
    for stem in ["wbc_yolo26classic_async_l640_20ep", "wbc_yolo26e2e_async_l640_20ep",
                 "wbc_yolo26e2e_async_x640_20ep", "wbc_yolo26classic_async_x640_20ep"]:
        for seed in (1, 2):
            cfg = f"{stem}_seed{seed}"
            m.append(Run(config=cfg, session=cfg, mode="async", output_dir=od_wbc, also_oracle=WBC_ORACLE_YAML))

    # --- reseed matrix (existing dataset, 9 main-table configs -> 3 seeds/mode) ---
    m += _reseed_pair("kfm12_sync_m480_20ep", od_kfm, "sync", seeds=(0,)) # never trained before, so seed0 is worth a look
    m += _reseed_pair("kfm12_sync_l640_20ep", od_kfm, "sync")
    m += _reseed_pair("kfm12_async_l640_20ep", od_kfm, "async")
    m += _reseed_pair("kfm12_yolo26_sync_m640_20ep", od_kfm_yolo26, "sync")
    m += _reseed_pair("kfm12_yolo26_async_m640_20ep", od_kfm_yolo26, "async")
    m += _reseed_pair("kfm12_yolo26_sync_x640_20ep", od_kfm_yolo26, "sync")
    m += _reseed_pair("kfm12_yolo26_async_x640_20ep", od_kfm_yolo26, "async")
    m += _reseed_pair("kfm12_yolo26e2e_sync_l640_20ep", od_kfm_yolo26e2e, "sync")
    m += _reseed_pair("kfm12_yolo26e2e_async_l640_20ep", od_kfm_yolo26e2e, "async")
    m += _reseed_pair("kfm12_yolo26e2e_sync_x640_20ep", od_kfm_yolo26e2e, "sync")
    m += _reseed_pair("kfm12_yolo26e2e_async_x640_20ep", od_kfm_yolo26e2e, "async")
    m += _reseed_pair("kfm12_yolo26e2e_sync_m480_20ep", od_kfm_yolo26e2e, "sync")
    m += _reseed_pair("kfm12_yolo26e2e_async_m480_20ep", od_kfm_yolo26e2e, "async")
    m += _reseed_pair("kfm12_sync_m640_20ep", od_kfm, "sync")
    m += _reseed_pair("kfm12_async_m640_20ep", od_kfm, "async")
    m += _reseed_pair("kfm12_async_m480_20ep", od_kfm, "async")
    m += _reseed_pair("kfm12_sync_m480_20ep", od_kfm, "sync", seeds=(1,2))

    return m


# Deformable DETR: identified and verified feasible (HuggingFace
# transformers, already installed; class_embed.{i}.{weight,bias} follows
# the same "dim 0 = class id" convention everything else here relies on)
# but NOT integrated -- unlike RT-DETR, it isn't an ultralytics model at
# all, so it needs its own data-loading adapter (HF's pixel_values +
# labels=[{"boxes","class_labels"}] COCO-style format, not
# build_node_dataset()'s YOLODataset) on top of its own model-building
# and loss path. Real, larger scope than RT-DETR turned out to need --
# not rushed. Left out of MATRIX entirely so run_matrix.py never claims
# to run something that was never built; see --list-deformabledetr.
DEFORMABLE_DETR_STATUS = (
    "Deformable DETR: verified feasible (transformers v4.41.2 already "
    "installed; DeformableDetrForObjectDetection.from_pretrained(..., "
    "num_labels=N) reshapes cleanly; class_embed.0.{weight,bias} comes "
    "out (N, 256)/(N,), same per-class-indexed-by-dim-0 pattern as "
    "everything else this project's aggregation relies on) but NOT yet "
    "integrated -- needs its own cofy/hf_adapter.py: a COCO-style "
    "data-loading path (HF's pixel_values + labels=[...] format, not "
    "ultralytics' YOLODataset), its own model-building path (not going "
    "through build_model()/DetectionModel at all), and its own eval path "
    "(no YOLO.val() equivalent). Comparable in scope to RT-DETR's own "
    "integration, which needed real, careful work (a wrong head.nl "
    "assumption caught by testing, a from-scratch class-masking design "
    "for a loss with no model.class_weights hook) -- not something to "
    "rush alongside everything else in this batch. Config file naming "
    "convention already reserved (kfm12_deformabledetr_*, "
    "wbc_deformabledetr_*) for whenever this is built."
)


def _find_summary(output_dir: str, session: str) -> Path | None:
    p = FEDYOLO_ROOT / "runs" / output_dir / session / "summary.json"
    return p if p.exists() else None


def is_done(run: Run) -> bool:
    p = _find_summary(run.output_dir, run.session)
    if p is None:
        return False
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    fed_key = f"federated_{run.mode}"
    if fed_key not in data or "map50" not in data.get(fed_key, {}):
        return False
    if run.also_centralized and "map50" not in data.get("centralized", {}):
        return False
    if run.also_oracle and "map50" not in data.get("oracle", {}):
        return False
    return True


def free_disk_gb() -> float:
    usage = shutil.disk_usage(FEDYOLO_ROOT)
    return usage.free / (1024 ** 3)


def preflight_interpreter() -> str | None:
    """Catch a wrong-python-environment problem ONCE, clearly, before it
    can cascade into every queued run failing identically. This is what
    actually happened the first time this script was run for real: it was
    launched from a shell sitting in the (base) conda env (no torch),
    every one of the 53 planned runs died in `import torch` inside its
    own subprocess before doing anything, and the matrix blew through all
    53 in seconds with nothing trained. Returns an error string, or None
    if the interpreter that will actually run each leg looks sane."""
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as e:
        return (
            f"{sys.executable} cannot import {e.name} -- this is the exact "
            f"failure mode that made every run in the matrix die instantly "
            f"before ({e}). Run this script via ./run_matrix.sh (which now "
            f"pins dl_env's own python3), or activate dl_env yourself first."
        )
    return None


def _already_running(session: str) -> str | None:
    """Guard against launching a session that's already running -- this
    happened for real: an earlier run_matrix.py process died (parent
    gone, reparented to PID 1) without its cofy.simulate child dying
    with it, and a later invocation of this script launched the exact
    same session again, running both at once and starving the machine.
    Checks live processes directly (not a lockfile that itself could go
    stale across a crash) via `pgrep -f` on the exact --session flag.
    Uses [[:space:]] rather than \\s for the boundary -- macOS's pgrep is
    BSD regex, not GNU/PCRE, and silently matches nothing with \\s
    (caught by testing directly, not assumed); the boundary itself is
    real and needed since several session names here are literal
    prefixes of others (e.g. kfm12_sync_m480_20ep is a prefix of
    kfm12_sync_m480_20ep_seed1) and a bare substring match would
    false-positive across them."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", f"cofy.simulate .*--session {session}([[:space:]]|$)"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None  # pgrep not available -- don't block on a platform without it
    pids = [p for p in out.stdout.split() if p]
    if pids:
        return ", ".join(pids)
    return None


def launch(run: Run) -> int:
    already = _already_running(run.session)
    if already:
        print(f"=== SKIPPING {run.session}: already running as PID(s) {already} -- "
              f"not launching a duplicate. If that process is actually dead/stuck, "
              f"kill it yourself and re-run this command. ===")
        return -1
    cmd = [
        sys.executable, "-m", "cofy.simulate",
        "--config", f"configs/{run.config}.yaml",
        "--session", run.session,
        "--mode", run.mode,
    ]
    if run.also_centralized:
        cmd.append("--also-centralized")
    if run.also_oracle:
        cmd += ["--also-oracle", run.also_oracle]
    log_path = FEDYOLO_ROOT / "runs" / f"_matrix_log_{run.session}.log"
    print(f"=== launching {run.session} ({run.mode}) -- streaming its log below, "
          f"also written to: {log_path} ===")
    print(f"    (to watch just this run from another terminal: tail -f {log_path})")
    # Stream the current run's own output live instead of only writing it
    # to a file the terminal never shows -- this is "the current run and
    # its log" the matrix is actually working on, not just a terse
    # launching/exited status line either side of an invisible wait.
    #
    # Deliberately raw bytes, NOT text=True line iteration: subprocess's
    # text mode forces universal-newline translation, which silently
    # rewrites the tqdm/ultralytics progress bar's bare '\r' (redraw the
    # same line) into '\n' (new line every tick) -- that's what produced
    # a wall of near-duplicate lines instead of one line updating in
    # place. Passing bytes straight through to sys.stdout.buffer is
    # exactly what a normal foreground run would show, control
    # characters included, so the bar behaves like tqdm because it's
    # literally the same bytes tqdm always wrote.
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=FEDYOLO_ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            logf.write(chunk)
        proc.wait()
    return proc.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    p.add_argument("--only", default=None, help="substring filter on session name")
    p.add_argument("--list-deformabledetr", action="store_true", help="print why Deformable DETR isn't in the matrix")
    args = p.parse_args()

    if args.list_deformabledetr:
        print(DEFORMABLE_DETR_STATUS)
        return

    matrix = build_matrix()
    if args.only:
        matrix = [r for r in matrix if args.only in r.session]

    done = [r for r in matrix if is_done(r)]
    todo = [r for r in matrix if r not in done]

    print(f"=== run_matrix.py: {len(done)}/{len(matrix)} already done, {len(todo)} to run ===")
    if args.dry_run:
        for r in matrix:
            status = "DONE" if r in done else "queued"
            print(f"  [{status:6s}] {r.session}  ({r.mode}, config={r.config}.yaml, out={r.output_dir})")
        print()
        print(DEFORMABLE_DETR_STATUS)
        return

    preflight_err = preflight_interpreter()
    if preflight_err:
        print(f"=== STOPPING before launching anything: {preflight_err} ===")
        return

    consecutive_failures = 0
    for run in todo:
        if is_done(run):  # re-check: another invocation of this script may have just finished it
            print(f"=== {run.session} completed since this run started -- skipping ===")
            continue
        free_gb = free_disk_gb()
        if free_gb < MIN_FREE_GB:
            print(f"=== STOPPING: only {free_gb:.1f}GB free (< {MIN_FREE_GB}GB minimum) -- "
                  f"free up space and re-run this exact command to resume ===")
            return
        rc = launch(run)
        if rc == -1:
            # Already running elsewhere (see _already_running) -- not a
            # failure, just move on; don't let this trip the
            # consecutive-failure circuit breaker.
            continue
        if rc != 0:
            log_path = FEDYOLO_ROOT / "runs" / f"_matrix_log_{run.session}.log"
            consecutive_failures += 1
            print(f"=== {run.session} exited {rc} (non-zero) -- log: {log_path} ===")
            if consecutive_failures >= 2:
                print(f"=== STOPPING: {consecutive_failures} runs in a row failed -- this "
                      f"looks systemic (env/config problem), not a one-off broken leg. "
                      f"Read {log_path}, fix the real cause, then re-run this exact "
                      f"command to resume where it left off. ===")
                return
            print(f"=== continuing to the next queued run -- re-run this exact command "
                  f"later to retry {run.session} too ===")
            continue
        consecutive_failures = 0
        if not is_done(run):
            print(f"=== {run.session} exited 0 but produced no valid summary.json -- "
                  f"treating as incomplete, will retry on next invocation ===")

    remaining = [r for r in matrix if not is_done(r)]
    if remaining:
        print(f"=== {len(remaining)} run(s) still not done -- re-run this exact command to continue ===")
    else:
        print("=== everything in the matrix is done ===")


if __name__ == "__main__":
    main()
