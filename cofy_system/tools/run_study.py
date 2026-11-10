#!/usr/bin/env python3
"""Resumable, sequential runner for the pre-registered study
(analysis/PREREGISTRATION.md). Safe to kill and re-launch at any time: a run
counts as done only when its summary.json holds all four results.

Order is interleaved by seed (seed 0 of every dataset, then seed 1, ...), so an
interruption leaves balanced data rather than one finished dataset. A dataset
whose configs don't exist yet (ChestX while it downloads) is skipped and
re-checked every 10 minutes. Launch detached so it outlives the shell:

    python3 tools/launch_detached.py runs/study/_runner.log -- python3 tools/run_study.py
"""
import json, shutil, subprocess, sys, time
from pathlib import Path

DATASETS, SEEDS, MIN_FREE_GB = ["kfm12", "wbc", "chestx"], range(5), 5.0
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KEYS = ("federated_sync", "federated_async", "centralized", "oracle")

def log(msg): print(time.strftime("%m-%d %H:%M"), msg, flush=True)

def done(ds, s):
    p = ROOT / f"runs/study/{ds}/s{s}/summary.json"
    try:
        d = json.loads(p.read_text())
        return all(k in d and "map50" in d[k] for k in KEYS)
    except (OSError, json.JSONDecodeError):
        return False

def other_training():
    from cofy.proc import find_processes   # Linux and macOS
    return [str(pid) for pid, _ in find_processes("cofy[.]simulate")]

def main():
    order = [(ds, s) for s in SEEDS for ds in DATASETS]
    fails = 0
    while True:
        pending = [(d, s) for d, s in order if not done(d, s)]
        if not pending:
            log("ALL RUNS DONE"); return
        ready = [(d, s) for d, s in pending if (ROOT / f"configs/study/{d}_s{s}.yaml").exists()]
        if not ready:
            log(f"{len(pending)} pending, none ready (waiting for configs) -- rechecking in 10 min")
            time.sleep(600); continue
        if other_training():
            log("another cofy.simulate is running -- waiting 10 min rather than share the GPU")
            time.sleep(600); continue
        free = shutil.disk_usage(ROOT).free / 1024**3
        if free < MIN_FREE_GB:
            log(f"STOP: only {free:.1f} GB free (< {MIN_FREE_GB}). Free space and relaunch."); return
        ds, s = ready[0]
        cmd = [sys.executable, "-m", "cofy.simulate", "--config", f"configs/study/{ds}_s{s}.yaml",
               "--session", f"s{s}", "--mode", "sync-async", "--also-centralized",
               "--also-oracle", f"data/study/{ds}_oracle/node_A/data.yaml"]
        runlog = ROOT / f"runs/study/_log_{ds}_s{s}.log"
        runlog.parent.mkdir(parents=True, exist_ok=True)
        log(f"START {ds} s{s}  ({len(order)-len(pending)}/{len(order)} done, {free:.0f} GB free)  log: {runlog}")
        t0 = time.time()
        with open(runlog, "wb") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        if done(ds, s):
            log(f"DONE  {ds} s{s} in {mins:.0f} min"); fails = 0
        else:
            fails += 1
            log(f"FAIL  {ds} s{s} rc={rc} after {mins:.0f} min -- see {runlog}")
            if fails >= 2:
                log("STOP: two failures in a row looks systemic; fix and relaunch."); return

if __name__ == "__main__":
    main()
