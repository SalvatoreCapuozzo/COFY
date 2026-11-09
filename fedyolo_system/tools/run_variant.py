#!/usr/bin/env python3
"""Resumable, sequential runner for a labeled variant of the pre-registered
study (HANDOFF_UBUNTU.md §7). Never touches configs/study/ or runs/study/.

Configs live in configs/variants/<name>/<ds>_s<seed>.yaml, results in
runs/variants/<name>/<ds>/s<seed>/summary.json. Only the federated mode(s)
the variant changes are trained; Fed:Cent is computed against the primary
study's centralized result for the same dataset and seed (same data and seed).
Same order and guards as run_study.py: interleaved by seed, one GPU job at a
time, stop below MIN_FREE_GB or after two consecutive failures.

    python3 tools/launch_detached.py runs/variants/fednova/_runner.log -- \
        python3 tools/run_variant.py --name fednova --mode sync --seeds 0 1 2
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

DATASETS, MIN_FREE_GB = ["kfm12", "wbc", "chestx"], 5.0
ROOT = Path(__file__).resolve().parent.parent

def log(msg): print(time.strftime("%m-%d %H:%M"), msg, flush=True)

def done(name, ds, s, keys):
    p = ROOT / f"runs/variants/{name}/{ds}/s{s}/summary.json"
    if name == "primary_ext":   # extra seeds of the primary study: reuse the ones it already ran (kfm12 s3)
        q = ROOT / f"runs/study/{ds}/s{s}/summary.json"
        if q.exists() and done_file(q, keys):
            return True
    return done_file(p, keys)

def done_file(p, keys):
    try:
        d = json.loads(p.read_text())
        return all(k in d and "map50" in d[k] for k in keys)
    except (OSError, json.JSONDecodeError):
        return False

def overall_position(name, ds, s):
    """' | run N/M of the whole queue' per runs/variants/_plan.txt, or '' if not in the plan."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from queue_status import all_runs, read_plan
        runs = [(r[0], r[2], r[3]) for r in all_runs(read_plan())]
        return f" | run {runs.index((name, ds, s)) + 1}/{len(runs)} of the whole queue"
    except Exception:
        return ""

def prune_duplicate_checkpoints(run_dir: Path) -> float:
    """Delete last-round/last-version checkpoints whose weights equal global_final.pt; returns GB freed.
    Every finished run otherwise keeps two identical ~100 MB copies of its final model."""
    import torch
    freed = 0.0
    for final in run_dir.glob("*/global_final.pt"):
        try:
            a = torch.load(final, map_location="cpu", weights_only=False)["state_dict"]
        except Exception:
            continue
        for other in final.parent.glob("global_*.pt"):
            if other == final:
                continue
            try:
                b = torch.load(other, map_location="cpu", weights_only=False)["state_dict"]
            except Exception:
                continue
            if a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a):
                freed += other.stat().st_size / 1024**3
                other.unlink()
    return freed

def prune_study_checkpoints(name: str, run_dir: Path) -> float:
    """Robustness (rb_*) and competitor (bl_*) runs keep only what their analysis reads: every
    intermediate global model goes; global_final.pt stays where it is still needed (calibration
    of bl_*, backdoor success rate of rb_*backdoor* / rb_*clean*). Returns GB freed."""
    if not name.startswith(("rb_", "bl_", "rbsmoke_")):
        return 0.0
    keep_final = (name.startswith("bl_") or "backdoor" in name or "clean" in name
                  or (name.endswith("_budget") and (run_dir.name == "s0" or "bnpoison" in name)))
    freed = 0.0
    for f in run_dir.glob("*/global_*.pt"):
        if f.name == "global_final.pt" and keep_final:
            continue
        freed += f.stat().st_size / 1024**3
        f.unlink()
    return freed

def point_current_log(target: Path):
    """runs/variants/_current.log -> the log of the run in progress (follow it with `tail -F`)."""
    link = ROOT / "runs/variants/_current.log"
    tmp = link.with_name("._current.log.tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    tmp.replace(link)

def f_append(path: Path):
    return open(path, "ab")

def other_training():
    out = subprocess.run(["pgrep", "-f", "fedyolo[.]simulate"], capture_output=True, text=True).stdout.split()
    return [p for p in out if p]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["sync", "async", "sync-async"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args()
    keys = {"sync": ("federated_sync",), "async": ("federated_async",),
            "sync-async": ("federated_sync", "federated_async")}[a.mode]
    order = [(ds, s) for s in a.seeds for ds in a.datasets]
    fails = 0
    while True:
        pending = [(d, s) for d, s in order if not done(a.name, d, s, keys)]
        if not pending:
            log(f"ALL VARIANT RUNS DONE ({a.name})"); return
        if other_training():
            log("another fedyolo.simulate is running -- waiting 10 min rather than share the GPU")
            time.sleep(600); continue
        free = shutil.disk_usage(ROOT).free / 1024**3
        if free < MIN_FREE_GB:
            log(f"STOP: only {free:.1f} GB free (< {MIN_FREE_GB})."); sys.exit(1)
        ds, s = pending[0]
        cfg = ROOT / f"configs/variants/{a.name}/{ds}_s{s}.yaml"
        cmd = [sys.executable, "-m", "fedyolo.simulate", "--config", str(cfg),
               "--session", f"s{s}", "--mode", a.mode]
        runlog = ROOT / f"runs/variants/{a.name}/_log_{ds}_s{s}.log"
        runlog.parent.mkdir(parents=True, exist_ok=True)
        pos = overall_position(a.name, ds, s)
        log(f"START {ds} s{s}  ({len(order)-len(pending)}/{len(order)} done, {free:.0f} GB free){pos}  log: {runlog}")
        point_current_log(runlog)
        t0 = time.time()
        with open(runlog, "wb") as f:
            f.write((f"#### QUEUE POSITION: variant {a.name}, run {order.index((ds, s)) + 1}/{len(order)} "
                     f"({ds} seed {s}){pos} -- started {time.strftime('%m-%d %H:%M')} ####\n").encode())
            f.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        if done(a.name, ds, s, keys):
            if a.name.endswith("_budget"):   # standard + class-isolated evaluation of the final model (followup_runs.py)
                subprocess.run([sys.executable, str(ROOT / "tools/followup_runs.py"), "--budget-run", a.name, ds, str(s)],
                               cwd=ROOT, stdout=f_append(runlog), stderr=subprocess.STDOUT)
            freed = prune_duplicate_checkpoints(ROOT / f"runs/variants/{a.name}/{ds}/s{s}")
            freed += prune_study_checkpoints(a.name, ROOT / f"runs/variants/{a.name}/{ds}/s{s}")
            log(f"DONE  {ds} s{s} in {mins:.0f} min" + (f" (removed {freed:.2f} GB of checkpoints not needed)" if freed else ""))
            fails = 0
            subprocess.run([sys.executable, str(ROOT / "tools/results_table.py"), "--out",
                            str(ROOT / "runs/variants/_results.txt")], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            fails += 1
            log(f"FAIL  {ds} s{s} rc={rc} after {mins:.0f} min -- see {runlog}")
            if fails >= 2:
                log("STOP: two failures in a row looks systemic."); sys.exit(1)

if __name__ == "__main__":
    main()
