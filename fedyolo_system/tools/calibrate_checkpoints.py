#!/usr/bin/env python3
"""Apply the final calibration round (fedyolo/calibrate.py) to finished runs and
evaluate it, without retraining. Each node re-estimates BatchNorm statistics and
retrains only its own class rows on the frozen final model.

For every checkpoint it evaluates four models on the same held-out eval set:
  before   the run's final model as saved (re-evaluated; must match summary.json)
  bn       BatchNorm statistics re-estimated only
  head     class rows retrained only
  bn+head  both (the full calibration round)

Results: runs/variants/calib/<source>/<ds>/s<seed>/<mode>.json (finished
checkpoints are skipped, so it can be stopped and relaunched).

    python tools/calibrate_checkpoints.py --source primary --modes sync async --seeds 0 1 2
    python tools/calibrate_checkpoints.py --source order_r1 --modes async --seeds 0
    python tools/calibrate_checkpoints.py --source centralized --seeds 0 1 2
"""
import argparse, dataclasses, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASETS = ["kfm12", "wbc", "chestx"]
STAGES = ["before", "bn", "head", "bn+head"]


def log(msg):
    print(time.strftime("%m-%d %H:%M"), msg, flush=True)


def locate(source, ds, seed, mode):
    """(config path, checkpoint path, run dir holding balanced_subsets) for one run."""
    if source == "v2_centralized":   # protocol v2 centralized reference (tools/run_baseline.py --protocol v2)
        run = ROOT / f"runs/variants/v2_central/{ds}/s{seed}"
        return ROOT / f"configs/variants/v2_central/{ds}_s{seed}.yaml", run / "centralized_baseline.pt", run
    if source in ("primary", "centralized"):
        cfg = ROOT / f"configs/study/{ds}_s{seed}.yaml"
        run = ROOT / f"runs/study/{ds}/s{seed}"
        if source == "centralized":
            if not (run / "centralized_baseline.pt").exists():   # a seed the primary study never ran
                run = ROOT / f"runs/variants/_baseline/{ds}/s{seed}"
            return cfg, run / "centralized_baseline.pt", run
        return cfg, run / mode / "global_final.pt", run / mode
    if source == "primary_ext" and (ROOT / f"runs/study/{ds}/s{seed}/{mode}/global_final.pt").exists():
        return locate("primary", ds, seed, mode)   # a seed the primary study itself ran (kfm12 s3)
    run = ROOT / f"runs/variants/{source}/{ds}/s{seed}"
    return ROOT / f"configs/variants/{source}/{ds}_s{seed}.yaml", run / mode / "global_final.pt", run / mode


def contribution_order(run_dir, seed):
    """Asynchronous calibration of a virtual-clock run: when the final model is published every
    node starts its contribution (one forward pass + one class-row epoch, about one training
    epoch of work); contributions arrive in the order they finish on the same clock: data size /
    the node's machine speed (from the run's virtual_schedule.json) x a seeded jitter."""
    import random
    v = json.loads((run_dir / "virtual_schedule.json").read_text())
    j = 0.2
    dur = {nm: v["n_images"][nm] / v["speed"][nm] * (1 + random.Random(f"virtual-cal-{seed}-{nm}").uniform(-j, j))
           for nm in v["speed"]}
    return sorted(dur, key=dur.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="primary, centralized, or a variant name")
    ap.add_argument("--modes", nargs="+", default=["sync"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    if args.source in ("centralized", "v2_centralized"):
        args.modes = ["centralized"]

    import torch
    from fedyolo.calibrate import calibrate, calibrate_progressive
    from fedyolo.config import load_config, load_config_from_dict
    from fedyolo.evaluate import evaluate_state_dict, per_class_map50
    sys.path.insert(0, str(ROOT / "tools"))
    import queue_status as qs
    entry = qs.entry_for(None, args.source) or f"calib_{args.source}"
    STAGES = qs.calib_stages(args.source)

    for seed in args.seeds:
        for ds in args.datasets:
            for mode in args.modes:
                out = ROOT / f"runs/variants/calib/{args.source}/{ds}/s{seed}/{mode}.json"
                res = json.loads(out.read_text()) if out.exists() else {}
                if all(s in res for s in STAGES):
                    log(f"skip {args.source} {ds} s{seed} {mode} (done)")
                    continue
                cfg_path, ckpt, run_dir = locate(args.source, ds, seed, mode)
                if not ckpt.exists():
                    log(f"missing checkpoint {ckpt} -- skipped")
                    continue
                work = out.with_suffix("")
                work.mkdir(parents=True, exist_ok=True)
                if (run_dir / "balanced_subsets").is_dir():     # the exact subsets the run trained on
                    shutil.copytree(run_dir / "balanced_subsets", work / "balanced_subsets", dirs_exist_ok=True)
                if cfg_path.exists():
                    cfg = load_config(cfg_path)
                else:   # a seed with no study config (extra baselines): study seed-0 config with the seed changed,
                    import yaml                       # exactly as tools/run_baseline.py trained it
                    raw = yaml.safe_load(cfg_path.with_name(f"{ds}_s0.yaml").read_text())
                    raw["seed"] = seed
                    cfg = load_config_from_dict(raw)
                cfg = dataclasses.replace(
                    cfg, output_dir=str(work),
                    federation=dataclasses.replace(cfg.federation, mode="async" if mode == "async" else "sync"))
                sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
                t0 = time.time()
                log(f"START {args.source} {ds} s{seed} {mode}")
                bn_sd, acal = None, None
                for stage in STAGES:
                    if stage in res:
                        continue
                    if stage == "before":
                        m = sd
                    elif stage == "bn":
                        m = bn_sd = calibrate(cfg, sd, "bn")
                    elif stage == "head":
                        m = calibrate(cfg, sd, "head")
                    elif stage.startswith("acal"):
                        if acal is None:
                            res["acal_order"] = order = contribution_order(run_dir, seed)
                            log(f"  asynchronous calibration, contributions in order {' '.join(order)}")
                            acal = calibrate_progressive(cfg, sd, order, refresh="acal_refresh" in STAGES)
                        m = (acal[len(order)] if stage == "acal_refresh" else acal[len(order) - 1]
                             if stage == "acal" else acal[int(stage.split("_")[1]) - 1])
                    else:
                        if bn_sd is None:
                            bn_sd = calibrate(cfg, sd, "bn")
                        m = calibrate(cfg, bn_sd, "head")
                    metrics = evaluate_state_dict(cfg, m, name=stage.replace("+", "_"))
                    res[stage] = {"map50": float(metrics.box.map50), "map50-95": float(metrics.box.map),
                                  "per_class_map50": per_class_map50(metrics)}
                    out.write_text(json.dumps(res, indent=1))
                    log(f"  {stage:8s} mAP50 {res[stage]['map50']:.4f}")
                    k = STAGES.index(stage) + 1
                    el = (time.time() - t0) / 60
                    later = len(args.modes) - args.modes.index(mode) - 1     # modes still to do for this dataset/seed
                    left = (len(STAGES) - k) * el / k + later * len(STAGES) * el / k
                    print(f"⏱ calibration {args.source} {ds} s{seed} {mode}: stage {k}/{len(STAGES)} ({stage}) done | "
                          + qs.eta_text((entry, ds, seed), el, left), flush=True)
                shutil.rmtree(work / "eval", ignore_errors=True)
                log(f"DONE  {args.source} {ds} s{seed} {mode} in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
