#!/usr/bin/env python3
"""Training-budget pilot (FOLLOWUP_PLAN §8.4): how many epochs the centralized
baseline needs before its mAP50 stops improving, per dataset.

Trains the centralized baseline (cofy/centralized.py, study config, seed 0)
for up to MAX_EPOCHS epochs and evaluates it every EVERY epochs on the eval set.
The learning rate is constant after warmup, so the first E epochs here are the
same training a run with an E-epoch budget does. Stops early once the best mAP50
has not improved by at least TOL (relative) for PATIENCE epochs.

Budget = the first evaluated epoch whose mAP50 is within TOL (relative) of the
best one. Every method and seed is then trained for that many epochs
(rounds / cycles = budget / local_epochs) in the confirmation at the new budget.

    python tools/budget_pilot.py --seeds 0

Results: runs/variants/_budget/<ds>/s<seed>/curve.json (rewritten after every
evaluation; a pilot interrupted midway restarts from epoch 0).
"""
import argparse, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASETS = ["kfm12", "wbc", "chestx"]
MAX_EPOCHS, EVERY, TOL, PATIENCE = 100, 5, 0.02, 30


def log(msg):
    print(time.strftime("%m-%d %H:%M"), msg, flush=True)


def budget(curve):
    best = max(c["map50"] for c in curve)
    return next(c["epoch"] for c in curve if c["map50"] >= (1 - TOL) * best), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args()
    import yaml
    from cofy.centralized import train_centralized_baseline
    from cofy.config import load_config_from_dict
    from cofy.evaluate import evaluate_state_dict
    sys.path.insert(0, str(ROOT / "tools"))
    import queue_status as qs
    entry = qs.entry_for("budget") or "budget_pilot"

    for s in a.seeds:
        for ds in a.datasets:
            out = ROOT / f"runs/variants/_budget/{ds}/s{s}"
            f = out / "curve.json"
            if f.exists() and "budget_epochs" in json.loads(f.read_text()):
                log(f"skip {ds} s{s} (done)")
                continue
            out.mkdir(parents=True, exist_ok=True)
            study = ROOT / f"configs/study/{ds}_s0.yaml"
            if study.exists():
                raw = yaml.safe_load(study.read_text())
            else:   # a held-out dataset (analysis/CONFIRMATORY_PREREG.md §7): its v2 template, at constant LR like the study's
                raw = yaml.safe_load((ROOT / f"configs/variants/v2_central/{ds}_s0.yaml").read_text())
                raw["federation"]["lr_schedule"] = "constant"
            raw["seed"] = s
            raw["output_dir"] = str(out)
            fed = raw["federation"]
            fed["rounds"] = MAX_EPOCHS // fed["local_epochs"]          # centralized epochs = rounds x local_epochs
            cfg = load_config_from_dict(raw)
            res = {"dataset": ds, "seed": s, "max_epochs": MAX_EPOCHS, "every": EVERY, "tol": TOL,
                   "patience": PATIENCE, "study_budget_epochs": 20, "curve": []}
            t0 = time.time()
            log(f"START {ds} s{s}  (centralized pilot, up to {MAX_EPOCHS} epochs, eval every {EVERY})")

            def on_epoch_end(epoch, model):
                el = (time.time() - t0) / 60
                print(f"⏱ pilot {ds} epoch {epoch}/{MAX_EPOCHS} done (may stop earlier: after {PATIENCE} epochs "
                      f"without a {TOL:.0%} gain) | " + qs.eta_text((entry, ds, s), el, (MAX_EPOCHS - epoch) * el / epoch),
                      flush=True)
                if epoch % EVERY:
                    return False
                sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                m = evaluate_state_dict(cfg, sd, name=f"e{epoch:03d}")
                shutil.rmtree(out / "eval", ignore_errors=True)
                res["curve"].append({"epoch": epoch, "map50": float(m.box.map50), "map50-95": float(m.box.map),
                                     "minutes": round((time.time() - t0) / 60, 1)})
                f.write_text(json.dumps(res, indent=1))
                best = max(res["curve"], key=lambda c: c["map50"])
                log(f"  {ds} epoch {epoch:3d}: mAP50 {res['curve'][-1]['map50']:.4f} (best {best['map50']:.4f} at {best['epoch']})")
                # stop once nothing came within TOL of a new best for PATIENCE epochs
                last_gain = max(c["epoch"] for c in res["curve"]
                                if c["map50"] >= (1 + TOL) * max([0] + [d["map50"] for d in res["curve"] if d["epoch"] < c["epoch"]]))
                return epoch - last_gain >= PATIENCE

            train_centralized_baseline(cfg, on_epoch_end=on_epoch_end)
            res["budget_epochs"], res["best_map50"] = budget(res["curve"])
            res["stopped_at"] = res["curve"][-1]["epoch"]
            f.write_text(json.dumps(res, indent=1))
            log(f"DONE  {ds} s{s} in {(time.time() - t0) / 60:.0f} min  budget {res['budget_epochs']} epochs "
                f"(best mAP50 {res['best_map50']:.4f}; mAP50 at the study's 20 epochs: "
                f"{next((c['map50'] for c in res['curve'] if c['epoch'] == 20), float('nan')):.4f})")


if __name__ == "__main__":
    main()
