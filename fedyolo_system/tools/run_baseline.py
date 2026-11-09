#!/usr/bin/env python3
"""Centralized baseline for seeds the primary study never ran, so variants given
extra seeds (tools/search_rules.py) have a Fed:Cent denominator for every seed.
Same procedure as the primary study's (fedyolo/centralized.py, study config with
only the seed changed); oracle is not trained (Fed:Oracle uses the seeds that
have one). Seeds already in runs/study are skipped.

    python tools/run_baseline.py --seeds 3 4 5

Results: runs/variants/_baseline/<ds>/s<seed>/summary.json ("centralized").
"""
import argparse, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from results_table import DATASETS, reference


def log(msg):
    print(time.strftime("%m-%d %H:%M"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--protocol", default="", help="'v2': LR decay + plateau budget (configs/variants/v2_central); "
                    "'swa', 'local', 'budgeteval': follow-up checks, see tools/followup_runs.py")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args()
    if a.protocol in ("swa", "local", "budgeteval"):   # follow-up checks for the paper (tools/followup_runs.py)
        import followup_runs
        followup_runs.run(a.protocol, a.seeds, a.datasets)
        return
    row = "v2_x" if a.protocol == "v2" else ""
    import dataclasses, torch, yaml
    from fedyolo.centralized import train_centralized_baseline
    from fedyolo.config import load_config_from_dict
    from fedyolo.evaluate import evaluate_state_dict, per_class_map50
    import queue_status as qs
    entry = qs.entry_for("baseline:v2" if a.protocol == "v2" else "baseline") or "baseline_ext"

    for s in a.seeds:
        for ds in a.datasets:
            if reference(ds, s, row=row) is not None:
                log(f"skip {ds} s{s} (centralized already available)")
                continue
            if shutil.disk_usage(ROOT).free / 1024**3 < 5:
                log("STOP: less than 5 GB free"); sys.exit(1)
            src = (ROOT / f"configs/variants/v2_central/{ds}_s0.yaml" if a.protocol == "v2"
                   else ROOT / f"configs/study/{ds}_s0.yaml")
            raw = yaml.safe_load(src.read_text())
            raw["seed"] = s
            out = ROOT / (f"runs/variants/v2_central/{ds}/s{s}" if a.protocol == "v2" else f"runs/variants/_baseline/{ds}/s{s}")
            out.mkdir(parents=True, exist_ok=True)
            raw["output_dir"] = str(out)
            cfg = load_config_from_dict(raw)
            log(f"START {ds} s{s}  (centralized baseline)")
            t0 = time.time()
            total = cfg.federation.rounds * cfg.federation.local_epochs

            def eta(epoch, model, ds=ds, s=s, t0=t0, total=total):
                el = (time.time() - t0) / 60
                print(f"⏱ baseline {ds} s{s} epoch {epoch}/{total} done | "
                      + qs.eta_text((entry, ds, s), el, (total - epoch) * el / epoch + 1), flush=True)
                return False

            sd = train_centralized_baseline(cfg, on_epoch_end=eta)
            torch.save({"state_dict": sd, "global_classes": cfg.global_classes}, out / "centralized_baseline.pt")
            m = evaluate_state_dict(cfg, sd, name="centralized")
            res = {"centralized": {"map50-95": float(m.box.map), "map50": float(m.box.map50),
                                   "per_class_map50": per_class_map50(m)}}
            (out / "summary.json").write_text(json.dumps(res, indent=1))
            shutil.rmtree(out / "eval", ignore_errors=True)
            log(f"DONE  {ds} s{s} in {(time.time() - t0) / 60:.0f} min  mAP50 {res['centralized']['map50']:.4f}")


if __name__ == "__main__":
    main()
