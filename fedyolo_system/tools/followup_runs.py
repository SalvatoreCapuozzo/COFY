#!/usr/bin/env python3
"""Follow-up checks for the paper (2026-11-07, exploratory, after the pre-registered study).
Queued as `baseline:<protocol>` plan lines (runs/variants/_queue.sh hands them to
tools/run_baseline.py, which calls this file); each job writes
runs/variants/<entry>/<ds>/s<seed>/summary.json with "complete": true when finished.

  swa         central_swa: the centralized reference trained with the weight averaging of the
              sequential scheme (mean of 4 snapshots over the last 2 epochs), then given the same
              calibration (BatchNorm re-estimation + class-row refit). Does averaging explain why
              the federated schemes score above centralized on WBC?
  local       local_only: each node trains a detector alone on its own data and classes, for the
              same number of epochs it trains in the federation; a class is scored on its owner's
              model (an ensemble that reports each class from its owner). What does federation add?
  budgeteval  rb_budget_clean: class-isolated output budget (fedyolo/evaluate.py nms_override)
              on the clean robustness runs that kept their final model, to measure its cost.

Single run (called by tools/run_variant.py after every *_budget robustness run):
    python tools/followup_runs.py --budget-run rb_seq_signflip_ora_budget wbc 0
"""
import argparse, dataclasses, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
ENTRY = {"swa": "central_swa", "local": "local_only", "budgeteval": "rb_budget_clean"}


def log(msg):
    print(time.strftime("%m-%d %H:%M"), msg, flush=True)


def tick(entry, ds, s, elapsed_min, left_min, what):
    try:
        import queue_status as qs
        print(f"⏱ {entry} {ds} s{s} {what} | " + qs.eta_text((entry, ds, s), elapsed_min, left_min), flush=True)
    except Exception:
        print(f"⏱ {entry} {ds} s{s} {what}", flush=True)


def metrics_dict(m, names, keep=None):
    """mAP50, mAP50-95 and per-class AP50 / AP50-95 of an Ultralytics metrics object."""
    from fedyolo.evaluate import per_class_map50
    pc = per_class_map50(m)
    maps = list(m.box.maps)
    cls = range(len(names)) if keep is None else keep
    return {"map50": float(m.box.map50), "map50-95": float(m.box.map),
            "per_class_map50": {names[c]: float(pc[names[c]]) for c in cls},
            "per_class_map5095": {names[c]: float(maps[c]) for c in cls}}


def central_cfg(ds, s, out):
    """The protocol-v2 centralized configuration of the confirmation, seed `s`, writing to `out`."""
    import yaml
    from fedyolo.config import load_config_from_dict
    raw = yaml.safe_load((ROOT / f"configs/variants/v2_central/{ds}_s0.yaml").read_text())
    raw["seed"] = s
    raw["output_dir"] = str(out)
    return load_config_from_dict(raw)


def run_swa(ds, s, out):
    from fedyolo.calibrate import calibrate
    from fedyolo.centralized import train_centralized_baseline
    from fedyolo.evaluate import evaluate_state_dict
    cfg = central_cfg(ds, s, out)
    cfg = dataclasses.replace(cfg, federation=dataclasses.replace(cfg.federation, mode="sync"))
    total = cfg.federation.rounds * cfg.federation.local_epochs
    t0 = time.time()

    def eta(epoch, model):
        el = (time.time() - t0) / 60
        tick(ENTRY["swa"], ds, s, el, (total - epoch) * el / epoch + 6, f"epoch {epoch}/{total} done")
        return False

    sd = train_centralized_baseline(cfg, on_epoch_end=eta, swa_last=4)
    res = {"before": metrics_dict(evaluate_state_dict(cfg, sd, name="swa_before"), cfg.global_classes)}
    log(f"  averaged, before calibration: mAP50 {res['before']['map50']:.4f}")
    cal = calibrate(cfg, calibrate(cfg, sd, "bn"), "head")
    res["bn+head"] = metrics_dict(evaluate_state_dict(cfg, cal, name="swa_bn_head"), cfg.global_classes)
    log(f"  averaged, calibrated:         mAP50 {res['bn+head']['map50']:.4f}")
    return {"centralized_swa": res}


def run_local(ds, s, out):
    from fedyolo.centralized import train_centralized_baseline
    from fedyolo.evaluate import evaluate_state_dict
    cfg = central_cfg(ds, s, out)
    names = cfg.global_classes
    nodes, per50, per5095, t0 = {}, {}, {}, time.time()
    for k, node in enumerate(cfg.nodes):
        own = node.owned_global_ids(names)
        ncfg = dataclasses.replace(cfg, nodes=[node], output_dir=str(out / node.name))
        log(f"  node {node.name}: trains alone on its own classes {[names[c] for c in own]}")
        sd = train_centralized_baseline(ncfg)
        m = metrics_dict(evaluate_state_dict(ncfg, sd, name=node.name, keep_classes=own), names, keep=own)
        nodes[node.name] = {"classes": [names[c] for c in own],
                            "own_map50": sum(m["per_class_map50"].values()) / len(own)}
        per50.update(m["per_class_map50"])
        per5095.update(m["per_class_map5095"])
        shutil.rmtree(out / node.name / "eval", ignore_errors=True)
        el = (time.time() - t0) / 60
        tick(ENTRY["local"], ds, s, el, el / (k + 1) * (len(cfg.nodes) - k - 1),
             f"node {k + 1}/{len(cfg.nodes)} done")
    res = {"map50": sum(per50.values()) / len(per50), "map50-95": sum(per5095.values()) / len(per5095),
           "per_class_map50": per50, "per_class_map5095": per5095, "nodes": nodes}
    log(f"  local-only ensemble: mAP50 {res['map50']:.4f}")
    return {"local_only": res}


def budget_eval_run(name, ds, s):
    """Standard and class-isolated evaluation of one robustness run's final model -> class_budget.json."""
    import torch
    from fedyolo.config import load_config
    from fedyolo.evaluate import evaluate_state_dict
    run = ROOT / f"runs/variants/{name}/{ds}/s{s}"
    ckpt = next((run / m / "global_final.pt" for m in ("async", "sync") if (run / m / "global_final.pt").exists()), None)
    if ckpt is None:
        log(f"  {name} {ds} s{s}: no final model -- skipped")
        return None
    cfg = load_config(ROOT / f"configs/variants/{name}/{ds}_s{s}.yaml")
    work = run / "_budget_eval"
    cfg = dataclasses.replace(cfg, output_dir=str(work))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    res = {"checkpoint": str(ckpt.relative_to(ROOT))}
    for tag, pcb in (("standard", False), ("class_budget", True)):
        res[tag] = metrics_dict(evaluate_state_dict(cfg, sd, name=tag, per_class_budget=pcb), cfg.global_classes)
        log(f"  {name} {ds} s{s} {tag:12s} mAP50 {res[tag]['map50']:.4f}")
    shutil.rmtree(work, ignore_errors=True)
    (run / "class_budget.json").write_text(json.dumps(res, indent=1))
    return res


def run_budgeteval(ds, s, out):
    done = {}
    for scheme in ("seq", "conc", "par"):
        r = budget_eval_run(f"rb_{scheme}_clean_ora", ds, s)
        if r is not None:
            done[scheme] = {k: r[k]["map50"] for k in ("standard", "class_budget")}
    return {"clean_ora": done}


def run(protocol, seeds, datasets):
    entry = ENTRY[protocol]
    fn = {"swa": run_swa, "local": run_local, "budgeteval": run_budgeteval}[protocol]
    for s in seeds:
        for ds in datasets:
            out = ROOT / f"runs/variants/{entry}/{ds}/s{s}"
            summ = out / "summary.json"
            try:
                if json.loads(summ.read_text()).get("complete"):
                    log(f"skip {entry} {ds} s{s} (done)")
                    continue
            except (OSError, json.JSONDecodeError):
                pass
            if shutil.disk_usage(ROOT).free / 1024**3 < 3:
                log("STOP: less than 3 GB free"); sys.exit(1)
            out.mkdir(parents=True, exist_ok=True)
            log(f"START {entry} {ds} s{s}")
            t0 = time.time()
            res = fn(ds, s, out)
            res["complete"] = True
            shutil.rmtree(out / "eval", ignore_errors=True)
            summ.write_text(json.dumps(res, indent=1))
            log(f"DONE  {entry} {ds} s{s} in {(time.time() - t0) / 60:.0f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=list(ENTRY))
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--datasets", nargs="+", default=["wbc"])
    ap.add_argument("--budget-run", nargs=3, metavar=("NAME", "DS", "SEED"))
    a = ap.parse_args()
    if a.budget_run:
        name, ds, s = a.budget_run
        budget_eval_run(name, ds, int(s))
    else:
        run(a.protocol, a.seeds, a.datasets)


if __name__ == "__main__":
    main()
