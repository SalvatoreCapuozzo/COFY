#!/usr/bin/env python3
"""Robustness study: what each attack did to the final model, with and without the defence.

For every robustness run (runs/variants/rb_*/<ds>/s<seed>/<mode>/global_final.pt):
  - mAP50 on the evaluation set, overall, on the attacker's own classes and on the other
    nodes' classes (from summary.json), and Fed:Cent against v2 centralized of the same seed;
  - backdoor runs: attack success rate (ASR) on the triggered evaluation set
    (tools/make_attack_data.py): share of images where the model reports the target class on
    the trigger (confidence >= 0.25, IoU >= 0.5). Also measured on the clean run and its
    defended twin, which never saw the trigger (false-alarm baseline);
  - the defence's own record (robust_log.jsonl): share of uploads clipped, honest and attacker.
Writes runs/variants/robust/<variant>/<ds>/s<seed>.json (skipped if present) and
analysis/ROBUSTNESS.md with --report.

    python tools/robust_eval.py [--report]
"""
import json, re, statistics, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VAR = ROOT / "runs/variants"
OUT = VAR / "robust"
NAME = re.compile(r"^rb_(seq|conc|par)_([a-z]+)(?:_(ora))?(?:_(noenf))?$")
SCHEME_SRC = {"seq": "v2_async", "conc": "v2_vclock", "par": "v2_sync"}   # clean, undefended runs


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def asr(cfg, sd, trig_dir, iou_min=0.5, conf=0.25):
    from ultralytics import YOLO
    from fedyolo.model import build_model
    t = json.loads((trig_dir / "triggers.json").read_text())
    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=False, class_names=cfg.global_classes)
    model.load_state_dict(sd)
    model.names = {i: n for i, n in enumerate(cfg.global_classes)}
    yolo = YOLO("yolov8n.yaml")
    yolo.model = model
    hits = n = 0
    for r in yolo.predict(source=str(trig_dir / "images"), imgsz=cfg.model.imgsz, conf=conf, stream=True,
                          verbose=False, device=cfg.federation.device):
        name = Path(r.path).name
        if name not in t["boxes"]:
            continue
        n += 1
        H, W = r.orig_shape
        x, y, w, h = t["boxes"][name]
        tb = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H
        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            if int(c) != t["target_id"]:
                continue
            ix = max(0, min(b[2], tb[2]) - max(b[0], tb[0])) * max(0, min(b[3], tb[3]) - max(b[1], tb[1]))
            union = (b[2] - b[0]) * (b[3] - b[1]) + (tb[2] - tb[0]) * (tb[3] - tb[1]) - ix
            if union > 0 and ix / union >= iou_min:
                hits += 1
                break
    return hits / n if n else None, n


def evaluate_run(variant, ds, s, mode):
    out = OUT / variant / ds / f"s{s}.json"
    if out.exists():
        return load(out)
    import torch
    from fedyolo.config import load_config
    cfg_path = ROOT / f"configs/variants/{variant}/{ds}_s{s}.yaml"
    run = VAR / f"{variant}/{ds}/s{s}"
    summ = load(run / "summary.json") or {}
    key = "federated_" + mode
    if "map50" not in (summ.get(key) or {}):
        return None
    cfg = load_config(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text())
    rb = (raw["federation"].get("robust") or {})
    attacker = cfg.nodes[cfg.seed % len(cfg.nodes)].name if rb.get("attacker") == "seed" else rb.get("attacker", "")
    m = NAME.match(variant)
    attack = m.group(2) if m else "clean"
    if not attacker and attack in ("labels", "backdoor", "clean"):   # data attacks name the node in data_yaml
        attacker = next((n.name for n in cfg.nodes if "data/attack/" in n.data_yaml), "") or cfg.nodes[s % len(cfg.nodes)].name
    own = set(next(n.owned_classes for n in cfg.nodes if n.name == attacker)) if attacker else set()
    pc = summ[key]["per_class_map50"]
    cent = (load(VAR / f"v2_central/{ds}/s{s}/summary.json") or {}).get("centralized", {})
    res = {"variant": variant, "ds": ds, "seed": s, "attack": attack, "attacker": attacker,
           "map50": summ[key]["map50"], "fed_cent": summ[key]["map50"] / cent["map50"] * 100 if cent else None,
           "map50_attacker_classes": statistics.mean(pc[c] for c in own) if own else None,
           "map50_other_classes": statistics.mean(v for c, v in pc.items() if c not in own)}
    logp = run / mode / "robust_log.jsonl"
    if logp.exists():
        recs = [json.loads(l) for l in logp.read_text().splitlines() if l.strip()]
        for who, flag in (("attacker", True), ("honest", False)):
            rr = [r for r in recs if r.get("attacker") == flag and "shared" in r]
            if rr:
                res[f"clipped_{who}"] = sum(r["shared"]["scale"] < 1 or r["bn"]["scale"] < 1 or bool(r["rows_scaled"])
                                            for r in rr) / len(rr)
    trig = ROOT / f"data/attack/{ds}/_eval_trigger_{attacker}"
    if attack in ("backdoor", "clean") and trig.exists():
        sd = torch.load(run / mode / "global_final.pt", map_location="cpu", weights_only=False)["state_dict"]
        res["asr"], res["asr_images"] = asr(cfg, sd, trig)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def main():
    rows = []
    for vdir in sorted(VAR.glob("rb_*")):
        m = NAME.match(vdir.name)
        if not m:
            continue
        mode = "sync" if m.group(1) == "par" else "async"
        for run in sorted(vdir.glob("*/s*")):
            r = evaluate_run(vdir.name, run.parent.name, int(run.name[1:]), mode)
            if r:
                rows.append(r)
    if "--report" not in sys.argv:
        for r in rows:
            print(json.dumps(r))
        return
    lines = ["# Robustness to a malicious node", "",
             "Generated by `tools/robust_eval.py`. One attacker among 4 nodes (it rotates with the seed). "
             "mAP50 on the evaluation set; 'own' = the attacker's classes, 'others' = everyone else's. "
             "ASR = attack success rate of the backdoor on the triggered evaluation set.", "",
             "| scheme | attack | defence | n | mAP50 | Fed:Cent | own classes | other classes | ASR | uploads clipped (attacker / honest) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    groups = {}
    for r in rows:
        m = NAME.match(r["variant"])
        groups.setdefault((m.group(1), r["attack"], ("ora" if m.group(3) else "none") + (" (ownership off)" if m.group(4) else "")),
                          []).append(r)
    mean = lambda xs: statistics.mean(xs) if xs else None
    fmt = lambda v, f="{:.3f}": "-" if v is None else f.format(v)
    for (sch, att, dfn), rs in sorted(groups.items()):
        lines.append(f"| {sch} | {att} | {dfn} | {len(rs)} | {fmt(mean([r['map50'] for r in rs]))} | "
                     f"{fmt(mean([r['fed_cent'] for r in rs if r['fed_cent'] is not None]), '{:.1f} %')} | "
                     f"{fmt(mean([r['map50_attacker_classes'] for r in rs if r['map50_attacker_classes'] is not None]))} | "
                     f"{fmt(mean([r['map50_other_classes'] for r in rs]))} | "
                     f"{fmt(mean([r['asr'] for r in rs if r.get('asr') is not None]), '{:.0%}')} | "
                     f"{fmt(mean([r['clipped_attacker'] for r in rs if 'clipped_attacker' in r]), '{:.0%}')} / "
                     f"{fmt(mean([r['clipped_honest'] for r in rs if 'clipped_honest' in r]), '{:.0%}')} |")
    (ROOT / "analysis/ROBUSTNESS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
