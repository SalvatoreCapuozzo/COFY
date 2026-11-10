#!/usr/bin/env python3
"""Score existing checkpoints on an explicit evaluation set (cfg.eval_data_yaml).

Used to re-measure runs made under the legacy per-node split on a set of
images that no model ever trained on, fully labeled -- without retraining.

    python3 tools/rescore_on_eval.py --config configs/X.yaml --eval-yaml data/_eval/.../data.yaml \
        --out results.json  name1=path/to/a.pt name2=path/to/b.pt ...
"""
import argparse, dataclasses, hashlib, json
from pathlib import Path
import torch
from cofy.config import load_config
from cofy.evaluate import evaluate_state_dict, per_class_map50

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--eval-yaml", required=True)
ap.add_argument("--out", required=True); ap.add_argument("ckpts", nargs="+")
a = ap.parse_args()
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
cfg = dataclasses.replace(load_config(a.config), eval_data_yaml=a.eval_yaml,
                          output_dir=str(out.parent / (out.stem + "_work")))
res = json.loads(out.read_text()) if out.exists() else {}
by_hash = {}
for item in a.ckpts:
    name, path = item.split("=", 1)
    if name in res:
        print(f"{name:28s} (cached) {res[name]['map50']:.4f}"); continue
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    if h in by_hash:                       # bit-identical checkpoint already scored
        res[name] = {**res[by_hash[h]], "same_as": by_hash[h]}
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = evaluate_state_dict(cfg, blob.get("state_dict", blob), name=name)
        res[name] = {"map50": float(m.box.map50), "map50-95": float(m.box.map),
                     "per_class_map50": per_class_map50(m), "ckpt": path}
        by_hash[h] = name
    out.write_text(json.dumps(res, indent=2))
    print(f"{name:28s} {res[name]['map50']:.4f}" + (f"  (bit-identical to {res[name]['same_as']})" if "same_as" in res[name] else ""))
print("DONE")
