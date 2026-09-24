#!/usr/bin/env python3
"""Re-score a finished run's federated / centralized / oracle checkpoints on a
single, leakage-free pooled validation set.

Why this exists
---------------
`tools/split_dataset_into_nodes.py` shuffles and splits train/val
INDEPENDENTLY PER NODE (its own `random.Random(seed).shuffle()` call, once per
node over that node's own images). The oracle dataset is built by the same
script with `--n-nodes 1`, so it shuffles the ENTIRE pooled image set with a
fresh `Random(seed)` call -- producing a train/val boundary with no
relationship to any individual node's boundary over those same images.

`fedyolo.simulate` then scores the oracle checkpoint on the POOLED 4-NODE
validation set. Measured consequence: ~81-83% of the images the oracle is
scored on were also in the oracle's own training split. Federated and
centralized are unaffected (each node's val split is disjoint from that same
node's train split by construction), but every Fed:Oracle ratio computed
against that inflated oracle is wrong.

What this does
--------------
Builds one pooled val set at full size (no balanced-fraction subsample),
removes every image that appears in the oracle's own training split, and
evaluates all three checkpoints on that identical set -- so the corrected
ratios are like-for-like rather than each model being scored on its own pool.

    python3 tools/eval_leakfree.py --run runs/wbc_detection_federation/wbc_yolo26classic_sync_l640_20ep \
                                   --config configs/wbc_yolo26classic_sync_l640_20ep.yaml \
                                   --oracle-images data/wbc_detection_oracle/node_A/images \
                                   --mode sync
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import torch

from fedyolo.config import load_config
from fedyolo.evaluate import evaluate_state_dict, per_class_map50
from fedyolo.pooling import materialize_pooled_dataset


def build_leakfree_pool(cfg, oracle_images: Path, out_dir: Path,
                        drop_leaked: bool = True) -> tuple[Path, int, int]:
    """Full pooled val, minus every image also present in the oracle's train
    split. Returns (yaml path, kept, dropped).

    drop_leaked=False keeps everything -- the full-pool control, used to
    separate the leakage effect (oracle only) from plain eval-set sampling
    noise (which moves federated/centralized too, since neither is leaked)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # fraction=None => every image, not the balanced_fraction subsample the
    # normal eval path uses. A leakage-free set is already small; don't
    # subsample it further.
    pooled_yaml = materialize_pooled_dataset(cfg, split="val", out_dir=out_dir,
                                             fraction=None, seed=cfg.seed)
    if not drop_leaked:
        n = sum(1 for p in (out_dir / "images").iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        return pooled_yaml, n, 0
    oracle_train = {p.name for p in (oracle_images / "train").iterdir()}

    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    kept = dropped = 0
    for img in list(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        # pooled names are "<node>__<original filename>"
        orig = img.name.split("__", 1)[1] if "__" in img.name else img.name
        if orig in oracle_train:
            img.unlink()
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                lbl.unlink()
            dropped += 1
        else:
            kept += 1
    return pooled_yaml, kept, dropped


def load_sd(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.get("state_dict", blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<output_dir>/<session>")
    ap.add_argument("--config", required=True)
    ap.add_argument("--oracle-images", required=True,
                    help="e.g. data/wbc_detection_oracle/node_A/images")
    ap.add_argument("--mode", default="sync", choices=["sync", "async"])
    ap.add_argument("--keep-leaked", action="store_true",
                    help="full-pool control: do not drop oracle-train images")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(args.config)
    # Keep every eval artifact out of the original run directory so a
    # re-score can never be mistaken for, or overwrite, the real result.
    work = run / ("fullpool_eval" if args.keep_leaked else "leakfree_eval")
    cfg = dataclasses.replace(cfg, output_dir=str(work))
    work.mkdir(parents=True, exist_ok=True)

    pooled_yaml, kept, dropped = build_leakfree_pool(
        cfg, Path(args.oracle_images), work / "pooled_val",
        drop_leaked=not args.keep_leaked)
    total = kept + dropped
    print(f"leakage-free pooled val: kept {kept} / {total} "
          f"({100*dropped/max(total,1):.1f}% dropped as oracle-train leakage)")

    targets = {
        f"federated_{args.mode}": run / args.mode / "global_final.pt",
        "centralized":            run / "centralized_baseline.pt",
        "oracle":                 run / "oracle_baseline.pt",
    }
    out = {"_eval": {"kept": kept, "dropped": dropped, "total": total}}
    for name, ckpt in targets.items():
        if not ckpt.exists():
            print(f"  {name}: checkpoint missing ({ckpt}) -- skipped")
            continue
        m = evaluate_state_dict(cfg, load_sd(ckpt), name=f"eval_{name}")
        out[name] = {"map50": float(m.box.map50), "map50-95": float(m.box.map),
                     "per_class_map50": per_class_map50(m)}
        print(f"  {name:18s} mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")

    (work / "summary_rescored.json").write_text(json.dumps(out, indent=2))
    fed = out.get(f"federated_{args.mode}", {}).get("map50")
    cent = out.get("centralized", {}).get("map50")
    orc = out.get("oracle", {}).get("map50")
    if fed and cent:
        print(f"  Fed:Cent   = {100*fed/cent:.1f}%")
    if fed and orc:
        print(f"  Fed:Oracle = {100*fed/orc:.1f}%")
    print(f"written: {work/'summary_rescored.json'}")


if __name__ == "__main__":
    main()
