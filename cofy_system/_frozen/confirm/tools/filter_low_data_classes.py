"""Derive a new federation (data dirs + config) that drops classes with too
little real data to learn or measure, from an existing federation config.

Exclusion rule (both applied):
  - zero validation instances -> unmeasurable, excluded regardless of train count
  - fewer than --min-train-instances real training instances -> too scarce to
    learn against classes with thousands of instances in the same model

Images stay as symlinks to the original files (no copying); only label .txt
files are rewritten (kept-class lines remapped to new compacted local ids;
images left with zero kept boxes are dropped entirely, not kept as background).

    python tools/filter_low_data_classes.py \\
        --config configs/kfm_250423_fed_260801_full.yaml \\
        --out-data data/kfm_250423_filtered \\
        --config-out configs/kfm_250423_filtered.yaml \\
        --min-train-instances 200 \\
        --model-arch yolov8m.yaml --imgsz 960
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedyolo.config import load_config
from fedyolo.stats import compute_class_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-data", required=True)
    p.add_argument("--config-out", required=True)
    p.add_argument("--min-train-instances", type=int, default=200)
    p.add_argument("--model-arch", default="yolov8m.yaml")
    p.add_argument("--imgsz", type=int, default=960)
    args = p.parse_args()

    cfg = load_config(args.config)
    stats = compute_class_counts(cfg)

    excluded = {
        c for c in cfg.global_classes
        if stats["inst_count"][c]["val"] == 0
        or stats["inst_count"][c]["train"] < args.min_train_instances
    }
    kept_global = [c for c in cfg.global_classes if c not in excluded]

    print(f"[filter] excluding {len(excluded)} classes: {sorted(excluded)}")
    print(f"[filter] keeping {len(kept_global)} classes: {kept_global}")

    out_root = Path(args.out_data)
    nodes_cfg = []

    for node in cfg.nodes:
        new_owned = [c for c in node.owned_classes if c not in excluded]
        if not new_owned:
            print(f"[filter] node {node.name}: no classes survive filtering -- dropping node entirely")
            continue

        new_local_id = {name: i for i, name in enumerate(new_owned)}

        data_yaml = yaml.safe_load(Path(node.data_yaml).read_text())
        base = Path(data_yaml.get("path", Path(node.data_yaml).parent))

        node_dir = out_root / node.name
        kept_imgs = dropped_imgs = 0

        for split in ("train", "val"):
            img_dir = base / data_yaml[split]
            lbl_dir = Path(str(img_dir).replace("images", "labels"))
            img_out = node_dir / "images" / split
            lbl_out = node_dir / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                label_path = lbl_dir / (img_path.stem + ".txt")
                kept_lines = []
                if label_path.exists():
                    for line in label_path.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        old_id = int(parts[0])
                        # old_local_id is keyed by name; invert via node.owned_classes index
                        cls_name = node.owned_classes[old_id]
                        if cls_name in new_local_id:
                            kept_lines.append(f"{new_local_id[cls_name]} {' '.join(parts[1:])}")

                if not kept_lines:
                    dropped_imgs += 1
                    continue

                dst_img = img_out / img_path.name
                if not dst_img.exists():
                    dst_img.symlink_to(img_path.resolve())
                (lbl_out / (img_path.stem + ".txt")).write_text("\n".join(kept_lines) + "\n")
                kept_imgs += 1

        print(f"[filter] node {node.name}: owned {node.owned_classes} -> {new_owned} | "
              f"images kept={kept_imgs} dropped={dropped_imgs}")

        new_data_yaml = {
            "path": str(node_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(new_owned)},
        }
        (node_dir / "data.yaml").write_text(yaml.safe_dump(new_data_yaml))
        nodes_cfg.append({"name": node.name, "data_yaml": str(node_dir / "data.yaml"), "owned_classes": new_owned})

    # Carry over the original federation (and target_run, if present) block
    # verbatim from the source config; caller can hand-tune afterward.
    raw = yaml.safe_load(Path(args.config).read_text())
    full_cfg = {
        "global_classes": kept_global,
        "model": {"arch": args.model_arch, "imgsz": args.imgsz},
        "federation": raw["federation"],
        "nodes": nodes_cfg,
        "output_dir": cfg.output_dir + "_filtered",
        "seed": cfg.seed,
    }
    if "target_run" in raw:
        full_cfg["target_run"] = raw["target_run"]

    Path(args.config_out).write_text(yaml.safe_dump(full_cfg, sort_keys=False))
    print(f"[filter] wrote {args.config_out}")


if __name__ == "__main__":
    main()
