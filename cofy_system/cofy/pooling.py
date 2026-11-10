"""Build a single pooled YOLO-format dataset directory out of every node's
split, with label files rewritten to use GLOBAL class ids. This lets
evaluation use Ultralytics' own, well-tested val() pipeline unmodified --
the only custom step is this on-disk remap + merge.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .config import FedYoloConfig
from .sampling import cached_balanced_image_subset


def _node_image_label_dirs(node, split: str) -> tuple[Path, Path]:
    data_yaml_path = Path(node.data_yaml)
    local = yaml.safe_load(data_yaml_path.read_text())
    base = Path(local.get("path", data_yaml_path.parent))
    img_dir = base / local[split]
    # standard ultralytics convention: .../images/xxx -> labels live in .../labels/xxx
    label_dir = Path(str(img_dir).replace(f"{os.sep}images", f"{os.sep}labels"))
    return img_dir, label_dir


def materialize_pooled_dataset(
    cfg: FedYoloConfig, split: str, out_dir: str | Path,
    fraction: float | None = None, seed: int = 0,
) -> Path:
    """fraction: same class-stratified "balanced fraction" as
    federation.balanced_fraction (data.py) -- applied here too so a fast
    local run also gets a fast, class-balanced pooled eval set instead of
    always scoring against every node's full validation split. None/1.0 =
    every image, unfiltered (original behavior)."""
    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for node in cfg.nodes:
        img_dir, label_dir = _node_image_label_dirs(node, split)
        id_map = node.class_id_map(cfg.global_classes)
        images = (
            cached_balanced_image_subset(
                img_dir, fraction, seed,
                cache_file=out_dir.parent / "balanced_subsets_pooled" / f"{node.name}_{split}.txt",
            )
            if fraction is not None and fraction < 1.0
            else sorted(Path(img_dir).iterdir())
        )
        for img_path in images:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            label_path = label_dir / (img_path.stem + ".txt")

            dst_name = f"{node.name}__{img_path.name}"
            dst_img = img_out / dst_name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path.resolve())
                except (OSError, NotImplementedError):
                    dst_img.write_bytes(img_path.read_bytes())

            dst_lbl = lbl_out / f"{node.name}__{img_path.stem}.txt"
            lines_out = []
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    local_cls = int(parts[0])
                    global_cls = id_map[local_cls]
                    lines_out.append(" ".join([str(global_cls), *parts[1:]]))
            dst_lbl.write_text("\n".join(lines_out))

    yaml_path = out_dir / "pooled.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images",
                "val": "images",
                "names": {i: n for i, n in enumerate(cfg.global_classes)},
            }
        )
    )
    return yaml_path
