"""Per-node data loading.

Each node's images/labels live in a standard Ultralytics YOLO-format dataset
(data.yaml with `path`, `train`, `val`, `names`), where label .txt files use
LOCAL class indices (0..k-1 over that node's own owned_classes). We load with
the node's local nc/names so Ultralytics' own label validation runs
correctly against what's actually on disk, then remap every loaded class id
to the GLOBAL class index space before batches ever reach the model/loss.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset

from .config import FedYoloConfig, NodeConfig
from .sampling import cached_balanced_image_subset


class RemappedYOLODataset(YOLODataset):
    """YOLODataset that remaps local class ids -> global class ids after the
    on-disk labels have been validated/cached against the local class list.
    """

    def __init__(self, *args, class_id_map: dict[int, int], **kwargs):
        self.class_id_map = class_id_map
        super().__init__(*args, **kwargs)

    def get_labels(self):
        labels = super().get_labels()
        mapper = np.vectorize(self.class_id_map.get)
        for lb in labels:
            if len(lb["cls"]):
                local_ids = lb["cls"].reshape(-1).astype(int)
                lb["cls"] = mapper(local_ids).reshape(-1, 1).astype(np.float32)
        return labels


def _hyp_namespace(imgsz: int, augment: bool) -> SimpleNamespace:
    # Minimal augmentation hyperparameters YOLODataset/Compose expect to find
    # on `hyp`. Kept conservative since nodes may have very few images.
    return SimpleNamespace(
        imgsz=imgsz,
        task="detect",
        fraction=1.0,
        rect=False,
        cache=None,
        single_cls=False,
        classes=None,
        mask_ratio=4,
        overlap_mask=True,
        mosaic=0.5 if augment else 0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        augmentations=None,
        degrees=0.0,
        translate=0.1 if augment else 0.0,
        scale=0.25 if augment else 0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5 if augment else 0.0,
        bgr=0.0,
        hsv_h=0.015 if augment else 0.0,
        hsv_s=0.4 if augment else 0.0,
        hsv_v=0.4 if augment else 0.0,
        auto_augment=None,
        erasing=0.0,
    )


def build_node_dataset(node: NodeConfig, cfg: FedYoloConfig, split: str = "train") -> RemappedYOLODataset:
    data_yaml_path = Path(node.data_yaml)
    local_data = yaml.safe_load(data_yaml_path.read_text())
    base = Path(local_data.get("path", data_yaml_path.parent))
    img_path = str(base / local_data[split])
    local_names = local_data["names"]
    if isinstance(local_names, dict):
        local_names = [local_names[i] for i in range(len(local_names))]

    if list(local_names) != list(node.owned_classes):
        raise ValueError(
            f"Node '{node.name}': names in {node.data_yaml} ({local_names}) "
            f"must exactly match owned_classes order ({node.owned_classes})"
        )

    balanced_fraction = cfg.federation.balanced_fraction
    if balanced_fraction is not None and balanced_fraction < 1.0:
        manifest = Path(cfg.output_dir) / "balanced_subsets" / f"{node.name}_{split}.txt"
        cached_balanced_image_subset(Path(img_path), balanced_fraction, cfg.seed, manifest)
        img_path = str(manifest)
        fraction = 1.0  # manifest is already the final list -- don't slice it again
    else:
        fraction = cfg.federation.data_fraction

    augment = split == "train"
    dataset = RemappedYOLODataset(
        img_path=img_path,
        imgsz=cfg.model.imgsz,
        batch_size=cfg.federation.batch_size,
        augment=augment,
        hyp=_hyp_namespace(cfg.model.imgsz, augment),
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0 if augment else 0.5,
        prefix=f"{node.name}[{split}]: ",
        task="detect",
        classes=None,
        data={"nc": len(local_names), "names": {i: n for i, n in enumerate(local_names)}, "channels": 3},
        fraction=fraction,
        class_id_map=node.class_id_map(cfg.global_classes),
    )
    return dataset


def build_node_dataloader(node: NodeConfig, cfg: FedYoloConfig, split: str = "train"):
    dataset = build_node_dataset(node, cfg, split)
    return build_dataloader(
        dataset,
        batch=cfg.federation.batch_size,
        workers=cfg.federation.workers,
        shuffle=split == "train",
    )


def per_class_instance_counts(dataset: RemappedYOLODataset, nc: int) -> dict[int, int]:
    counts = {c: 0 for c in range(nc)}
    for lb in dataset.labels:
        for c in lb["cls"].reshape(-1).astype(int).tolist():
            counts[c] = counts.get(c, 0) + 1
    return counts
