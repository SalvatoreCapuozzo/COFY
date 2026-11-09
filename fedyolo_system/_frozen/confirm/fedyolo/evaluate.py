"""Evaluation against a pooled validation set spanning every node, using
Ultralytics' own DetectionValidator (via the standard YOLO.val() entrypoint)
for trusted, standard mAP numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import RTDETR, YOLO

from .config import FedYoloConfig
from .model import _is_rtdetr_arch, build_model
from .pooling import materialize_pooled_dataset


def per_class_map50(metrics) -> dict[str, float]:
    """Per-class AP50 (IoU=0.5), full nc-length with a fallback for classes
    Ultralytics didn't evaluate (no validation instances that round).

    Ultralytics' own `metrics.box.maps` gives this shape for AP50-95, but
    doesn't expose an AP50 equivalent -- `metrics.box.ap50` is only indexed
    by `ap_class_index` (classes that had >=1 val instance), so a class
    outside that dies silently when zipped positionally against every
    class name. Rebuilt here the same way Ultralytics' own `Metric.maps`
    property builds its AP50-95 version: start every class at the OVERALL
    mAP50 (not 0 -- an untested class isn't known to be zero, so filling
    with the mean is Ultralytics' own convention, not a fabricated number),
    then overwrite the classes that were actually measured.
    """
    box = metrics.box
    names = list(metrics.names.values())
    out = np.full(len(names), box.map50, dtype=float)
    for i, c in enumerate(box.ap_class_index):
        out[c] = box.ap50[i]
    return {name: float(v) for name, v in zip(names, out)}


def evaluate_state_dict(cfg: FedYoloConfig, state_dict: dict,
                        name: str = "model", verbose: bool = False):
    """Run validation and return an Ultralytics metrics object.

    verbose=False suppresses all Ultralytics stdout (table, per-class rows,
    speed summary) so the caller can display only what it wants via Rich.
    """
    if cfg.eval_data_yaml:
        # A dedicated fully-labeled eval set: every model is scored on the
        # same images with every class annotated. Never subsampled -- it is
        # small by construction, and it is the number comparisons rest on.
        pooled_yaml = Path(cfg.eval_data_yaml)
    else:
        pooled_yaml = materialize_pooled_dataset(
            cfg, split="val", out_dir=Path(cfg.output_dir) / "pooled_val",
            fraction=cfg.federation.balanced_fraction, seed=cfg.seed,
        )
    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained, class_names=cfg.global_classes)
    model.load_state_dict(state_dict)
    model.names = {i: n for i, n in enumerate(cfg.global_classes)}
    # The wrapper class here is just a task shell for its own .val()
    # machinery -- `.model` gets fully replaced on the next line regardless
    # of what arch it was constructed from. Always use a stock arch (never
    # cfg.model.arch) so this doesn't depend on ultralytics' own filename
    # resolution finding fedyolo's custom, non-stock arch files (which
    # build_model() above already resolved correctly through its own
    # scale-aware loader -- see model.py's _resolve_arch_cfg).
    #
    # The wrapper CLASS matters, not just the arch filename: `YOLO(...)`
    # always dispatches to DetectionValidator regardless of what `.model`
    # gets swapped to afterward, and DetectionValidator cannot decode
    # RTDETRDetectionModel's raw output format -- confirmed for real, not
    # assumed: it silently ran to completion and reported exactly 0.0
    # mAP50 for a pretrained RT-DETR model, every time, rather than
    # erroring. RTDETRValidator is a genuinely different class
    # (ultralytics dispatches on the wrapper type, not `.task`, which
    # reads "detect" for both). Use the RTDETR(...) shell for RT-DETR
    # archs so the correct validator is picked.
    yolo = RTDETR("rtdetr-l.yaml") if _is_rtdetr_arch(cfg.model.arch) else YOLO("yolov8n.yaml")
    yolo.model = model
    metrics = yolo.val(
        data=str(pooled_yaml),
        imgsz=cfg.model.imgsz,
        batch=cfg.federation.batch_size,
        device=cfg.federation.device,
        plots=False,
        verbose=verbose,
        project=str(Path(cfg.output_dir) / "eval"),
        name=name,
    )
    return metrics
