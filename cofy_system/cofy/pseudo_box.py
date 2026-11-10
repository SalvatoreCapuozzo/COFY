"""Box-level pseudo-labels for classes a node does not own (opt-in:
federation.pseudo_label.enabled + pseudo_label.boxes).

The teacher (the global model the node just pulled) runs on the node's own
training batch; its NMS'd detections of not-owned classes with confidence >=
conf_thresh are appended to the batch as extra ground truth, so they go
through the normal task-aligned assignment and get box, DFL and class loss.
The class mask for not-owned classes is raised from 0 to box_class_weight.
This is docs/FED_YOLO26_ASYNC_SPEC.md §6.3 (and proposal A1): it targets
datasets where other nodes' objects are present but unlabeled on a node's
images (measured: ChestX 53% of boxes hidden from their node, kfm12 22%,
WBC 0%).
"""

from __future__ import annotations

import torch
from ultralytics.utils.nms import non_max_suppression


def append_pseudo_targets(batch: dict, teacher_y: torch.Tensor, unowned_ids: list[int], nc: int,
                          conf: float, iou: float = 0.7, max_det: int = 100) -> int:
    """Append teacher detections of `unowned_ids` to batch['batch_idx'/'cls'/'bboxes'] in place.

    teacher_y: eval-mode head output (bs, 4 + nc, anchors), xywh in input pixels.
    Returns the number of pseudo boxes added.
    """
    h, w = batch["img"].shape[2:]
    dets = non_max_suppression(teacher_y.detach().float(), conf_thres=conf, iou_thres=iou,
                               classes=list(unowned_ids), max_det=max_det, nc=nc)
    idx, cls, box = [], [], []
    for i, d in enumerate(dets):
        if not len(d):
            continue
        d = d.cpu()
        x1, y1, x2, y2 = d[:, 0].clamp(0, w), d[:, 1].clamp(0, h), d[:, 2].clamp(0, w), d[:, 3].clamp(0, h)
        keep = (x2 - x1 > 1) & (y2 - y1 > 1)
        if not keep.any():
            continue
        xywh = torch.stack([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], 1)[keep]
        idx.append(torch.full((len(xywh),), float(i)))
        cls.append(d[keep, 5:6])
        box.append(xywh)
    if not idx:
        return 0
    dev = batch["cls"].device
    b_idx = torch.cat([batch["batch_idx"].float().view(-1), torch.cat(idx).to(dev)])
    b_cls = torch.cat([batch["cls"].float().view(-1, 1), torch.cat(cls).to(dev)])
    b_box = torch.cat([batch["bboxes"].float().view(-1, 4), torch.cat(box).to(dev)])
    order = torch.argsort(b_idx, stable=True)   # ultralytics' target preprocess assumes image-sorted rows
    batch["batch_idx"], batch["cls"], batch["bboxes"] = b_idx[order], b_cls[order], b_box[order]
    return int(sum(len(x) for x in idx))
