"""Partial-label losses for disjoint class ownership (opt-in).

Baseline behaviour (client.py's `model.class_weights` mask): a node's
classification loss is zeroed for every class it doesn't own, at every
anchor. That also throws away a signal the node does have: an anchor the
assigner matched to one of the node's own labeled boxes holds an object of
exactly that class, so it is certainly NOT any other class.

ExclusionDetectionLoss keeps the baseline mask everywhere except at those
foreground anchors, where not-owned classes are trained as negatives
(the "exclusion loss" of Shi et al., MedIA 2021, applied per anchor).
With `skip` set (teacher-confident not-owned classes), exclusion is not
applied there -- guards against objects of another class overlapping the
labeled box (co-occurring ChestX findings).

It also records the foreground mask of the last call (`last_fg_mask`), which
conditional pseudo-labeling uses to distill only outside the node's own
ground truth (ConDist-style foreground filter).
"""

from __future__ import annotations

import torch
from ultralytics.utils.loss import v8DetectionLoss


class ExclusionDetectionLoss(v8DetectionLoss):
    def __init__(self, model, exclude_ids: list[int]):
        super().__init__(model)
        base = self.class_weights
        self._base_w = base if base is not None else torch.ones(1, 1, self.nc, device=self.device)
        self.exclude_ids = list(exclude_ids)
        self.skip: torch.Tensor | None = None   # (bs, anchors, len(exclude_ids)) bool
        self.last_fg_mask: torch.Tensor | None = None
        assigner = self.assigner

        def hooked_assigner(*args, **kwargs):
            out = assigner(*args, **kwargs)
            # (bs, anchors); float zeros (not bool) when the batch has no boxes at all
            fg = out[3].bool()
            self.last_fg_mask = fg.detach()
            w = self._base_w.expand(fg.shape[0], fg.shape[1], -1).clone()
            if self.exclude_ids:
                add = fg.unsqueeze(-1).to(w.dtype).expand(-1, -1, len(self.exclude_ids))
                if self.skip is not None:
                    add = add * (~self.skip).to(w.dtype)
                w[..., self.exclude_ids] = add
            # consumed by the parent's `bce_loss *= self.class_weights` a few lines later
            self.class_weights = w
            return out

        self.assigner = hooked_assigner
