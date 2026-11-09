"""Optional cross-round pseudo-labeling.

A node never has ground truth for classes it doesn't own, so its loss is
masked to zero for those class channels (see client.py). That stops the node
from teaching the global model to suppress classes it can't see -- but it
also means that node contributes nothing toward learning what those classes
look like.

Once the global model is reasonably trained (after `start_round`), we close
part of that gap with simple output-level self-distillation: the previous
round's global model acts as a frozen teacher, and on the SAME local images
we add a loss that pulls the student's logits for not-owned classes toward
the teacher's, but only at anchor locations where the teacher is confident
(prob > conf_thresh or < 1 - conf_thresh). This requires no box matching or
NMS -- it works directly on the dense anchor grid -- and is weighted down
relative to the real supervised loss via `pseudo_label.weight`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_scores: torch.Tensor,   # (bs, num_anchors, nc) raw logits
    teacher_scores: torch.Tensor,   # (bs, num_anchors, nc) raw logits, no_grad
    unowned_ids: list[int],
    conf_thresh: float,
    anchor_mask: torch.Tensor | None = None,   # (bs, num_anchors) bool; False = skip anchor
    temperature: float = 1.0,                  # >1 softens both sides (FedNTD); loss scaled by T^2
) -> torch.Tensor:
    if not unowned_ids:
        return torch.zeros((), device=student_scores.device)

    s = student_scores[..., unowned_ids]
    t_prob = teacher_scores[..., unowned_ids].detach().sigmoid()
    confident = (t_prob > conf_thresh) | (t_prob < (1.0 - conf_thresh))
    if anchor_mask is not None:
        confident = confident & anchor_mask.unsqueeze(-1)
    n = confident.sum()
    if n.item() == 0:
        return torch.zeros((), device=student_scores.device)

    if temperature != 1.0:
        s = s / temperature
        t_prob = (teacher_scores[..., unowned_ids].detach() / temperature).sigmoid()
    bce = F.binary_cross_entropy_with_logits(s, t_prob, reduction="none")
    return (bce * confident).sum() / n.clamp(min=1) * temperature ** 2
