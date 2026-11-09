"""Class geometry of the per-class output rows (opt-in).

Under disjoint class ownership, the row of class c (cv3.{i}.2.weight[c]) is
trained by one node that never sees the other classes, so nothing keeps the
rows of different nodes' classes apart; look-alike classes owned by different
nodes can end up with similar directions.

- ETF head (federation.etf_head): the rows are fixed to a simplex equiangular
  tight frame -- unit vectors at equal, maximal angles -- identical on every
  node (same seed) and never trained; only the features and biases learn.
  FedETF (ICCV 2023), FedGELA (NeurIPS 2023), FedNCA-ML (2025, sigmoid
  multi-label). Rows are rescaled to the norm of the initial rows so initial
  logit magnitudes are unchanged.
- Class spreading (federation.class_spread > 0): FedAwS' spreadout
  regulariser (ICML 2020), moved onto the node: a node pushes its own class
  rows at least an ETF-like cosine distance away from the other classes'
  rows (taken from the downloaded global model, detached). No server step.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import per_class_param_names


def etf_matrix(nc: int, dim: int, seed: int = 0) -> torch.Tensor:
    """(nc, dim) rows forming a simplex ETF: unit norm, pairwise cosine -1/(nc-1)."""
    if dim < nc:
        raise ValueError(f"ETF head needs row dim >= nc ({dim} < {nc})")
    g = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(dim, nc, generator=g))            # orthonormal columns
    center = torch.eye(nc) - torch.full((nc, nc), 1.0 / nc)
    m = math.sqrt(nc / (nc - 1)) * (u @ center)                           # (dim, nc)
    return m.t().contiguous()


def cls_weight_keys(model) -> list[str]:
    return [wk for wk, _ in per_class_param_names(model)]


def apply_etf_head(model, seed: int = 0) -> None:
    """Overwrite every per-class weight tensor with a fixed simplex ETF (in place)."""
    sd = model.state_dict()
    for level, wk in enumerate(cls_weight_keys(model)):
        w = sd[wk]                                                        # (nc, C, 1, 1)
        nc, dim = w.shape[0], w[0].numel()
        scale = w.flatten(1).norm(dim=1).mean().clamp(min=1e-3)
        rows = etf_matrix(nc, dim, seed + level) * scale
        with torch.no_grad():
            w.copy_(rows.view_as(w).to(w.dtype))


def freeze_etf_head(model) -> None:
    keys = set(cls_weight_keys(model))
    for name, p in model.named_parameters():
        if name in keys:
            p.requires_grad_(False)


def spread_penalty(model, owned_ids: list[int], reference_sd: dict, nc: int) -> torch.Tensor:
    """Mean squared hinge on cosine distance between this node's class rows and
    every other class row (others taken detached from reference_sd). Target =
    the ETF cosine distance 1 + 1/(nc-1)."""
    params = dict(model.named_parameters())
    target = 1.0 + 1.0 / max(nc - 1, 1)
    unowned = [c for c in range(nc) if c not in owned_ids]
    total, n = None, 0
    for wk in cls_weight_keys(model):
        w = params[wk].flatten(1)                                          # (nc, C) trainable
        ref = reference_sd[wk].to(w.device, w.dtype).flatten(1)
        mine = F.normalize(w[owned_ids], dim=1)
        others = torch.cat([F.normalize(ref[unowned], dim=1),
                            mine]) if unowned else mine
        dist = 1.0 - mine @ others.t()                                     # (|own|, |others|)
        if unowned:
            k = len(unowned)
            pen_other = F.relu(target - dist[:, :k]).pow(2)
            pen_self = F.relu(target - dist[:, k:]).pow(2)
        else:
            pen_other, pen_self = dist.new_zeros(1), F.relu(target - dist).pow(2)
        off_diag = ~torch.eye(len(owned_ids), dtype=torch.bool, device=w.device)
        terms = torch.cat([pen_other.flatten(), pen_self[off_diag].flatten()])
        total = terms.sum() if total is None else total + terms.sum()
        n += terms.numel()
    return total / max(n, 1)
