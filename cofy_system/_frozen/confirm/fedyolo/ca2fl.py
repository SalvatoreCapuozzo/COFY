"""Class-aware CA2FL aggregation for the async server (opt-in: federation.async_rule: ca2fl).

Buffered asynchronous aggregation (FedBuff, arXiv 2106.06639) with a server-side
cache of every node's latest update used to calibrate each global step (CA2FL,
Wang et al., ICLR 2024), polynomial staleness down-weighting (FedAsync,
arXiv 1903.03934), and per-class-row weighting of the classification output
rows: row c only moves with the nodes that own class c. Ported from the
"class-aware CA2FL" server of docs/FED_YOLO26_ASYNC_SPEC.md (Appendix A),
adapted to fedyolo's state dicts and per_class_param_names().

Why here: with max_concurrent_nodes=1 fedyolo's delta-async trains the nodes
strictly in turn and applies each update in full, so the last node of the
cycle dominates (measured position gradient in per-class AP). The cache keeps
every node's latest direction in every global step.

Per global step, for shared float tensors:
    v = sum_{i in snapshot, age <= max_age} p_i h_i
        + sum_{b in buffer} a_b (m_b / sum m) (delta_b - h_b_old)
with p/m image shares, a_b = (1+s_b)^-exp, h_old = 0 on a node's first arrival.
Class rows use class shares n_{i,c} / sum_j n_{j,c} instead of image shares.
BN running stats: image-weighted mean of every cached node's latest absolute
values. Integer buffers are left untouched.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

_BN_STAT = re.compile(r"\.(running_mean|running_var)$")


@dataclass
class _Cached:
    delta: dict
    bn: dict
    class_counts: torch.Tensor   # [nc]
    n_images: float
    arrived_step: int


class CA2FLAggregator:
    def __init__(self, global_sd: dict, row_keys: set[str], n_nodes: int, buffer_size: int = 0,
                 staleness_exp: float = 0.5, max_staleness: int = 20, cache_max_age: int = 10,
                 server_lr: float = 1.0, use_cache: bool = True):
        self.row_keys = set(row_keys)
        self.float_keys = [k for k, v in global_sd.items()
                           if torch.is_floating_point(v) and not _BN_STAT.search(k)]
        self.bn_keys = [k for k, v in global_sd.items() if _BN_STAT.search(k)]
        self.M = buffer_size if buffer_size > 0 else max(2, math.ceil(n_nodes / 3))
        self.exp, self.max_s, self.max_age, self.lr = staleness_exp, max_staleness, cache_max_age, server_lr
        self.use_cache = use_cache   # False = FedBuff: no cache calibration, buffer average only
        self.steps = 0
        self.cache: dict[str, _Cached] = {}
        self.snapshot: dict[str, _Cached] = {}
        self.buffer: list[tuple] = []   # (node, diff, alpha, record)
        self.log: list[dict] = []

    def push(self, global_sd: dict, node: str, trained_sd: dict, pulled_sd: dict, pulled_step: int,
             class_counts, n_images: float) -> bool:
        """Register one node update; returns True if a global step was applied to global_sd (in place)."""
        s = self.steps - pulled_step
        if s > self.max_s:
            self.log.append(dict(event="discard", node=node, staleness=s, step=self.steps))
            return False
        alpha = (1.0 + s) ** (-self.exp)
        d = {k: trained_sd[k].float() - pulled_sd[k].float() for k in self.float_keys}
        old = self.cache.get(node) if self.use_cache else None
        diff = {k: d[k] - old.delta[k] for k in d} if old is not None else {k: v.clone() for k, v in d.items()}
        rec = _Cached(delta=d, bn={k: trained_sd[k].float().clone() for k in self.bn_keys},
                      class_counts=torch.as_tensor(class_counts, dtype=torch.float32),
                      n_images=float(n_images), arrived_step=self.steps)
        self.cache[node] = rec
        self.buffer.append((node, diff, alpha, rec))
        self.log.append(dict(event="push", node=node, staleness=s, alpha=alpha, step=self.steps))
        if len(self.buffer) >= self.M:
            self.step(global_sd)
            return True
        return False

    @staticmethod
    def _shares(counts: torch.Tensor) -> torch.Tensor:
        """[n, nc] counts -> per-class shares over nodes (0 where nobody has the class)."""
        tot = counts.sum(0, keepdim=True)
        return torch.where(tot > 0, counts / tot.clamp(min=1e-12), torch.zeros_like(counts))

    @staticmethod
    def _rows(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return t * w.view(-1, *([1] * (t.dim() - 1)))

    def step(self, global_sd: dict) -> None:
        """Apply one global step from the current buffer (also used to flush a partial buffer at the end)."""
        if not self.buffer:
            return
        buf = self.buffer
        h_ids = ([i for i in self.snapshot if self.steps - self.snapshot[i].arrived_step <= self.max_age]
                 if self.use_cache else [])
        b_alpha = torch.tensor([a for _, _, a, _ in buf])
        b_img = torch.tensor([r.n_images for _, _, _, r in buf])
        b_w = b_alpha * b_img / b_img.sum().clamp(min=1e-12)
        b_rows = self._shares(torch.stack([r.class_counts for _, _, _, r in buf])) * b_alpha.view(-1, 1)
        if h_ids:
            h_img = torch.tensor([self.snapshot[i].n_images for i in h_ids])
            h_w = h_img / h_img.sum().clamp(min=1e-12)
            h_rows = self._shares(torch.stack([self.snapshot[i].class_counts for i in h_ids]))
        for k in self.float_keys:
            ref = global_sd[k]
            if k in self.row_keys:
                corr = sum(self._rows(diff[k], b_rows[j]) for j, (_, diff, _, _) in enumerate(buf))
                hbar = (sum(self._rows(self.snapshot[i].delta[k], h_rows[j]) for j, i in enumerate(h_ids))
                        if h_ids else 0.0)
            else:
                corr = sum(b_w[j] * diff[k] for j, (_, diff, _, _) in enumerate(buf))
                hbar = sum(h_w[j] * self.snapshot[i].delta[k] for j, i in enumerate(h_ids)) if h_ids else 0.0
            global_sd[k] = (ref.float() + self.lr * (hbar + corr)).to(ref.dtype)
        bn_ids = [i for i in self.cache if self.cache[i].bn]
        if bn_ids and self.bn_keys:
            n = torch.tensor([self.cache[i].n_images for i in bn_ids])
            w = n / n.sum().clamp(min=1e-12)
            for k in self.bn_keys:
                global_sd[k] = sum(w[j] * self.cache[i].bn[k] for j, i in enumerate(bn_ids)).to(global_sd[k].dtype)
        self.snapshot = dict(self.cache)
        self.steps += 1
        self.log.append(dict(event="step", step=self.steps, buffer=[b[0] for b in buf]))
        self.buffer = []
