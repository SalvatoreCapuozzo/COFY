"""Final calibration round (opt-in: `federation.final_calibration`).

After the last aggregation every node, starting from the final global model:

1. re-estimates the BatchNorm statistics on its own training images (forward
   passes only, labels unused); the server pools them, weighted by batch count;
2. with the pooled statistics in place and every other weight frozen, retrains
   only its own class rows (`cv3.{i}.2` rows of the classes it owns) on its own
   labels, with the usual class mask; the server writes each node's rows into
   the global model. Each class has one owner, so nothing is averaged.

Why: the class rows of different nodes were trained against different versions
of the shared layers (sync: each node's local copy before averaging; async: the
version current when the owner last trained), so after the final aggregation
they do not match the shared layers they are used with. Retraining each row on
the final, frozen features aligns all of them to the same features, and
removes the advantage of the node that trained last. The BatchNorm statistics
of an averaged model are re-estimated for the same reason (the averaged
weights were never trained, so neither were their statistics).

Sources: classifier re-training on frozen features (Kang et al., ICLR 2020;
CCVR, NeurIPS 2021, which does it on the server: here it is done by the nodes,
so the server still holds no data and trains nothing); local classifier
fine-tuning after aggregation (layer-peeled analysis, arXiv 2502.03231, 2025);
BatchNorm re-estimation after weight averaging (SWA, UAI 2018).
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from ultralytics.utils.loss import v8DetectionLoss

from .client import _class_mask
from .config import FedYoloConfig, NodeConfig
from .data import build_node_dataloader
from .model import build_model, per_class_param_names, tower_groups
from .partial_label import ExclusionDetectionLoss

log = logging.getLogger("cofy.calibrate")


def _load(cfg: FedYoloConfig, state_dict: dict, device):
    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz,
                        pretrained=cfg.model.pretrained, class_names=cfg.global_classes,
                        towers=tower_groups(cfg)).to(device)
    model.load_state_dict(state_dict)
    return model


def node_bn_stats(node: NodeConfig, cfg: FedYoloConfig, state_dict: dict, device) -> tuple[dict, int]:
    """Node side, step 1: BatchNorm mean/var of the given weights on this node's images."""
    model = _load(cfg, state_dict, device)
    bns = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
    for m in bns.values():
        m.reset_running_stats()
        m.momentum = None          # cumulative average over all batches
    model.train()
    n_batches = 0
    with torch.no_grad():
        for batch in build_node_dataloader(node, cfg, split="train"):
            model.forward(batch["img"].to(device).float() / 255.0)
            n_batches += 1
    stats = {n: (m.running_mean.detach().cpu().clone(), m.running_var.detach().cpu().clone())
             for n, m in bns.items()}
    return stats, n_batches


def _bn_contribution(node: NodeConfig, cfg: FedYoloConfig, state_dict: dict, device) -> tuple[dict, int]:
    """What a node sends in step 1 of the simulated calibration: its honest statistics, or, for
    the attacker with federation.robust.calib_attack, poisoned ones (cofy/robust.py)."""
    s, n = node_bn_stats(node, cfg, state_dict, device)
    rc = cfg.federation.robust
    if rc.calib_attack:
        from .robust import attacker_name, poison_bn_stats
        if node.name == attacker_name(cfg):
            s = poison_bn_stats(s, rc.attack_factor)
    return s, n


def pool_bn_stats(state_dict: dict, node_stats: list[tuple[dict, int]], rule: str = "mean") -> dict:
    """Server side, step 1: pooled mean and variance into a copy of state_dict.
    rule "mean": count-weighted pooled moments (the registered rule).
    rule "median": coordinate-wise median of the entries' means and of their variances, unweighted
    (batch counts are reported by the nodes and are not trusted). With one poisoned entry among
    four, every pooled value stays within the range of the honest ones."""
    out = {k: v.clone() for k, v in state_dict.items()}
    if rule == "median":
        for name in node_stats[0][0]:
            for i, suffix in ((0, "running_mean"), (1, "running_var")):
                q = torch.quantile(torch.stack([s[name][i].float() for s, _ in node_stats]), 0.5, dim=0)
                out[f"{name}.{suffix}"] = q.clamp_min(0).to(out[f"{name}.{suffix}"].dtype) if i else \
                    q.to(out[f"{name}.{suffix}"].dtype)
        return out
    if rule != "mean":
        raise ValueError(f"unknown BatchNorm pooling rule {rule!r}")
    total = sum(n for _, n in node_stats)
    for name in node_stats[0][0]:
        mean = sum(n * s[name][0] for s, n in node_stats) / total
        second = sum(n * (s[name][1] + s[name][0] ** 2) for s, n in node_stats) / total
        out[f"{name}.running_mean"] = mean.to(out[f"{name}.running_mean"].dtype)
        out[f"{name}.running_var"] = (second - mean ** 2).clamp_min(0).to(out[f"{name}.running_var"].dtype)
    return out


def node_head_align(node: NodeConfig, cfg: FedYoloConfig, state_dict: dict, device,
                    epochs: int) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Node side, step 2: retrain this node's class rows on the frozen model.
    Returns ({param name: rows of the owned classes}, owned class ids)."""
    owned = node.owned_global_ids(cfg.global_classes)
    unowned = [c for c in range(cfg.nc) if c not in owned]
    model = _load(cfg, state_dict, device)
    model.class_weights = _class_mask(cfg.nc, owned, device)
    criterion = (ExclusionDetectionLoss(model, unowned) if cfg.federation.exclusion_loss
                 else v8DetectionLoss(model))
    pairs = per_class_param_names(model)
    params = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    train = [bk for _, bk in pairs] + ([] if cfg.federation.etf_head else [wk for wk, _ in pairs])
    for k in train:
        params[k].requires_grad_(True)
    opt = torch.optim.SGD([params[k] for k in train], lr=cfg.federation.lr0, momentum=0.9)
    model.train()
    for m in model.modules():                 # keep the pooled statistics fixed
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    loader = build_node_dataloader(node, cfg, split="train")
    total, done = epochs * len(loader), 0
    for _ in range(epochs):
        for batch in loader:
            if cfg.federation.lr_schedule == "linear":   # same decay as training, within this round
                for g in opt.param_groups:
                    g["lr"] = cfg.federation.lr0 * (1 - (1 - cfg.federation.lrf) * done / total)
            done += 1
            batch["img"] = batch["img"].to(device).float() / 255.0
            opt.zero_grad()
            loss, _ = criterion.loss(criterion.parse_output(model.forward(batch["img"])), batch)
            loss.sum().backward()
            for k in train:                   # rows of other nodes' classes stay as they are
                params[k].grad[unowned] = 0
            torch.nn.utils.clip_grad_norm_([params[k] for k in train], max_norm=cfg.federation.grad_clip)
            opt.step()
    return {k: params[k].detach().cpu()[owned].clone() for k in train}, owned


def calibrate(cfg: FedYoloConfig, state_dict: dict, stages: str = "bn+head") -> dict:
    """The whole round, simulated: every node runs its side, the server combines.
    stages: "bn", "head" or "bn+head"."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32))
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.federation.device)
    sd = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
    if "bn" in stages:
        stats = []
        for node in cfg.nodes:
            s, n = _bn_contribution(node, cfg, sd, device)
            log.info(f"[calibration] {node.name}: BatchNorm statistics from {n} batches")
            stats.append((s, n))
        sd = pool_bn_stats(sd, stats, cfg.federation.calibration_bn_pool)
    if "head" in stages:
        new = {k: v.clone() for k, v in sd.items()}
        for node in cfg.nodes:
            rows, owned = node_head_align(node, cfg, sd, device, cfg.federation.calibration_epochs)
            for k, r in rows.items():
                new[k][owned] = r.to(new[k].dtype)
            log.info(f"[calibration] {node.name}: class rows retrained for {owned}")
        sd = new
    return sd


def calibrate_progressive(cfg: FedYoloConfig, state_dict: dict, order: list[str],
                          refresh: bool = False) -> list[dict]:
    """Asynchronous calibration: no barrier, nobody waits.

    The final model M is published uncalibrated; nodes contribute one at a time,
    whenever they are available (here: in `order`). A contribution is the node's
    BatchNorm statistics of M (merged by the server at once) followed by its own
    class rows re-fitted on M with the statistics merged so far. After every
    contribution the server releases a model:
      - BatchNorm: pooled over the contributors, with M's own stored statistics
        standing in for the nodes not heard from yet (weighted by their batch
        count, which the server knows from their training uploads), so a partial
        release never rests on one node's images alone;
      - class rows: re-fitted rows of the contributors, M's rows for the others.
    After the last contribution this is the complete calibration.
    refresh: afterwards every node, when it next connects (same order), re-fits its rows once
    against the complete BatchNorm pool -- early contributors had fitted theirs to a partial pool.
    Returns the released state_dict after 1, 2, ..., N contributions (and after the refresh)."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32))
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.federation.device)
    base = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
    nodes = {n.name: n for n in cfg.nodes}
    bn_names = [k[:-len(".running_mean")] for k in base if k.endswith(".running_mean")]
    stored = {n: (base[f"{n}.running_mean"].float(), base[f"{n}.running_var"].float()) for n in bn_names}
    prior = {nm: len(build_node_dataloader(nodes[nm], cfg, split="train")) for nm in order}
    contrib: dict[str, tuple[dict, int]] = {}
    rows: dict[str, torch.Tensor] = {}
    released = []
    for nm in order:
        s, n = _bn_contribution(nodes[nm], cfg, base, device)
        contrib[nm] = (s, n)
        entries = list(contrib.values()) + [(stored, prior[m]) for m in order if m not in contrib]
        pooled = pool_bn_stats(base, entries, cfg.federation.calibration_bn_pool)
        for k, r in rows.items():
            pooled[k] = r.clone()
        new_rows, owned = node_head_align(nodes[nm], cfg, pooled, device, cfg.federation.calibration_epochs)
        for k, r in new_rows.items():
            rows.setdefault(k, base[k].clone())[owned] = r.to(base[k].dtype)
            pooled[k] = rows[k].clone()
        log.info(f"[async calibration] {nm} contributed ({len(contrib)}/{len(order)}): "
                 f"BatchNorm from {n} batches, rows {owned}")
        released.append(pooled)
    if refresh:
        final = {k: v.clone() for k, v in released[-1].items()}
        for nm in order:
            new_rows, owned = node_head_align(nodes[nm], cfg, final, device, cfg.federation.calibration_epochs)
            for k, r in new_rows.items():
                final[k][owned] = r.to(final[k].dtype)
            log.info(f"[async calibration] {nm} refreshed its rows {owned} on the complete pool")
        released.append(final)
    return released
