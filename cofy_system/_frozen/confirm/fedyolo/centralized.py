"""Centralized baseline: a single model trained directly (no federated
aggregation at all) by cycling through every node's dataloader and stepping
the SAME set of weights every time, switching the class mask to match
whichever node's batch is currently being trained on.

This is deliberately NOT "ignore node boundaries and assume every image has
every class labeled" -- that data usually doesn't exist (a node's images
were never annotated for classes it doesn't own). Instead this baseline
gets exactly the same information the federation gets (same images, same
partial per-node labels, same total number of local epochs over each node's
data) but updates one set of weights directly instead of aggregating
independently-trained copies. That isolates the cost of federation itself
from the cost of partial labels, which both approaches face equally.

If you DO have a fully cross-annotated dataset (every image labeled for
every class), point a single extra "oracle" node at it and evaluate that
checkpoint instead -- that measures the federation+masking cost combined
with the partial-label cost, i.e. the absolute ceiling.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

from .client import _class_mask
from .config import FedYoloConfig
from .data import build_node_dataloader
from .model import build_model
from .optim_utils import schedule_factor, set_warmup_lr

log = logging.getLogger("fedyolo.centralized")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_centralized_baseline(cfg: FedYoloConfig, on_epoch_end=None) -> dict:
    """on_epoch_end(epoch, model) -> bool, optional: called after every epoch (all nodes);
    returning True stops training early (tools/budget_pilot.py)."""
    device = torch.device(cfg.federation.device)
    # See client.py's run_node_round for why random/np.random are seeded
    # here too, not just torch.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32))
    torch.manual_seed(cfg.seed)

    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained, class_names=cfg.global_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.federation.lr0, momentum=0.9, weight_decay=5e-4)

    node_loaders = [
        (node, node.owned_global_ids(cfg.global_classes), build_node_dataloader(node, cfg, split="train"))
        for node in cfg.nodes
    ]

    # Match total epochs to whichever mode is active
    if cfg.federation.mode == "async":
        total_epochs = cfg.federation.async_node_cycles * cfg.federation.local_epochs
    else:
        total_epochs = cfg.federation.rounds * cfg.federation.local_epochs
        
    end2end = getattr(model, "end2end", False)

    step = 0
    total_steps = total_epochs * sum(len(loader) for _, _, loader in node_loaders)
    for epoch in range(total_epochs):
        log.info(f"=== centralized epoch {epoch + 1}/{total_epochs} ===")
        for node, owned_ids, loader in node_loaders:
            model.class_weights = _class_mask(cfg.nc, owned_ids, device)
            # Force rebuild so the new mask is picked up. `= None` is not
            # enough for YOLOEModel: its own loss() checks `hasattr(self,
            # "criterion")`, which is already True once the attribute
            # exists at all (even set to None) -- it would skip rebuilding
            # and then crash calling `None(preds, batch)`. Deleting the
            # attribute makes `hasattr` False again, which both
            # BaseModel.loss()'s `getattr(..., None) is None` check and
            # YOLOEModel.loss()'s `hasattr` check agree means "rebuild".
            if hasattr(model, "criterion"):
                del model.criterion
            if end2end:
                # E2ELoss ramps one2many/one2one loss weighting from
                # o2m=0.8 at construction to o2m=0.1 by the end of
                # training, but only across the epochs its OWN instance
                # lives through -- and this loop forces a fresh instance
                # every single node switch (`del model.criterion` above),
                # every epoch, so a lazily-rebuilt E2ELoss would never
                # accumulate any decay progress at all: same permanent
                # 80/20 split bug as client.py's federated path (see the
                # comment there), just reset even more often here. Fix:
                # build the criterion ourselves instead of leaving it to
                # BaseModel.loss()'s lazy `init_criterion()`, scope its
                # decay to this run's real total_epochs (not the
                # node-visit count), and fast-forward `.updates` to where
                # this epoch actually sits in that schedule so decay
                # progress survives the rebuild instead of restarting from
                # o2m=0.8 every node switch.
                from ultralytics.utils.loss import E2ELoss
                crit = E2ELoss(model)
                crit.one2many.hyp.epochs = crit.one2one.hyp.epochs = total_epochs
                crit.updates = epoch
                crit.o2m = crit.decay(crit.updates)
                crit.o2o = max(crit.total - crit.o2m, 0)
                model.criterion = crit
            model.train()
            for batch in loader:
                set_warmup_lr(optimizer, step, cfg.federation.warmup_steps, cfg.federation.lr0,
                              schedule_factor(cfg, step / total_steps))
                step += 1
                batch["img"] = batch["img"].to(device).float() / 255.0
                optimizer.zero_grad()
                loss, _ = model.loss(batch)
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(
                	model.parameters(), max_norm=cfg.federation.grad_clip
                )
                optimizer.step()
        if on_epoch_end is not None and on_epoch_end(epoch + 1, model):
            break

    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
