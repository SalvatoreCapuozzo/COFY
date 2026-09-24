"""One node's local training for one federated round.

`run_node_round` is the function each worker process executes. It:
  1. builds a model with the GLOBAL class count and loads the current global weights
  2. sets a binary class mask so the classification loss is zero for classes
     this node doesn't own (the false-negative-suppression fix)
  3. optionally adds a self-distillation term against the previous round's
     global model for not-owned classes (pseudo-labeling)
  4. trains locally for `local_epochs`
  5. returns the updated state_dict (on CPU, ready to pickle back to the server)
     plus the metadata the server needs for weighted aggregation
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import FedYoloConfig, NodeConfig
from .data import build_node_dataloader, build_node_dataset, per_class_instance_counts
from .model import MaskedRTDETRDetectionLoss, build_model
from .optim_utils import set_warmup_lr
from .pseudo_label import distillation_loss

# Guards concurrent appends to live_training_logs.txt in async mode
_log_file_lock = threading.Lock()


@dataclass
class NodeRoundResult:
    name: str
    state_dict: dict
    num_images: int
    class_counts: dict
    owned_global_ids: list
    optimizer_state: dict
    # Local optimizer steps actually taken this round. Nodes hold different
    # amounts of data under a uniform local_epochs, so this differs per node --
    # which is exactly the bias FedNova-style merge normalization removes
    # (see fedyolo/merge.py step_scales()).
    num_steps: int = 0


def _optimizer_state_to_cpu(state_dict: dict) -> dict:
    """Move an SGD optimizer state_dict's per-parameter buffers (momentum_buffer)
    to CPU so it can be pickled back to the server process (sync mode) or held
    across cycles (async mode) without pinning GPU memory between calls."""
    state_dict["state"] = {
        k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        for k, v in state_dict["state"].items()
    }
    return state_dict


def _class_mask(nc: int, owned_ids: list[int], device) -> torch.Tensor:
    mask = torch.zeros(nc, device=device)
    mask[owned_ids] = 1.0
    return mask
    
def _read_latest_map(output_dir: str) -> str:
    """Read the latest mAP from live_map.json written by the server after each
    evaluation. Returns an empty string when no evaluation has run yet (first
    round/cycle) so the progress bar degrades gracefully."""
    try:
        data = json.loads(Path(output_dir).joinpath("live_map.json").read_text())
        return f" [dim]| mAP50: {data['map50']:.4f}[/dim]"
    except Exception:
        return ""


def run_node_round(
    node: NodeConfig,
    cfg: FedYoloConfig,
    global_state_dict: dict,
    round_idx: int,
    teacher_state_dict: dict | None,
    optimizer_state: dict | None = None,
) -> NodeRoundResult:
    # Seed every RNG the training path actually draws from, not just torch's.
    # ultralytics' own augmentation pipeline (Mosaic, RandomPerspective,
    # RandomHSV, RandomFlip, ...) reads from Python's `random` and numpy's
    # RNG, not torch's -- leaving those unseeded meant every batch's mosaic
    # composition, crop offsets, and color jitter were drawn from whatever
    # state the process happened to auto-seed from OS entropy at startup,
    # different every invocation regardless of `cfg.seed`. That was the
    # actual dominant cause of the same-config-different-result runs
    # documented in the reports (previously mischaracterized there as
    # "MPS non-determinism" -- corrected: MPS-level nondeterminism may still
    # contribute at the margin, but this was the bigger, fixable gap).
    random.seed(cfg.seed + round_idx)
    np.random.seed((cfg.seed + round_idx) % (2**32))
    torch.manual_seed(cfg.seed + round_idx)
    device = torch.device(cfg.federation.device)

    owned_ids = node.owned_global_ids(cfg.global_classes)
    unowned_ids = [c for c in range(cfg.nc) if c not in owned_ids]

    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained, class_names=cfg.global_classes).to(device)
    model.load_state_dict(global_state_dict)
    model.class_weights = _class_mask(cfg.nc, owned_ids, device)  # zeroes loss for unowned classes
    model.train()

    # end2end archs (YOLO26 native mode) forward() a nested
    # {"one2many": {...}, "one2one": {...}} dict instead of one flat
    # {boxes, scores, feats} dict -- same E2ELoss class ultralytics' own
    # DetectionModel.init_criterion() picks for end2end models, so the loss
    # value matches what model.loss(batch) would give (used by
    # centralized.py/simulate_to_target.py, which go through that generic
    # path and needed no changes here). Both of E2ELoss's internal
    # v8DetectionLoss instances read model.class_weights off this same
    # `model` at construction time, so the existing masking mechanism
    # applies to both the one2many and one2one branches automatically.
    from ultralytics.nn.modules.head import RTDETRDecoder
    is_rtdetr = isinstance(model.model[-1], RTDETRDecoder)
    end2end = getattr(model.model[-1], "end2end", False)
    if is_rtdetr:
        # RTDETRDetectionModel.loss(batch) (called below in place of the
        # usual forward()-then-criterion() two-step -- RT-DETR's forward
        # needs the batch during prediction for its denoising queries, so
        # ultralytics bundles forward+loss into one method for it) picks
        # up model.criterion if already set, rather than building its own
        # (unmasked) RTDETRDetectionLoss via init_criterion() -- see
        # BaseModel/RTDETRDetectionModel.loss()'s own
        # `if not hasattr(self, "criterion")` check.
        criterion = MaskedRTDETRDetectionLoss(nc=cfg.nc, owned_ids=owned_ids, use_vfl=True)
        model.criterion = criterion
    else:
        criterion = E2ELoss(model) if end2end else v8DetectionLoss(model)
    if end2end:
        # BUG FOUND (this session): E2ELoss ramps its one2many/one2one loss
        # weighting from o2m=0.8/o2o=0.2 at construction to o2m=0.1/o2o=0.9
        # by the end of training -- but only if something calls
        # criterion.update() once per epoch, which nothing in fedyolo ever
        # did. Every end2end (Phase 2) run in this whole investigation
        # trained with the one2one branch permanently stuck at its initial
        # 20% loss weight -- and one2one is the ONLY branch actually used
        # at inference/eval (Detect.forward() picks preds["one2one"] when
        # self.training is False), so the branch that determines every
        # reported mAP50 for Phase 2 was chronically undertrained the
        # entire time. fedyolo's model.args is a minimal SimpleNamespace
        # (see model.py's _default_train_args) with no `epochs` field at
        # all, which decay() needs -- set it here to this round's
        # local_epochs (not the federation's total epoch budget): the
        # model/criterion are rebuilt fresh every round from the
        # aggregated global weights, so local_epochs is the actual span
        # this criterion instance will live for, and scoping the decay to
        # it means one2one gets meaningfully trained within every single
        # round's contribution instead of needing a persistent counter
        # that would never survive the per-round rebuild anyway.
        criterion.one2many.hyp.epochs = cfg.federation.local_epochs
        criterion.one2one.hyp.epochs = cfg.federation.local_epochs

    use_distill = (
        cfg.federation.pseudo_label.enabled
        and round_idx >= cfg.federation.pseudo_label.start_round
        and teacher_state_dict is not None
        and len(unowned_ids) > 0
    )
    teacher = None
    if use_distill:
        teacher = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained, class_names=cfg.global_classes).to(device)
        teacher.load_state_dict(teacher_state_dict)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    dataset = build_node_dataset(node, cfg, split="train")
    dataloader = build_node_dataloader(node, cfg, split="train")
    optimizer = torch.optim.SGD(
    	model.parameters(), lr=cfg.federation.lr0, momentum=0.9, weight_decay=5e-4
    )
    if optimizer_state is not None:
        # Carry this node's momentum buffers forward from its previous cycle/round
        # instead of restarting cold every time. The model WEIGHTS still get
        # reloaded from the (possibly aggregated-elsewhere) global state above --
        # only the gradient-momentum statistics persist. LR is unaffected: the
        # per-step warmup below overwrites param_groups['lr'] before every
        # optimizer.step() regardless of what's in the loaded state.
        optimizer.load_state_dict(optimizer_state)

    step = 0
    log_path = Path(cfg.output_dir) / "live_training_logs.txt"
    # width=200: force_terminal alone doesn't stop Rich truncating columns to
    # "..." -- with no real tty to measure, it falls back to a narrow default
    # (~80 cols) that a description+bar+counts+timers row doesn't fit in.
    custom_console = Console(force_terminal=True, width=200)
    
    # Total steps for the whole round = epochs × batches/epoch
    # This lets us use a SINGLE progress task instead of one per epoch,
    # which is what was causing the flickering.
    total_steps = cfg.federation.local_epochs * len(dataloader)

    # Read the latest mAP once before training starts (server writes this file
    # after each evaluation; we refresh it once per epoch below).
    map_str = _read_latest_map(cfg.output_dir)
    
    # Create a custom Rich progress bar with Elapsed and Remaining time separated cleanly
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]⏱️ Passed:[/green]"),
        TimeElapsedColumn(),
        TextColumn("[yellow]⏳ Left:[/yellow]"),
        TimeRemainingColumn(),
        transient=False,
        console=custom_console,
    ) as progress:
        
        task = progress.add_task(
            f"[cyan]{node.name} Ep 1/{cfg.federation.local_epochs}[/cyan]",
            total=total_steps,
        )
        
        for epoch in range(cfg.federation.local_epochs):
            # Variables to track the average loss for this specific epoch
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                set_warmup_lr(optimizer, step, cfg.federation.warmup_steps, cfg.federation.lr0)
                step += 1

                batch["img"] = batch["img"].to(device).float() / 255.0
                optimizer.zero_grad()

                if is_rtdetr:
                    # Bundled forward+loss, not the forward()-then-criterion()
                    # two-step below -- RT-DETR's forward needs the batch
                    # during prediction for its denoising queries, so
                    # ultralytics bundles both into one method for it (see
                    # the is_rtdetr branch above where model.criterion is
                    # set to the masked wrapper this picks up).
                    loss, _ = model.loss(batch)
                elif end2end:
                    # preds is already {"one2many": {...}, "one2one": {...}}
                    # -- E2ELoss.__call__ does its own internal parse/split,
                    # no parse_output() step needed (that method is only a
                    # no-op passthrough for a non-tuple input anyway).
                    preds = model.forward(batch["img"])
                    loss, _ = criterion(preds, batch)
                else:
                    preds = model.forward(batch["img"])
                    parsed = criterion.parse_output(preds)
                    loss, _ = criterion.loss(parsed, batch)
                loss = loss.sum()

                if use_distill and is_rtdetr:
                    raise NotImplementedError(
                        "pseudo-labeling isn't wired through RT-DETR's loss path -- "
                        "leave federation.pseudo_label.enabled: false for RT-DETR configs"
                    )
                if use_distill:
                    with torch.no_grad():
                        t_preds = teacher.forward(batch["img"])
                    # teacher.eval() means teacher.forward() returns the
                    # eval-mode (y, preds) tuple, not the raw dict -- unwrap
                    # it the same way parse_output() does for the non-e2e
                    # path below (student stays the raw training-mode dict,
                    # no unwrap needed there).
                    if isinstance(t_preds, tuple):
                        t_preds = t_preds[1]
                    if end2end:
                        # one2one: the NMS-free branch actually used at
                        # inference, so that's what the teacher's "soft
                        # label" should reflect for the not-owned classes.
                        student_scores = preds["one2one"]["scores"].permute(0, 2, 1).contiguous()
                        teacher_scores = t_preds["one2one"]["scores"].permute(0, 2, 1).contiguous()
                    else:
                        student_scores = parsed["scores"].permute(0, 2, 1).contiguous()
                        teacher_scores = t_preds["scores"].permute(0, 2, 1).contiguous()
                    d_loss = distillation_loss(
                        student_scores, teacher_scores, unowned_ids, cfg.federation.pseudo_label.conf_thresh
                    )
                    loss = loss + cfg.federation.pseudo_label.weight * d_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.federation.grad_clip)
                optimizer.step()

                # Accumulate the loss for our text file logging
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update the SINGLE task — description changes every batch,
                # the bar advances by 1 step; no new task is created.
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"[cyan]{node.name} "
                        f"Ep {epoch + 1}/{cfg.federation.local_epochs}[/cyan] "
                        f"[white]Loss: {loss.item():.4f}[/white]"
                        f"{map_str}"
                    ),
                )

            # --- END OF EPOCH LOGGING ---
            avg_epoch_loss = epoch_loss / max(1, num_batches)

            if end2end:
                # Advance the one2many->one2one decay schedule now that a
                # full epoch has actually happened -- see the comment where
                # criterion.one2many.hyp.epochs was set above.
                criterion.update()

            # Refresh mAP string once per epoch so the next epoch's batches
            # show the latest evaluation result without hammering the filesystem.
            map_str = _read_latest_map(cfg.output_dir)
            
            # Append to the shared log file; lock guards concurrent writes in async mode.
            with _log_file_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"Round/Cycle: {round_idx + 1:02d} | Node: {node.name:10s} | "
                        f"Epoch: {epoch + 1:02d}/{cfg.federation.local_epochs:02d} | "
                        f"Avg Loss: {avg_epoch_loss:.4f}\n"
                    )

    custom_console.print(f"[bold green]✔ {node.name}[/bold green] finished local training!")

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    class_counts = per_class_instance_counts(dataset, cfg.nc)
    returned_optimizer_state = _optimizer_state_to_cpu(optimizer.state_dict())

    return NodeRoundResult(
        name=node.name,
        state_dict=state_dict,
        num_images=len(dataset),
        class_counts=class_counts,
        owned_global_ids=owned_ids,
        optimizer_state=returned_optimizer_state,
        num_steps=step,
    )
