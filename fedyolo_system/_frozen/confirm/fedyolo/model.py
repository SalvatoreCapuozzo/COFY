"""Model construction, and the parameter split that makes class-heterogeneous
federation work: every weight tensor is either SHARED (aggregated across all
nodes) or PER-CLASS (aggregated only across the nodes that own that class).

Why this split is exact and not a heuristic
--------------------------------------------
YOLOv8's Detect head ends each of its `nl` scale branches (cv3) with a final
1x1 conv whose output channel `c` is literally the logit for class `c`
(verified against the installed ultralytics build below). Every other weight
in the network (backbone, neck, box-regression branch cv2, DFL, the two
feature-mixing convs inside cv3 before the final projection) is class-agnostic
-- it produces shared features, not a class score -- so it can be FedAvg'd
across every node regardless of which classes that node owns.

    model.{H}.cv3.{i}.2.weight   shape (nc, c3, 1, 1)   <- per-class, dim 0 = class
    model.{H}.cv3.{i}.2.bias     shape (nc,)            <- per-class, dim 0 = class
    everything else                                      <- shared

YOLOE is the one exception, not a variant of the same rule: its head
(YOLOEDetect) has NO class-indexed weight anywhere. Its `cv3` final conv
outputs a fixed `embed`-width visual feature (shape independent of `nc`);
class identity comes entirely from externally-supplied text-prompt
embeddings compared against that feature by a small GLOBAL (not per-class)
contrastive scale/bias in `cv4`. There is nothing to split for YOLOE --
per_class_param_names() returns [] for it and every weight is SHARED, see
_attach_frozen_class_embeddings() below for how the (frozen, not even part
of the model's state_dict) class embeddings are wired in instead.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
import yaml
from ultralytics.nn.modules.head import RTDETRDecoder, YOLOEDetect
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel, YOLOEModel

from .merge import delta_stats, merge_shared

log = logging.getLogger("fedyolo.model")

# Per-arch cache of {shared_key: pretrained_tensor}, populated on first use.
# build_model() is called very often (once per node per round/cycle), so
# without this every call would re-load the .pt checkpoint from disk.
_PRETRAINED_SHARED_CACHE: dict[str, dict] = {}

# (arch, tuple(class_names)) -> frozen text-prompt-embedding tensor, for
# YOLOE archs. Computing these calls a CLIP/mobileclip text encoder, which
# is far too slow to redo on every build_model() call (once per node per
# round) -- cached the same way pretrained shared weights are above.
_TEXT_EMBED_CACHE: dict[tuple, torch.Tensor] = {}


_CUSTOM_ARCH_SCALE_RE = re.compile(r"^(.+)_([nsmlx])$")


def _resolve_arch_cfg(arch: str):
    """Turn `arch` into whatever `DetectionModel(cfg=...)` should receive.

    Stock ultralytics arch files (yolov8l.yaml, yolo26n.yaml, yoloe-26s.yaml,
    ...) are passed through as a bare string unchanged -- ultralytics' own
    `guess_model_scale()` filename magic (strip trailing `<digits><scale
    letter>`, look up the unified multi-scale yaml, e.g. yolov8l.yaml ->
    yolov8.yaml + scale='l') already handles those correctly.

    fedyolo's own custom arch files (configs/arch/*.yaml, e.g. the YOLO26
    "classic head" bridge in configs/arch/yolo26_classic.yaml) do NOT follow
    that naming convention on purpose -- naming a file e.g. `yolo26l.yaml`
    would collide with ultralytics' own filename-scale guessing and silently
    resolve to the *stock* yolo26.yaml instead of our override. Custom archs
    instead encode scale as an explicit `_<letter>` suffix on the stem (e.g.
    `yolo26_classic_l.yaml`) and are resolved here by hand: load the
    unsuffixed base yaml as a dict and set `scale` on it directly, which
    bypasses ultralytics' filename parsing entirely (DetectionModel accepts
    cfg as a pre-loaded dict, see nn/tasks.py `_initialize_yolo_model`).
    """
    path = Path(arch)
    m = _CUSTOM_ARCH_SCALE_RE.match(path.stem)
    if m and path.parent.name == "arch":
        base_path = path.with_name(f"{m.group(1)}.yaml")
        if base_path.exists():
            d = yaml.safe_load(base_path.read_text())
            d["scale"] = m.group(2)
            d["yaml_file"] = str(path)
            return d
    return arch


def _pretrained_ckpt_name(arch: str) -> str:
    """Map `arch` to the ultralytics checkpoint name used to initialize
    SHARED parameters (see build_model docstring). Stock archs map to
    themselves with `.pt`. fedyolo's custom classic-head YOLO26 bridge maps
    to the *stock* yolo26{scale}.pt checkpoint: its backbone+neck are an
    unmodified copy of yolo26's, so those tensors still match by shape;
    only the detect head differs (end2end/reg_max), and
    _load_pretrained_shared_state already skips any key whose shape doesn't
    match (the reg_max-dependent cv2 branch), same as it does for every
    other arch -- no special-casing needed beyond picking the right file.

    YOLOE maps to the stock yolo26{scale}.pt checkpoint too, for the same
    reason but a stronger version of it: diffing ultralytics'
    cfg/models/26/yolo26.yaml against yoloe-26.yaml directly shows the
    backbone AND neck are byte-for-byte identical -- only the final Detect
    vs YOLOEDetect layer differs. No yoloe-26{scale}.pt is reachable through
    the existing download path at all, so without this mapping YOLOE trains
    its shared backbone from scratch every time; with it, backbone+neck
    transfer just as well as any other arch's does (again, shape-matching
    in _load_pretrained_shared_state naturally skips the YOLOE-specific
    head params -- cv3/cv4/reprta/savpe don't exist in the stock checkpoint
    at all, so they simply don't match and stay random, which is correct:
    class identity comes from the frozen text embeddings, not this
    checkpoint, per _attach_frozen_class_embeddings() below)."""
    stem = Path(arch).stem
    m = re.match(r"^yolo26_classic_([nsmlx])$", stem)
    if m:
        return f"yolo26{m.group(1)}.pt"
    m = re.match(r"^yoloe-26([nsmlx])$", stem)
    if m:
        return f"yolo26{m.group(1)}.pt"
    return arch.replace(".yaml", ".pt")


def _is_yoloe_arch(arch: str) -> bool:
    return "yoloe" in Path(arch).stem.lower()


def _is_rtdetr_arch(arch: str) -> bool:
    return "rtdetr" in Path(arch).stem.lower()


def build_model(
    arch: str, nc: int, imgsz: int, pretrained: bool = True, class_names: list[str] | None = None
) -> DetectionModel:
    """Build a YOLOv8/YOLO26/YOLOE detection model.

    pretrained=True (default): the class-agnostic SHARED parameters --
    backbone, neck, box-regression branch, DFL, and the two feature-mixing
    convs inside cv3 -- are initialized from the architecture's official
    COCO-pretrained checkpoint (e.g. yolov8m.yaml -> yolov8m.pt). This is
    exactly the SHARED half of the per-class/shared split this module is
    built around (see module docstring), so it transfers with zero change
    to the federated aggregation or loss-masking logic downstream.

    The per-class head (cv3.{i}.2, and one2one_cv3.{i}.2 for end2end archs)
    always stays randomly initialized regardless of `pretrained`: its shape
    depends on `nc`, which will not match COCO's 80 classes for a custom
    class set, so there's no principled way to transfer it anyway.

    pretrained=False reproduces the original from-scratch (fully blank)
    behavior -- training from random initialization end to end.

    class_names: required for YOLOE archs (ignored otherwise) -- the global
    class-name list (cfg.global_classes) used to compute the frozen
    text-prompt embeddings every node predicts against. See
    _attach_frozen_class_embeddings().
    """
    is_yoloe = _is_yoloe_arch(arch)
    is_rtdetr = _is_rtdetr_arch(arch)
    model_cls = YOLOEModel if is_yoloe else (RTDETRDetectionModel if is_rtdetr else DetectionModel)
    model = model_cls(cfg=_resolve_arch_cfg(arch), nc=nc, verbose=False)
    model.args = _default_train_args(imgsz)
    if pretrained:
        shared_state = _load_pretrained_shared_state(arch, model)
        if shared_state:
            sd = model.state_dict()
            sd.update(shared_state)
            model.load_state_dict(sd)
    if is_yoloe:
        if not class_names:
            raise ValueError(f"build_model({arch!r}): YOLOE archs require class_names (pass cfg.global_classes)")
        _attach_frozen_class_embeddings(model, arch, class_names)
    return model


def _attach_frozen_class_embeddings(model: "YOLOEModel", arch: str, class_names: list[str]) -> None:
    """Wire YOLOE's class identity in -- NOT a per-class weight (see module
    docstring): the same frozen text-prompt embedding vector for every
    class, computed once by the arch's text encoder (`text_model` in the
    yaml, e.g. mobileclip2:b) and cached across calls (this runs once per
    node per round; re-encoding class names through a CLIP-style text model
    every time would be a large, pointless slowdown).

    `model.set_classes()` stores the embeddings as a plain tensor attribute
    (`self.pe`), NOT an nn.Parameter or a registered buffer -- so it never
    appears in `model.state_dict()` at all, and needs no explicit exclusion
    from FedAvg the way cv3.{i}.2 does for YOLOv8/YOLO26 (there's simply
    nothing to exclude; every real state_dict key is class-agnostic and
    already handled by aggregate_shared()). `model.forward(img)` -- exactly
    the call every other call site in this codebase already makes, no
    signature changes needed -- automatically falls back to this cached
    `self.pe` for its class prompts when called with no extra kwargs
    (ultralytics' own YOLOEModel.predict() -> get_cls_pe() fallback path).
    """
    key = (arch, tuple(class_names))
    if key not in _TEXT_EMBED_CACHE:
        try:
            embeddings = model.get_text_pe(class_names, cache_clip_model=True)
        except Exception as exc:
            embed_dim = model.model[-1].embed
            log.warning(
                f"[yoloe] could not compute text embeddings via this arch's text_model ({exc}) -- "
                f"falling back to random class-embedding init"
            )
            embeddings = torch.randn(1, len(class_names), embed_dim)
        _TEXT_EMBED_CACHE[key] = embeddings.detach().cpu()
        log.info(f"[yoloe] computed frozen text-prompt embeddings for {len(class_names)} classes via {arch}'s text_model")
    model.set_classes(class_names, _TEXT_EMBED_CACHE[key])


def _load_pretrained_shared_state(arch: str, model: DetectionModel) -> dict:
    """Return {key: tensor} for every SHARED (non-per-class) key in `model`
    that also exists with a matching shape in `arch`'s official COCO
    checkpoint. Falls back to an empty dict (pure random init, with a
    warning) if the checkpoint can't be fetched -- a transient network/
    download failure shouldn't take down every node's training."""
    if arch in _PRETRAINED_SHARED_CACHE:
        return _PRETRAINED_SHARED_CACHE[arch]

    ckpt_name = _pretrained_ckpt_name(arch)
    try:
        from ultralytics import YOLO
        pretrained_sd = YOLO(ckpt_name).model.state_dict()
    except Exception as exc:
        log.warning(f"[pretrained] could not load {ckpt_name} ({exc}) -- falling back to random init")
        _PRETRAINED_SHARED_CACHE[arch] = {}
        return {}

    per_class_keys = {k for pair in per_class_param_names(model) for k in pair}
    own_sd = model.state_dict()
    shared_pretrained = {}
    skipped = []
    for k, v in own_sd.items():
        if k in per_class_keys:
            continue
        if k in pretrained_sd and pretrained_sd[k].shape == v.shape:
            shared_pretrained[k] = pretrained_sd[k].clone()
        else:
            skipped.append(k)

    n_shared = len(own_sd) - len(per_class_keys)
    log.info(
        f"[pretrained] {ckpt_name}: loaded {len(shared_pretrained)}/{n_shared} shared-parameter "
        f"tensors (per-class head stays random -- class count differs from COCO's 80)"
        + (f"; {len(skipped)} shared keys had no shape match and stay random too" if skipped else "")
    )
    _PRETRAINED_SHARED_CACHE[arch] = shared_pretrained
    return shared_pretrained


def _default_train_args(imgsz: int):
    from types import SimpleNamespace

    # Minimal set of hyperparameters v8DetectionLoss / BboxLoss read from model.args.
    return SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, imgsz=imgsz)


def _detect_module_index(model: DetectionModel) -> int:
    return len(model.model) - 1


def per_class_param_names(model: DetectionModel) -> list[tuple[str, str]]:
    """Return [(weight_key, bias_key), ...] for each detection scale's final
    per-class projection layer, e.g. [('model.22.cv3.0.2.weight', 'model.22.cv3.0.2.bias'), ...]

    end2end models (YOLO26's native mode: `Detect.__init__` deep-copies
    `cv2`/`cv3` into `one2one_cv2`/`one2one_cv3` when `end2end=True`, see
    ultralytics.nn.modules.head.Detect) have a SECOND per-class projection
    layer at `one2one_cv3.{i}.2` -- same shape, same per-class semantics,
    just the NMS-free one-to-one branch instead of the one-to-many branch.
    Included here too so every caller (split/aggregate/async-delta, all of
    which just iterate this list) handles both branches automatically with
    no further changes.

    YOLOE (YOLOEDetect head) returns [] -- see the module docstring: its
    cv3 outputs a fixed-width visual embedding, not a per-class logit, so
    there is no per-class weight to split at all. Every caller treats an
    empty list correctly already: aggregate_per_class() becomes a no-op and
    aggregate_shared()/split_state_dict() FedAvg the entire state_dict.
    """
    h = _detect_module_index(model)
    head = model.model[h]
    if isinstance(head, YOLOEDetect):
        return []
    if isinstance(head, RTDETRDecoder):
        # RTDETRDecoder has its own, differently-shaped per-class tensors --
        # a Linear layer's weight per decoder query-scoring head, not a
        # conv's output channels, but the SAME "dim 0 indexes class id"
        # convention (verified directly: enc_score_head.weight comes out
        # shape (nc, 256)). One shared enc_score_head, one dec_score_head
        # per decoder layer -- NOTE: head.nl means "num feature levels" for
        # this head (3), NOT "num decoder layers" the way it means "num
        # detection scales" for Detect -- using it here would have
        # silently dropped half of dec_score_head's real 6 entries (caught
        # by direct inspection, not assumed: len(head.dec_score_head) == 6
        # while head.nl == 3). Read the actual module list's length instead.
        #
        # denoising_class_embed.weight (nc, 256) is deliberately excluded:
        # it has no bias companion (breaks the weight/bias pair contract
        # every caller here assumes) and it only feeds the denoising
        # auxiliary training signal, not the actual inference-time class
        # score -- left in the shared/FedAvg set rather than extending
        # every caller to handle an unpaired weight-only entry for this
        # one tensor.
        names = [(f"model.{h}.enc_score_head.weight", f"model.{h}.enc_score_head.bias")]
        names += [
            (f"model.{h}.dec_score_head.{i}.weight", f"model.{h}.dec_score_head.{i}.bias")
            for i in range(len(head.dec_score_head))
        ]
        return names
    nl = head.nl
    names = [(f"model.{h}.cv3.{i}.2.weight", f"model.{h}.cv3.{i}.2.bias") for i in range(nl)]
    if getattr(head, "end2end", False):
        names += [(f"model.{h}.one2one_cv3.{i}.2.weight", f"model.{h}.one2one_cv3.{i}.2.bias") for i in range(nl)]
    return names


def split_state_dict(state_dict: dict, model: DetectionModel) -> tuple[dict, dict]:
    """Split a state_dict into (shared_params, per_class_params).

    per_class_params keeps the SAME key names as state_dict but every tensor's
    dim-0 still indexes classes -- callers slice by class id directly.
    """
    per_class_keys = {k for pair in per_class_param_names(model) for k in pair}
    shared = {k: v for k, v in state_dict.items() if k not in per_class_keys}
    per_class = {k: v for k, v in state_dict.items() if k in per_class_keys}
    return shared, per_class


def aggregate_shared(node_state_dicts: list[dict], node_weights: list[float],
                     model: DetectionModel, prev_global: dict | None = None,
                     rule: str = "mean", trim: float = 0.2,
                     node_steps: list[float] | None = None,
                     node_fishers: list[dict] | None = None,
                     fisher_damping: float = 0.01) -> dict:
    """Sample-weighted FedAvg over every node, restricted to shared keys.

    rule="mean" (the default) is the original behavior, unchanged. rule="ties"
    resolves sign conflicts between nodes instead of averaging through them --
    see fedyolo/merge.py. `prev_global` is only needed by the latter.
    """
    per_class_keys = {k for pair in per_class_param_names(model) for k in pair}
    keys = [k for k in node_state_dicts[0].keys() if k not in per_class_keys]
    return merge_shared(node_state_dicts, node_weights, keys, prev_global,
                        rule=rule, trim=trim, node_steps=node_steps,
                        node_fishers=node_fishers, fisher_damping=fisher_damping)


def aggregate_per_class(
    node_state_dicts: list[dict],
    node_owned_global_ids: list[list[int]],
    node_class_counts: list[dict[int, int]],
    model: DetectionModel,
    nc: int,
) -> dict:
    """Class-conditional aggregation: for each global class id, average the
    relevant output-channel slice only across the nodes that own that class,
    weighted by how many labeled instances of that class each node has.

    Classes with zero contributing nodes keep their (randomly initialized)
    value from node 0 unchanged -- this should not happen if config
    validation passed, but is handled defensively.
    """
    out = {}
    for weight_key, bias_key in per_class_param_names(model):
        ref_w = node_state_dicts[0][weight_key]
        ref_b = node_state_dicts[0][bias_key]
        new_w = torch.zeros_like(ref_w, dtype=torch.float32)
        new_b = torch.zeros_like(ref_b, dtype=torch.float32)

        for c in range(nc):
            contributors = [
                i for i, owned in enumerate(node_owned_global_ids) if c in owned
            ]
            if not contributors:
                new_w[c] = ref_w[c].to(torch.float32)
                new_b[c] = ref_b[c].to(torch.float32)
                continue
            total = sum(max(node_class_counts[i].get(c, 0), 1) for i in contributors)
            for i in contributors:
                w = max(node_class_counts[i].get(c, 0), 1) / total
                new_w[c] += node_state_dicts[i][weight_key][c].to(torch.float32) * w
                new_b[c] += node_state_dicts[i][bias_key][c].to(torch.float32) * w

        out[weight_key] = new_w.to(ref_w.dtype)
        out[bias_key] = new_b.to(ref_b.dtype)
    return out


def aggregate(
    node_state_dicts: list[dict],
    node_num_images: list[int],
    node_owned_global_ids: list[list[int]],
    node_class_counts: list[dict[int, int]],
    model: DetectionModel,
    nc: int,
    prev_global: dict | None = None,
    rule: str = "mean",
    trim: float = 0.2,
    diag_out: list | None = None,
    node_steps: list[float] | None = None,
    weight_basis: str = "images",
    node_instance_totals: list[float] | None = None,
    node_fishers: list[dict] | None = None,
    fisher_damping: float = 0.01,
) -> dict:
    """Full server-side aggregation for one federated round.

    prev_global/rule/trim select the shared-parameter merge rule (see
    fedyolo/merge.py); the defaults reproduce the original FedAvg exactly.
    Passing `diag_out` (a list) appends this round's per-layer-group
    node-disagreement statistics to it -- diagnostic only, never affects the
    merge.
    """
    # M4: what a node's share of the shared trunk should be proportional to.
    # "images" is the original behavior. Under this project's disjoint-class
    # split a node's images are only LABELED for its own classes, so image
    # count measures how much data a node holds rather than how much
    # supervision it actually contributed -- "instances" uses the labeled
    # instance count instead, matching what aggregate_per_class already does.
    if weight_basis == "instances" and node_instance_totals:
        weights = [float(max(n, 1)) for n in node_instance_totals]
    else:
        weights = [float(n) for n in node_num_images]
    per_class_keys = {k for pair in per_class_param_names(model) for k in pair}
    keys = [k for k in node_state_dicts[0].keys() if k not in per_class_keys]

    if diag_out is not None and prev_global is not None:
        diag_out.append(delta_stats(
            node_state_dicts, prev_global, keys, weights,
            head_prefix=f"model.{_detect_module_index(model)}.",
            per_class_keys=per_class_keys,
        ))

    shared = aggregate_shared(node_state_dicts, weights, model,
                              prev_global=prev_global, rule=rule, trim=trim,
                              node_steps=node_steps, node_fishers=node_fishers,
                              fisher_damping=fisher_damping)
    per_class = aggregate_per_class(node_state_dicts, node_owned_global_ids, node_class_counts, model, nc)
    merged = {**shared, **per_class}
    return merged


class MaskedRTDETRDetectionLoss:
    """Class-masked RTDETRDetectionLoss -- RT-DETR's real-analogue of
    _class_mask()/model.class_weights, which v8DetectionLoss/E2ELoss read
    directly but RTDETRDetectionLoss has no equivalent hook for at all:
    confirmed directly (its forward signature is (preds, batch, dn_bboxes=,
    dn_scores=, dn_meta=) -- no model reference, nothing resembling
    model.class_weights anywhere in it or its DETRLoss base class).

    Masks the classification loss specifically (not localization/giou, not
    the Hungarian matching cost -- matching still sees real predicted
    scores, exactly mirroring what real per-node data would produce, since
    a node's own ground truth never contains unowned-class boxes anyway)
    by cutting gradient flow to pred_scores' unowned-class columns right
    before DETRLoss._get_loss_class() computes the classification loss --
    the one insertion point that method's own signature exposes
    (pred_scores shape (B, N, nc), verified directly). This achieves the
    same outcome _class_mask() achieves for YOLO: a node never updates the
    per-class score-head weight ROWS for classes it doesn't own, because
    those rows never receive a nonzero gradient from this node's loss.

    Built as a thin wrapper (composition), not a subclass -- RTDETRDetectionLoss's
    __init__ signature and DETRLoss's internal method-calling structure
    aren't guaranteed stable enough across ultralytics versions to safely
    override _get_loss_class via inheritance; wrapping and monkey-patching
    just that one bound method on our own instance is more contained and
    easier to reason about if ultralytics changes DETRLoss's internals."""

    def __init__(self, nc: int, owned_ids: list[int], **kwargs):
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        self._inner = RTDETRDetectionLoss(nc=nc, **kwargs)
        self._owned_ids = owned_ids
        self._owned_mask = None
        self._real_get_loss_class = self._inner._get_loss_class
        self._inner._get_loss_class = self._masked_get_loss_class

    def _masked_get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        if self._owned_mask is None or self._owned_mask.device != pred_scores.device:
            m = torch.zeros(pred_scores.shape[-1], dtype=torch.bool, device=pred_scores.device)
            m[self._owned_ids] = True
            self._owned_mask = m
        masked = pred_scores.clone()
        masked[..., ~self._owned_mask] = masked[..., ~self._owned_mask].detach()
        return self._real_get_loss_class(masked, targets, gt_scores, num_gts, postfix)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)
