"""Configuration schema and loader for a federated object-detection run.

Everything about a federation (global class list, model architecture, per-node
data locations and owned classes, federated hyperparameters) lives in one YAML
file. See any configs/kfm12_*.yaml for a worked example, or generate one
from your own dataset with tools/split_dataset_into_nodes.py --config-out.

federation.mode controls which server is used:
  "sync"  — classic FedAvg barrier: all nodes finish round N before
            the server aggregates and starts round N+1 (FedServer).
  "async" — each node independently cycles pull→train→push; the server
            aggregates immediately on every push with a staleness discount
            (AsyncFedServer). No barrier, no rounds — the global model
            version increments by 1 after each single-node submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    arch: str = "yolov8n.yaml"  # ultralytics architecture yaml
    imgsz: int = 640
    pretrained: bool = True     # init SHARED params (backbone/neck/box-head/DFL)
                                # from arch's official COCO checkpoint (e.g.
                                # yolov8m.yaml -> yolov8m.pt). The per-class
                                # head always stays random (nc != COCO's 80).
                                # Set False to reproduce the original from-
                                # scratch (fully blank) behavior.


@dataclass
class PseudoLabelConfig:
    enabled: bool = False
    # sync: interpreted as round index; async: interpreted as per-node cycle index
    start_round: int = 5
    conf_thresh: float = 0.5
    weight: float = 0.5


@dataclass
class FederationConfig:
    # ── selector ──────────────────────────────────────────────────────────────
    mode: str = "sync"              # "sync" | "async"

    # ── shared by both modes ──────────────────────────────────────────────────
    local_epochs: int = 5           # epochs per round (sync) or per cycle (async)
    batch_size: int = 8
    lr0: float = 0.001
    warmup_steps: int = 20          # per-round/cycle LR warmup steps
    grad_clip: float = 10.0
    workers: int = 0
    device: str = "auto"            # "auto" resolves at load time to the best
                                    # available backend on THIS machine: CUDA
                                    # if present, else Apple MPS, else CPU. Set
                                    # an explicit "cuda"/"cuda:0"/"mps"/"cpu"
                                    # to pin it instead (e.g. to force CPU on a
                                    # CUDA box, or to target a specific GPU).
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    data_fraction: float = 1.0      # use only the first N% of each node's images (both
                                    # splits); for fast smoke-tests on a large real dataset
                                    # without touching the config's node/class layout.
                                    # 1.0 = full dataset (default, no behaviour change).
                                    # This is Ultralytics' own naive prefix slice -- NOT
                                    # class-aware. Prefer balanced_fraction below on any
                                    # dataset with imbalanced classes (a plain fraction can
                                    # silently drop a rare class to zero images).
    balanced_fraction: float | None = None
                                    # Class-stratified alternative to data_fraction, for fast
                                    # local iteration that still checks every class: every
                                    # class keeps max(1, round(count * balanced_fraction)) of
                                    # its images, in BOTH train and the pooled val set used
                                    # for evaluation (data_fraction only ever touched train).
                                    # None (default) = disabled, data_fraction applies as
                                    # normal. Takes priority over data_fraction when set.
                                    # e.g. 0.10 for a balanced ~10% subset.

    # ── sync only (ignored in async mode) ─────────────────────────────────────
    rounds: int = 20                # federated rounds; total epochs/node = rounds × local_epochs

    # ── async only (ignored in sync mode) ─────────────────────────────────────
    async_node_cycles: int = 10     # pull→train→push cycles per node
                                    # total submissions = n_nodes × async_node_cycles
                                    # total epochs/node = async_node_cycles × local_epochs
    staleness_alpha: float = 0.5    # staleness discount: w = 1 / (1 + alpha × staleness)
                                    # staleness = global_version_now − version_when_node_pulled
                                    # higher alpha = stronger penalty for stale updates
    max_concurrent_nodes: int = 1   # how many nodes may train simultaneously — both modes.
                                    # sync:  caps the round's ProcessPoolExecutor workers
                                    #        (nodes within a round are independent; the
                                    #        barrier still waits for all of them either way).
                                    # async: 1 = sequential execution with async semantics
                                    #        (safe default); >1 = true parallelism.
                                    # >1 requires enough RAM/VRAM for N models simultaneously
                                    # (each yolov8l at imgsz=960 ≈ 1–2 GB).

    # ── shared-parameter merge rule (both modes) ─────────────────────────────
    merge_rule: str = "mean"        # "mean" = sample-weighted FedAvg (original,
                                    #   unchanged default).
                                    # "ties" = TIES-Merging conflict resolution:
                                    #   trim each node's smallest-magnitude delta
                                    #   entries, elect one sign per coordinate by
                                    #   summed weighted magnitude, average only the
                                    #   nodes agreeing with it. Aimed at the extreme
                                    #   non-IID case where nodes optimize disjoint
                                    #   objectives and plain averaging lets opposing
                                    #   updates cancel. See fedyolo/merge.py.
    merge_trim: float = 0.2         # "ties" only: fraction of each node's smallest
                                    #   delta entries zeroed before the sign election.
    weight_basis: str = "images"    # M4: "images" (original) or "instances" -- what a
                                    #   node's share of the shared-trunk merge is
                                    #   proportional to. Under the disjoint-class split a
                                    #   node's images are only labeled for its own classes,
                                    #   so instance count tracks contributed supervision
                                    #   more directly than image count does.
    normalize_steps: bool = False   # M5: FedNova -- rescale each node's delta to a common
                                    #   effective step count before weighting, removing the
                                    #   bias from nodes taking different numbers of local
                                    #   steps under a uniform local_epochs.
    keep_checkpoints: int = 3       # how many recent round checkpoints to retain.
                                    #   >1 lets a finished run be re-scored at several
                                    #   late rounds on a different eval set, which this
                                    #   config needs: its round-to-round mAP on the small
                                    #   balanced-fraction eval set swings by up to 0.10,
                                    #   so one final-round number is a noisy basis for
                                    #   comparing two runs.
    merge_diagnostics: bool = False # log per-layer-group cross-node delta agreement
                                    #   each round to merge_diagnostics.json.
                                    #   Diagnostic only -- never affects the merge.

    # ── live evaluation (both modes) ──────────────────────────────────────────
    eval_interval: int = 1          # sync:  evaluate every N rounds
                                    # async: evaluate every N submissions
                                    #        (default 1 = every submission; set to n_nodes for
                                    #         one eval per "virtual round" worth of submissions)


@dataclass
class NodeConfig:
    name: str
    data_yaml: str
    owned_classes: list[str]   # subset of global_classes, IN THE ORDER local label ids 0..k-1 map to them

    def class_id_map(self, global_classes: list[str]) -> dict[int, int]:
        """local label id -> global label id."""
        return {local_id: global_classes.index(name) 
        	for local_id, name in enumerate(self.owned_classes)}

    def owned_global_ids(self, global_classes: list[str]) -> list[int]:
        return [global_classes.index(name) for name in self.owned_classes]


@dataclass
class FedYoloConfig:
    global_classes: list[str]
    model: ModelConfig
    federation: FederationConfig
    nodes: list[NodeConfig]
    output_dir: str = "runs/federation"
    seed: int = 0
    # A fully-labeled, deduplicated evaluation set (every class on every
    # image, global class ids) that federated, centralized and oracle are all
    # scored on. Written by tools/split_dataset_into_nodes.py --split-mode
    # global. When unset, evaluation falls back to pooling each node's own
    # val split -- which on multi-object data holds duplicate copies of the
    # same image with different partial labels, and can include images
    # another node trained on (see that tool's docstring for the numbers).
    eval_data_yaml: str | None = None

    @property
    def nc(self) -> int:
        return len(self.global_classes)


def _resolve_device(device: str) -> str:
    """Turn "auto" into a concrete torch device string for this machine.
    Any other value (including an already-concrete one) passes through
    unchanged, so pinning a device explicitly always still works."""
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str | Path) -> FedYoloConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return load_config_from_dict(raw)


def load_config_from_dict(raw: dict, partial: bool = False) -> FedYoloConfig:
    """Same parsing load_config() does, split out so fedyolo/net/ can build
    an identical FedYoloConfig from a dict a server sent over the wire
    (fedyolo/net/rooms.py serves each room's own already-loaded raw config
    dict verbatim) instead of only ever from a local YAML file -- the
    client then overrides its own node's `data_yaml` in that dict before
    calling this, so every other hyperparameter (arch, imgsz, local_epochs,
    lr0, ...) stays identical to what the server is authoritative for.

    partial=True (fedyolo/net/client_app.py passes this for dynamic
    rooms): skip the "every global class is owned by someone" check --
    dynamic rooms' global_classes is pre-sized to max_classes with
    placeholder names for slots nobody has claimed yet, a real,
    permanent state for those rooms, not a config error. Found via
    real end-to-end testing: without this, every dynamic-room client
    crashed on startup with "No node owns these global classes" for its
    own room's still-unclaimed placeholder slots."""
    model = ModelConfig(**raw.get("model", {}))

    fed_raw = dict(raw.get("federation", {}))
    pl_raw = fed_raw.pop("pseudo_label", {})
    federation = FederationConfig(pseudo_label=PseudoLabelConfig(**pl_raw), **fed_raw)
    federation.device = _resolve_device(federation.device)

    if federation.mode not in ("sync", "async"):
        raise ValueError(f"federation.mode must be 'sync' or 'async', got '{federation.mode}'")

    nodes = [NodeConfig(**n) for n in raw["nodes"]]

    cfg = FedYoloConfig(
        global_classes=list(raw["global_classes"]),
        model=model,
        federation=federation,
        nodes=nodes,
        output_dir=raw.get("output_dir", "runs/federation"),
        seed=raw.get("seed", 0),
        eval_data_yaml=raw.get("eval_data_yaml"),
    )
    _validate(cfg, partial=partial)
    return cfg


def _validate(cfg: FedYoloConfig, partial: bool = False) -> None:
    """partial=True skips the "every global class must be owned by someone"
    check -- for fedyolo/net/'s dynamic rooms, global_classes is pre-sized
    to federation.max_classes with placeholder names for slots nobody has
    claimed yet (see rooms.py), which is an expected, permanent state for
    those rooms, not a config error. Static (file-loaded) rooms always use
    the full, strict check (partial=False, the default) -- unchanged."""
    for node in cfg.nodes:
        for cls in node.owned_classes:
            if cls not in cfg.global_classes:
                raise ValueError(
                    f"Node '{node.name}' owns class '{cls}' which is not in "
                    f"global_classes={cfg.global_classes}"
                )
        if len(set(node.owned_classes)) != len(node.owned_classes):
            raise ValueError(
                f"Node '{node.name}' has duplicate owned_classes: {node.owned_classes}"
            )

    if partial:
        return

    covered = {cls for node in cfg.nodes for cls in node.owned_classes}
    missing = set(cfg.global_classes) - covered
    if missing:
        raise ValueError(
            f"No node owns these global classes, they can never be learned: {sorted(missing)}"
        )
