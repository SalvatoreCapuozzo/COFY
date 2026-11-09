# Implementation spec for Claude Code: asynchronous, class-disjoint federated YOLO26

> **How to use this file.** Put it in the root of the existing project (for example as `docs/FED_YOLO26_ASYNC_SPEC.md`) and ask Claude Code: *"Read docs/FED_YOLO26_ASYNC_SPEC.md and implement it in this repository following Section 7, one task at a time, running each acceptance test before moving on."* Appendix A contains a tested reference module (`fedyolo26_async.py`). Appendix B contains the tests that were run against it. Adapt them to the project's structure; do not paste them in blindly.

---

## 0. Scope and non-negotiables

- **Model:** Ultralytics YOLO26 detection (`yolo26{n,s,m,l,x}.yaml`, `end2end: True`, `reg_max: 1`). It has a dual head: one-to-many (`cv2`/`cv3`) and one-to-one, NMS-free (`one2one_cv2`/`one2one_cv3`).
- **Verified against:** `ultralytics==8.4.164`. **Pin this version.** Several hooks below (`E2ELoss.o2m`, `model.class_weights`, `_setup_train`, `set_class_weights`) are internal APIs. Re-verify them after any upgrade (task T0).
- **Federation:** asynchronous. Nodes pull the global model whenever they are free, train locally, and push a delta. The server never waits for all nodes.
- **Data:** nodes have disjoint or partially disjoint label sets. All nodes must use **one global class-index map** (the union taxonomy, same `nc` and the same `names` order in every node's `data.yaml`). A node that has no class *c* still keeps class *c* as an index; it simply has no boxes of it.
- **Goal metric:** gap to a centrally trained YOLO26 on the union of node subsets: ΔmAP50-95 = mAP(central) − mAP(federated), on a held-out union test set, overall and per class.

## 1. Recommended strategy (what to implement and why)

**Composite: class-aware CA2FL + label-masked local loss + absent-class distillation from the pulled global model + global-progress schedules. Add post-hoc head re-calibration at milestones in Phase 2.**

| Layer | Choice | What it fixes | Source |
|---|---|---|---|
| Async protocol | **Buffered asynchronous aggregation** (step after M arrivals) | No waiting for slow nodes; bounded server work | FedBuff, arXiv:2106.06639 |
| Heterogeneity under async | **Server cache of each node's latest update, used to calibrate every global step (CA2FL)** | With class-disjoint nodes, a buffer of M arrivals often contains only a few label sets. The cache keeps absent nodes' directions in every step, so classes owned by slow nodes are not forgotten between their arrivals | CA2FL, Wang et al., ICLR 2024 (openreview 4aywmeb97I) |
| Staleness | Polynomial down-weighting α(s) = (1+s)^−a, plus a discard threshold | Very stale deltas are applied with less weight | FedAsync, arXiv:1903.03934 |
| Head aggregation | **Per-class-row weighting of the final classification 1×1 convolutions** (weight `[nc,C,1,1]`, bias `[nc]`) of both heads. Row *c* is averaged only over nodes with n_{k,c} > 0 | Nodes without class *c* cannot move row *c*. This matters because weight decay and **MuSGD's orthogonalised update mix rows**: in our test, masked rows still changed locally | FedCPD / class-wise aggregation; UltraFlwr partial aggregation, arXiv:2503.15161 |
| Local loss | **Mask the BCE of classes not annotated at the node** (via `model.class_weights`) | Prevents "absent class = background" negatives (background ambiguity) | Yao et al. arXiv:2001.04621; Zhao et al. arXiv:2008.06614; FedRS |
| Local loss | **Logit distillation on absent classes from the pulled global model** (teacher), for both heads | Keeps the node from erasing what the others taught about classes it lacks, and keeps shared features discriminative for them | FedNTD arXiv:2106.03097; ConDistFL arXiv:2308.04070; FedVLS arXiv:2401.02329; FedLMD arXiv:2409.13136 |
| Schedules | ProgLoss o2m→o2o weights and learning rate driven by **global progress**, not local epochs | Ultralytics resets both at every `train()` call; left alone, every round restarts the schedules | verified in source (§3) |
| Phase 2 | **Post-hoc classification-head re-calibration** from per-class feature statistics (CCVR-style) | Removes residual classifier bias after async training | CCVR arXiv:2106.05001; Fed3R arXiv:2406.01116 |
| Ablation | FedAdam-style adaptive server step (`server_opt="adam"`) | Robustness to heterogeneous update scales | Reddi et al. arXiv:2003.00295; FADAS arXiv:2407.18365 |

**Why not the alternatives for a first run:**
- Weight-space merging (TIES/DARE/Fisher) and neuron matching do not address class-disjointness.
- Server-side ensemble distillation (FedDF) needs unlabeled server images plus a box-fusion pipeline. Add it only if a public unlabeled pool exists.
- Pseudo-labelling is added only for nodes flagged "absent classes may be present but unannotated" (Phase 3, §6.3).

## 2. Two cases the data manifest must declare

For every node *k* and every class *c* outside *S_k* (the node's label set), record in the node manifest whether *c* is:
1. `absent`: the class truly never appears. Masking plus KD is sufficient.
2. `unannotated`: it may appear but is not labelled. Masking plus KD is still correct. Pseudo-labels (Phase 3) can add positives.

Never train a node with `mask_absent=False` unless its annotation is exhaustive over the union taxonomy.

## 3. Verified Ultralytics facts and gotchas (8.4.164)

1. **Classification-row keys:** `model.23.cv3.{0,1,2}.2.{weight,bias}` and `model.23.one2one_cv3.{0,1,2}.2.{weight,bias}` (12 tensors for P3–P5). Use the regex in `cls_row_keys()`, not hard-coded indices. The layer index changes with P2/P6 variants.
2. **The one-to-one head is detached from the backbone** (`x_detach` in `Detect.forward`). Only the one-to-many branch trains the backbone and neck.
3. **Loss:** `DetectionModel.init_criterion()` returns `E2ELoss(model)`, which wraps two `v8DetectionLoss` instances (topk 10, and topk 7 with topk2 = 1). The classification BCE is `BCEWithLogitsLoss(reduction="none")` of shape `(B, anchors, nc)`, multiplied by `model.class_weights.view(1,1,nc)` if that attribute exists at criterion construction. **This is the mask hook.** `set_class_weights()` runs in `_setup_train()` before the EMA copy, so override it there.
4. **ProgLoss:** `E2ELoss.o2m` starts at 0.8 and decays linearly to 0.1 over `hyp.epochs`, stepped by `criterion.update()` at each epoch end. With short local rounds this schedule restarts every round. Drive it from global progress (`AbsentClassKDLoss._set_prog`).
5. **`optimizer="auto"` picks AdamW with a tiny fitted LR when iterations < 10 000**, which is always true for a local round. Always pin `optimizer="MuSGD"` (YOLO26 default recipe) or `"SGD"`, with an explicit `lr0`.
6. **Gradient accumulation:** `accumulate = round(nbs / batch)` with `nbs=64`. If a node's round has fewer than `accumulate` batches, **zero optimizer steps happen** (observed in testing: deltas were exactly 0). Set `nbs=batch`.
7. **LR schedule:** set `lrf=1.0` (constant inside a round). The server passes `lr0` from a global cosine schedule. Warmup: Ultralytics warms up for `min(warmup_epochs, epochs−1)` epochs, so `epochs=1` means no warmup. Use `warmup_epochs=0` except on a node's first round.
8. **Mosaic:** `close_mosaic` is relative to local epochs. Set `close_mosaic=0` locally, and let the server switch `mosaic=0.0` for the final ~10 % of global progress.
9. **EMA:** a new `ModelEMA` is created in every `_setup_train`. Send the **raw** model weights (`unwrap_model(trainer.model)`), not EMA.
10. **Built-in `distill_model`/`DistillationModel`** does feature-level distillation through a projector (extra parameters, all classes). **Do not use it here.** Use the class-selective logit distillation in `AbsentClassKDLoss`.
11. **Evaluation:** the validator selects the inference head with `nms`. Use `nms=False` to evaluate the NMS-free one-to-one head (the YOLO26 deployment path). Also report the default (`nms=None`, one-to-many + NMS). A saved checkpoint `{"model": model.half(), "train_args": {}, "epoch": -1}` with `model.names` and `model.args` set loads with `YOLO(path).val(...)`.
12. **Sandboxed or HPC nodes:** set `YOLO_CONFIG_DIR` to a writable directory, and `YOLO_OFFLINE=1` if nodes have no internet.

## 4. Architecture

```
           +--------------------------- server ----------------------------+
 pull() -> | global weights w_t, version t, lr(t), progress p=t/T, flags   |
           |                                                               |
 push() -> | staleness s=t-tau ; alpha=(1+s)^-a ; drop if s>s_max           |
           | diff_k = Delta_k - cache_k(old) ; cache_k <- Delta_k ;        |
           | buffer += (k, diff_k, alpha)                                   |
           | if |buffer| >= M:                                             |
           |    v = h_bar(snapshot) + sum_b w_b * diff_b                    |
           |       (class rows: w per class c = n_{b,c}/sum n_{.,c} * a_b)  |
           |    w_{t+1} = w_t + eta_g * v  (or Adam on v)                   |
           |    BN stats = image-weighted mean of cached node stats         |
           |    t += 1 ; snapshot <- cache ; checkpoint + async eval        |
           +---------------------------------------------------------------+
 node k:  loop { (w,t,lr,p,flags) = pull(); train E local epochs with
                 BCE mask(S_k) + lambda*KD_absent(teacher=w) ; push(Delta, t, n_{k,c}, n_img, BN) }
```

**Messages** (serialize tensors with `safetensors` or `torch.save` to bytes; compress fp16 deltas if bandwidth matters):
- `PullResponse{version:int, weights:dict[str,Tensor], lr:float, progress:float, mosaic:bool, kd_lambda:float, round_epochs:int}`
- `PushRequest{node_id:str, base_version:int, delta:dict[str,Tensor], bn:dict[str,Tensor], class_counts:list[float](nc), n_images:int, train_seconds:float, loss_items:dict}`

**Concurrency:** the server serializes `push()` behind a lock. `pull()` returns a copy. Global steps are atomic with respect to `pull()`.

## 5. Algorithms (reference: Appendix A)

### 5.1 Server step (class-aware CA2FL)
For non-row tensors:
v = Σ_{i∈snapshot, age ≤ A_max} p_i h_i + Σ_{b∈buffer} α_b (m_b / Σ m) (Δ_b − h_b^{old}),
where p_i is the image share, m is image count, and h^{old} = 0 for a first arrival.

For row *c* of each classification-row tensor, replace the image shares with class shares n_{i,c}/Σ_j n_{j,c} (0 if nobody owns *c*).

BN `running_mean`/`running_var` are absolute values: image-weighted mean over all cached nodes. `num_batches_tracked` is left untouched.

Defaults: `M = max(2, ceil(K/3))`, a = 0.5, s_max = 20, A_max = 10, η_g = 1.0, `server_opt="none"`.

### 5.2 Client local objective
L = L_box + L_l1 + λ_cls·Σ_{c∈S_k} BCE_c + λ_kd·λ_cls·Σ_{branch} w_branch·BCE(s_{c∉S_k}/T, σ(t_{c∉S_k}/T)) / Σσ(t/T),
summed over both branches with the ProgLoss weights (o2m, o2o). The teacher is the frozen pulled global model in `eval()` mode.

Defaults: λ_kd = 1.0 (tune in {0.25, 0.5, 1, 2}), T = 2.0.

**KD warm start:** send `kd_lambda=0` until the server's cache covers every class at least once and version ≥ `kd_start_version` (default 2·K). Before that, the teacher's rows for unseen classes are randomly initialised and distilling them adds noise.

### 5.3 Global schedules (computed by the server and sent in `PullResponse`)
- `progress = t / T_planned`, where T_planned = planned global versions.
- `lr = lr_min + 0.5(lr0 − lr_min)(1 + cos(π·progress))` with lr0 = 0.01 and lr_min = 1e-4 for MuSGD/SGD.
- `mosaic = progress < 0.9`.
- `progress_per_epoch = 1 / (T_planned · M / K)`, so the ProgLoss decay spans the whole federation. This is approximate and fine.

## 6. Phases

### 6.1 Phase 1: core (must work end-to-end)
Server + client as above. Keep all nodes on the same initialisation. The server creates the model once from `yolo26{scale}.yaml` with the union `nc`, loads COCO-pretrained `yolo26{scale}.pt` with shape-matching transfer (backbone/neck/box head; classification rows re-initialised), and broadcasts it as version 0.

### 6.2 Phase 2: head re-calibration (milestones: every R versions and at the end)
1. The server broadcasts `w_t` together with a `calibration` request.
2. Each node runs the frozen global model on its train images (no augmentation). It gets positive anchors from `criterion.one2many.get_assigned_targets_and_loss(...)[0]` → `fg_mask, target_gt_idx`. For each FPN level and each head, it collects the **input features of the final classification conv** (output of `cv3[l][1]` / `one2one_cv3[l][1]`, dimension c3 = max(ch[0], min(nc, 100))) at positive anchors, grouped by GT class. It also collects a background sample from anchors with max IoU < 0.1 to any GT.
3. Each node uploads per-class and background counts, means and covariances per (head, level).
4. The server samples virtual features from the pooled Gaussians (mixture pooling across nodes). It refits **only** the final 1×1 classification conv weight and bias of each (head, level) with BCE for 200–500 steps, starting from `w_t`, with per-class balanced sampling. It publishes the result as a new version.

**Acceptance:** union-val mAP50-95 does not decrease, and per-class AP for classes owned by few nodes increases. If mAP decreases, keep the previous head.

### 6.3 Phase 3 (optional): pseudo-labels for `unannotated` classes
Override `preprocess_batch` in the client trainer to append teacher detections of classes outside *S_k* with confidence ≥ 0.6 as extra targets (`cls`, `bboxes` in normalised xywh, `batch_idx`). Set `class_weights` for those classes to 0.5 instead of 0. Disable KD for those classes to avoid double supervision. Only enable this for nodes whose manifest marks the class `unannotated`.

## 7. Implementation tasks for Claude Code (in order; each ends with its acceptance test)

- **T0 – Recon.** Read the existing repo. Identify the current FL transport (Flower, gRPC, HTTP, MQ?), where YOLO training is invoked, the data layout, and the node configs. Write `docs/fed_integration_notes.md` mapping this spec onto the repo. Pin `ultralytics==8.4.164`. Re-verify facts 1, 3, 4, 5 and 6 of §3 by grepping the installed package.
  *Accept:* notes file exists; `cls_row_keys()` returns 12 keys on `yolo26n.yaml` with nc = union size.
- **T1 – Union taxonomy and manifests.** Add `configs/taxonomy.yaml` (ordered class names). Validate every node `data.yaml` against it. Add a node manifest: `node_id`, `label_set`, per-class status `absent|unannotated`, `n_images`.
  *Accept:* a script fails loudly on any index or name mismatch.
- **T2 – Server aggregator.** Port `ClassAwareCA2FLServer` (Appendix A) into the server package, behind a lock. Add checkpointing of `w`, `version`, `cache`, `snapshot` and the optimizer moments, so the server can restart.
  *Accept:* Appendix B `test_fed.py` passes: row updates `[1,1,2,3]` then `[5,5,2,3]`, generic update 6.0, stale push discarded.
- **T3 – Client trainer.** Port `FedClientTrainer`, `AbsentClassKDLoss` and `local_overrides`. Pin the optimizer, `nbs=batch`, `lrf=1`, `close_mosaic=0`, and send raw weights.
  *Accept:* Appendix B `test_client.py` on the synthetic 3-node set. `class_weights` must equal the node mask. Deltas must be nonzero. `o2m` must follow global progress, not restart. The direct KD check must show `cls(full) > cls(base)` when λ > 0.
- **T4 – Transport.** Implement `pull`/`push` on the existing transport. If using Flower, do **not** use a synchronous `Strategy.aggregate_fit`. Run the aggregator as an independent service, or use Flower only as a message channel. Add retry/idempotency on `push` (keyed by node_id + base_version).
  *Accept:* 3 simulated nodes with different sleep times run concurrently for 20 versions; the log shows staleness > 0 events and no deadlock.
- **T5 – Global schedules and KD warm start** (§5.2, §5.3) on the server.
  *Accept:* unit test of the lr, mosaic and kd flags versus version.
- **T6 – Evaluation service.** After every global step (or every N), write `global_v{t}.pt` and evaluate it asynchronously on the union val set with `nms=False` and with `nms=None`. Log mAP50-95, mAP50 and per-class AP; the per-class AP is split by how many nodes own the class.
  *Accept:* CSV `metrics/global_eval.csv` with one row per version.
- **T7 – Baselines and ablation switches.** Centralized YOLO26 (same epochs-equivalent budget); local-only; synchronous FedAvg; FedBuff (`use_cache=False, class_aware_rows=False, kd_lambda=0, mask_absent=False`); +cache (CA2FL); +class rows; +mask; +KD; +server Adam; +Phase 2 re-calibration. Everything is controlled from one YAML config.
  *Accept:* `make ablation` (or equivalent) produces one results table with ΔmAP versus centralized.
- **T8 – Phase 2 head re-calibration** (§6.2). *Accept:* §6.2 criterion.
- **T9 (optional) – Phase 3 pseudo-labels** (§6.3).

## 8. Experiment protocol

- Seeds: ≥ 3. Report mean ± sd.
- Partitions: (i) natural (each node = one biomedical source); (ii) fully class-disjoint; (iii) partially disjoint, each class at *m* of *K* nodes; (iv) Dirichlet α ∈ {1.0, 0.3, 0.1}.
- Asynchrony: simulate node speed ratios {1:1:1, 1:2:4, 1:5:20}; report staleness histograms.
- Budget parity: compare at equal total local image-epochs with the centralized run.
- Primary metric: ΔmAP50-95 (central − fed) on the union test set, NMS-free head. Secondary: per-class AP by node coverage; wall-clock to reach 95 % of central mAP; bytes transferred.

## 9. Pitfalls checklist
- [ ] Same `nc`, class order and init checkpoint on all nodes.
- [ ] `optimizer` pinned; `nbs=batch`; `lrf=1`; `close_mosaic=0`; `val=False` locally.
- [ ] Raw weights (not EMA) sent; delta computed against the exact pulled weights.
- [ ] BN running stats sent as absolute values, not deltas.
- [ ] KD off until class coverage is complete.
- [ ] A node with exhaustive annotation may set `mask_absent=False`; everyone else uses `True`.
- [ ] Evaluate with `nms=False` (deployment head).
- [ ] Differential privacy / secure aggregation: the cache stores per-node deltas in plain form on the server. If this is a problem for your deployment, encrypt the cache at rest. Phase 2 feature statistics are more sensitive than deltas.

## 10. References
FedAsync arXiv:1903.03934 · FedBuff arXiv:2106.06639 · CA2FL ICLR 2024 openreview.net/forum?id=4aywmeb97I · FADAS arXiv:2407.18365 · FedStaleWeight arXiv:2406.02877 · FedAdam arXiv:2003.00295 · FedRS KDD'21 doi:10.1145/3447548.3467254 · FedNTD arXiv:2106.03097 · FedLMD arXiv:2409.13136 · FedVLS arXiv:2401.02329 · ConDistFL arXiv:2308.04070 · CCVR arXiv:2106.05001 · Fed3R arXiv:2406.01116 · Cross-dataset detection arXiv:2001.04621 · Unified label space arXiv:2008.06614 · FedSTO arXiv:2310.17097 · UltraFlwr arXiv:2503.15161 · Ultralytics YOLO26 arXiv:2606.03748.

---

## Appendix A — `fedyolo26_async.py` (reference implementation, tested)

```python
"""
fedyolo26_async.py -- reference implementation (verified against ultralytics==8.4.164, YOLO26 end2end head)

Asynchronous, class-aware federated training of Ultralytics YOLO26 detectors on nodes with
disjoint / partially disjoint label sets.

Components
----------
1. ClassAwareCA2FLServer   : buffered asynchronous aggregation with a server-side cache of each node's latest
                             update (CA2FL, Wang et al. ICLR 2024), polynomial staleness down-weighting (FedAsync),
                             and per-class-row weighting of the classification output convolutions
                             (only nodes that own class c move row c).
2. FedClientTrainer        : DetectionTrainer subclass for one local round on one node:
                             - per-class BCE masking of classes the node does not annotate (via model.class_weights)
                             - logit-level distillation of absent classes from the received global model (teacher)
                             - ProgLoss (o2m->o2o) and LR driven by *global* progress instead of local epochs
3. helpers                 : cls_row_keys(), state-dict delta utilities, class-count extraction.

This file does NOT implement transport (gRPC/Flower/HTTP). Wire `ClassAwareCA2FLServer.pull()/push()` to your
existing messaging layer.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import unwrap_model

# --------------------------------------------------------------------------------------------------------------
# State-dict helpers
# --------------------------------------------------------------------------------------------------------------
_BN_STAT = re.compile(r"\.(running_mean|running_var)$")
_BN_COUNT = re.compile(r"\.num_batches_tracked$")


def cls_row_keys(model_or_sd, n_levels: int | None = None) -> list[str]:
    """Return state-dict keys of the final 1x1 classification convs (weight [nc,C,1,1] and bias [nc]) of BOTH the
    one-to-many (cv3) and one-to-one (one2one_cv3) YOLO26 heads. Row c of these tensors == class c."""
    sd = model_or_sd.state_dict() if isinstance(model_or_sd, nn.Module) else model_or_sd
    pat = re.compile(r"^model\.\d+\.(cv3|one2one_cv3)\.(\d+)\.2\.(weight|bias)$")
    keys = [k for k in sd if pat.match(k)]
    if not keys:
        raise RuntimeError("No classification-row keys found; check the Detect head layout of this ultralytics version")
    return keys


def float_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached fp32 CPU copy of all floating-point entries of the state dict (params + BN running stats)."""
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items() if v.is_floating_point()}


def delta(new: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Parameter delta new - base, excluding BN running statistics (these are sent as absolute values)."""
    return {k: new[k] - base[k] for k in base if not _BN_STAT.search(k)}


def bn_stats(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in sd.items() if _BN_STAT.search(k)}


def class_counts_from_dataset(dataset, nc: int) -> np.ndarray:
    """Per-class instance counts from an ultralytics YOLODataset (labels list)."""
    cls = np.concatenate([lb["cls"].flatten() for lb in dataset.labels], 0) if len(dataset.labels) else np.zeros(0)
    return np.bincount(cls.astype(int), minlength=nc).astype(np.float64)


# --------------------------------------------------------------------------------------------------------------
# Server: class-aware CA2FL with staleness weighting
# --------------------------------------------------------------------------------------------------------------
@dataclass
class _Cached:
    delta: dict[str, torch.Tensor]
    bn: dict[str, torch.Tensor]
    class_counts: torch.Tensor  # [nc]
    n_images: float
    version: int  # global version the client trained from
    arrived: int = 0  # global version at arrival


@dataclass
class ServerConfig:
    buffer_size: int = 3  # M: number of arrivals that trigger a global step (FedBuff/CA2FL)
    staleness_exp: float = 0.5  # alpha(s) = (1+s)^-a  (FedAsync polynomial)
    max_staleness: int = 20  # arrivals older than this many versions are discarded
    server_lr: float = 1.0  # eta_g; 1.0 == plain CA2FL step
    server_opt: str = "none"  # "none" | "adam" (FedAdam/FADAS-style adaptive server step on pseudo-gradient)
    adam_betas: tuple = (0.9, 0.99)
    adam_eps: float = 1e-3
    class_aware_rows: bool = True  # False -> plain CA2FL (ablation)
    use_cache: bool = True  # False -> FedBuff (ablation)
    cache_max_age: int = 10  # cached updates older than this many versions are dropped from h_bar


class ClassAwareCA2FLServer:
    """Asynchronous aggregator. Thread-safety is the caller's responsibility (wrap push() in a lock)."""

    def __init__(self, init_model: nn.Module, nc: int, cfg: ServerConfig = ServerConfig()):
        self.cfg = cfg
        self.nc = nc
        self.w = float_state(init_model)
        self.version = 0
        self.cache: dict[str, _Cached] = {}  # latest update of every client (updated on arrival)
        self.snapshot: dict[str, _Cached] = {}  # cache as it was at the previous global step (h_t in CA2FL)
        self.buffer: list[tuple] = []  # (client, calibrated diff, alpha, cache record)
        self.row_keys = set(cls_row_keys(self.w))
        self.m = {k: torch.zeros_like(v) for k, v in self.w.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.w.items()}
        self.log: list[dict] = []

    # ---- API used by the transport layer ----
    def pull(self) -> tuple[dict[str, torch.Tensor], int]:
        return {k: v.clone() for k, v in self.w.items()}, self.version

    def push(self, client_id: str, d: dict[str, torch.Tensor], base_version: int, class_counts, n_images: int,
             bn: dict[str, torch.Tensor] | None = None) -> bool:
        """Receive one local update. Returns True if a global step was applied."""
        s = self.version - base_version
        if s > self.cfg.max_staleness:
            self.log.append(dict(event="discard", client=client_id, staleness=s, version=self.version))
            return False
        alpha = (1.0 + s) ** (-self.cfg.staleness_exp)
        old = self.cache.get(client_id)
        if self.cfg.use_cache and old is not None:
            diff = {k: d[k] - old.delta[k] for k in d}
        else:
            diff = {k: d[k].clone() for k in d}
        self.cache[client_id] = _Cached(
            delta={k: t.clone() for k, t in d.items()}, bn=bn or {},
            class_counts=torch.as_tensor(class_counts, dtype=torch.float32), n_images=float(n_images),
            version=base_version, arrived=self.version)
        self.buffer.append((client_id, diff, alpha, self.cache[client_id]))
        self.log.append(dict(event="push", client=client_id, staleness=s, alpha=alpha, version=self.version))
        if len(self.buffer) >= self.cfg.buffer_size:
            self._step()
            return True
        return False

    # ---- internals ----
    def _row_weights(self, ids, counts_fn):
        """[n_ids, nc] matrix of per-class weights normalised over ids (0 where nobody owns the class)."""
        W = torch.stack([counts_fn(i) for i in ids])  # [n, nc]
        tot = W.sum(0, keepdim=True)
        return torch.where(tot > 0, W / tot.clamp(min=1e-12), torch.zeros_like(W))

    @staticmethod
    def _apply_rows(t: torch.Tensor, w_rows: torch.Tensor) -> torch.Tensor:
        # t: [nc, ...]; w_rows: [nc]
        return t * w_rows.view(-1, *([1] * (t.dim() - 1)))

    def _step(self):
        cfg = self.cfg
        buf = self.buffer
        # ---- h_bar: mean of the cached updates as they were at the previous step (CA2FL h_t)
        snap = self.snapshot if cfg.use_cache else {}
        h_ids = [i for i in snap if self.version - snap[i].arrived <= cfg.cache_max_age]

        def img_w(ids, src):
            n = torch.tensor([src[i].n_images for i in ids], dtype=torch.float32)
            return n / n.sum().clamp(min=1e-12)
        upd = {}
        # buffer weights (staleness x images); normalised by images only so staleness shrinks the step
        b_ids = [cid for cid, _, _, _ in buf]
        b_rec = [rec for _, _, _, rec in buf]
        b_alpha = torch.tensor([a for _, _, a, _ in buf], dtype=torch.float32)
        b_img = torch.tensor([r.n_images for r in b_rec], dtype=torch.float32)
        b_w = b_alpha * b_img / b_img.sum().clamp(min=1e-12)
        if cfg.class_aware_rows:
            B_rows = torch.stack([r.class_counts for r in b_rec])  # [nb, nc]
            B_rows = torch.where(B_rows.sum(0, keepdim=True) > 0, B_rows / B_rows.sum(0, keepdim=True).clamp(min=1e-12),
                                 torch.zeros_like(B_rows)) * b_alpha.view(-1, 1)
            H_rows = self._row_weights(h_ids, lambda i: snap[i].class_counts) if h_ids else None
        hw = img_w(h_ids, snap) if h_ids else None
        for k in buf[0][1]:
            if k in self.row_keys and cfg.class_aware_rows:
                corr = sum(self._apply_rows(diff[k], B_rows[j]) for j, (_, diff, _, _) in enumerate(buf))
                hbar = (sum(self._apply_rows(snap[i].delta[k], H_rows[j]) for j, i in enumerate(h_ids))
                        if h_ids else torch.zeros_like(corr))
            else:
                corr = sum(b_w[j] * diff[k] for j, (_, diff, _, _) in enumerate(buf))
                hbar = sum(hw[j] * snap[i].delta[k] for j, i in enumerate(h_ids)) if h_ids else torch.zeros_like(corr)
            upd[k] = hbar + corr
        # ---- server optimizer
        for k, u in upd.items():
            if cfg.server_opt == "adam":
                b1, b2 = cfg.adam_betas
                self.m[k].mul_(b1).add_(u, alpha=1 - b1)
                self.v[k].mul_(b2).addcmul_(u, u, value=1 - b2)
                self.w[k] += cfg.server_lr * self.m[k] / (self.v[k].sqrt() + cfg.adam_eps)
            else:
                self.w[k] += cfg.server_lr * u
        # ---- BN running stats: image-weighted mean of the latest stats of every cached client
        bn_ids = [i for i in self.cache if self.cache[i].bn]
        if bn_ids:
            ww = img_w(bn_ids, self.cache)
            for k in self.cache[bn_ids[0]].bn:
                self.w[k] = sum(ww[j] * self.cache[i].bn[k] for j, i in enumerate(bn_ids))
        self.snapshot = dict(self.cache)
        self.version += 1
        self.log.append(dict(event="step", version=self.version, buffer=[c for c in b_ids]))
        self.buffer = []

    def global_model(self, template: nn.Module) -> nn.Module:
        m = copy.deepcopy(template)
        sd = m.state_dict()
        sd.update({k: v.to(sd[k].dtype) for k, v in self.w.items()})
        m.load_state_dict(sd)
        return m


# --------------------------------------------------------------------------------------------------------------
# Client: masked + absent-class-distilled local training
# --------------------------------------------------------------------------------------------------------------
class AbsentClassKDLoss:
    """Wraps ultralytics E2ELoss. Adds logit distillation from a frozen teacher (the received global model) on the
    classes the node does NOT annotate, for both the one-to-many and one-to-one branches. Drives the ProgLoss
    o2m/o2o weights from global progress instead of local epoch count."""

    def __init__(self, base, teacher: nn.Module | None, absent: torch.Tensor, lam: float, T: float,
                 progress: float, progress_per_epoch: float):
        self.base = base
        self.teacher = teacher
        self.absent = absent  # LongTensor of absent class indices
        self.lam, self.T = lam, T
        self.progress = progress
        self.progress_per_epoch = progress_per_epoch
        self._set_prog()

    def __getattr__(self, name):  # forward everything else (e.g. .one2many, .hyp) to the base criterion
        return getattr(self.base, name)

    def _set_prog(self):
        b = self.base
        if hasattr(b, "o2m_copy"):
            p = min(max(self.progress, 0.0), 1.0)
            b.o2m = (1 - p) * (b.o2m_copy - b.final_o2m) + b.final_o2m
            b.o2o = max(b.total - b.o2m, 0)

    def update(self):  # called by the trainer at every local epoch end
        self.progress += self.progress_per_epoch
        self._set_prog()

    def __call__(self, preds, batch):
        loss, items = self.base(preds, batch)
        if self.teacher is None or self.lam <= 0 or self.absent.numel() == 0:
            return loss, items
        preds = self.base.one2many.parse_output(preds)
        with torch.no_grad():
            t = self.teacher(batch["img"])
            t = t[1] if isinstance(t, tuple) else t
        bs = batch["img"].shape[0]
        idx = self.absent.to(batch["img"].device)
        branches = [("one2many", getattr(self.base, "o2m", 1.0)), ("one2one", getattr(self.base, "o2o", 0.0))]
        kd = 0.0
        for name, wgt in branches:
            if name not in preds or wgt == 0:
                continue
            s = preds[name]["scores"].index_select(1, idx).float()  # [B, |absent|, A]
            tt = t[name]["scores"].index_select(1, idx).float()
            tgt = torch.sigmoid(tt / self.T)
            l = F.binary_cross_entropy_with_logits(s / self.T, tgt, reduction="sum")
            kd = kd + wgt * l / tgt.sum().clamp(min=1.0)
        kd = kd * self.lam * self.base.one2many.hyp.cls * bs
        loss = loss.clone()
        loss[1] = loss[1] + kd  # fold into the cls component (keeps ultralytics logging intact)
        return loss, items


class FedClientTrainer(DetectionTrainer):
    """One local round on one node.

    Required extra overrides (pass via `fed=` dict):
        global_sd     : dict[str, Tensor]  global weights pulled from the server
        label_set     : list[int]          classes annotated at this node (global class indices)
        progress      : float in [0,1]     global training progress (server version / planned versions)
        progress_per_epoch : float         how much progress one local epoch represents
        kd_lambda     : float              weight of absent-class distillation (0 disables)
        kd_T          : float              distillation temperature
        mask_absent   : bool               zero the BCE of classes not in label_set
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, fed: dict | None = None):
        self.fed = fed or {}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        gsd = self.fed.get("global_sd")
        if gsd is not None:
            sd = model.state_dict()
            missing = [k for k in gsd if k not in sd]
            assert not missing, f"global state has unknown keys: {missing[:5]}"
            sd.update({k: v.to(sd[k].dtype) for k, v in gsd.items()})
            model.load_state_dict(sd)
        return model

    def set_class_weights(self):
        """Replace ultralytics inverse-frequency weights with the node's label mask (optionally x inv-freq)."""
        super().set_class_weights()
        model = unwrap_model(self.model)
        nc = self.data["nc"]
        mask = torch.zeros(nc)
        mask[list(self.fed.get("label_set", range(nc)))] = 1.0
        if not self.fed.get("mask_absent", True):
            mask = torch.ones(nc)
        base = getattr(model, "class_weights", None)
        w = mask if base is None else base.detach().cpu().float() * mask
        model.class_weights = w.to(self.device)

    def _setup_train(self):
        super()._setup_train()
        model = unwrap_model(self.model)
        nc = self.data["nc"]
        label_set = set(self.fed.get("label_set", range(nc)))
        absent = torch.tensor([c for c in range(nc) if c not in label_set], dtype=torch.long)
        teacher = None
        if self.fed.get("kd_lambda", 0) > 0 and absent.numel() and self.fed.get("global_sd") is not None:
            teacher = copy.deepcopy(model).eval()
            tsd = teacher.state_dict()
            tsd.update({k: v.to(tsd[k].dtype) for k, v in self.fed["global_sd"].items()})
            teacher.load_state_dict(tsd)
            for p in teacher.parameters():
                p.requires_grad_(False)
            if hasattr(teacher, "class_weights"):
                del teacher.class_weights
        base = model.init_criterion()  # built AFTER set_class_weights so the mask is used by both branches
        model.criterion = AbsentClassKDLoss(
            base, teacher, absent, lam=self.fed.get("kd_lambda", 0.0), T=self.fed.get("kd_T", 1.0),
            progress=self.fed.get("progress", 0.0), progress_per_epoch=self.fed.get("progress_per_epoch", 0.0))

    def local_result(self, base_sd: dict[str, torch.Tensor]) -> dict:
        """Package the update to send to the server (raw weights, not EMA)."""
        model = unwrap_model(self.model)
        new = float_state(model)
        nc = self.data["nc"]
        counts = class_counts_from_dataset(self.train_loader.dataset, nc)
        return dict(delta=delta(new, base_sd), bn=bn_stats(new), class_counts=counts,
                    n_images=len(self.train_loader.dataset))


def local_overrides(data_yaml: str, model_cfg: str, epochs: int, lr: float, first_round: bool, mosaic_on: bool,
                    imgsz: int = 640, batch: int = 16, device="", project="runs/fed", name="node", optimizer="MuSGD"):
    """Ultralytics overrides that make a short local round behave like a slice of one long training run."""
    return dict(
        model=model_cfg, data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        nbs=batch,  # accumulate=1: with nbs=64 short rounds on small nodes may perform ZERO optimizer steps
        optimizer=optimizer,  # NEVER 'auto': auto picks AdamW with a tiny lr when iterations < 10k
        lr0=lr, lrf=1.0,  # constant LR inside the round; the server schedules lr across rounds
        warmup_epochs=1.0 if first_round else 0.0,
        close_mosaic=0, mosaic=1.0 if mosaic_on else 0.0,
        cls_pw=0.0, val=False, plots=False, save=False, exist_ok=True, project=project, name=name,
        pretrained=False, amp=True, verbose=False,
    )
```

## Appendix B — tests run against Appendix A

### B.1 `make_synth.py` (synthetic 3-node class-disjoint dataset)
```python
import numpy as np, os, cv2
rng=np.random.default_rng(0)
NC=4; names=['c0','c1','c2','c3']; colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
nodes={'nodeA':[0,1],'nodeB':[2],'nodeC':[3]}
def img(classes,n):
    for i in range(n):
        im=np.full((96,96,3),40,np.uint8); lines=[]
        for _ in range(rng.integers(1,3)):
            c=int(rng.choice(classes)); w,h=rng.integers(16,40,2); x,y=rng.integers(0,96-w),rng.integers(0,96-h)
            cv2.rectangle(im,(int(x),int(y)),(int(x+w),int(y+h)),colors[c],-1)
            lines.append(f"{c} {(x+w/2)/96:.4f} {(y+h/2)/96:.4f} {w/96:.4f} {h/96:.4f}")
        yield im,lines
for node,cls in list(nodes.items())+[('test',[0,1,2,3])]:
    for split in ['train','val']:
        os.makedirs(f'synth/{node}/images/{split}',exist_ok=True); os.makedirs(f'synth/{node}/labels/{split}',exist_ok=True)
        for i,(im,l) in enumerate(img(cls,16 if split=='train' else 4)):
            cv2.imwrite(f'synth/{node}/images/{split}/{i}.jpg',im); open(f'synth/{node}/labels/{split}/{i}.txt','w').write("\n".join(l))
    open(f'synth/{node}.yaml','w').write(f"path: {os.path.abspath('synth/'+node)}\ntrain: images/train\nval: images/val\nnc: {NC}\nnames: {names}\n")
print("ok")
```

### B.2 `test_fed.py` (server unit test)
```python
import torch, copy, numpy as np, os, sys
os.environ["YOLO_OFFLINE"]="1"
from ultralytics.nn.tasks import DetectionModel
from fedyolo26_async import *
torch.manual_seed(0)
NC=4
tmpl=DetectionModel("yolo26n.yaml", nc=NC, verbose=False)
keys=cls_row_keys(tmpl); print("row keys:",len(keys), keys[:2])
assert len(keys)==12  # 3 levels x 2 heads x (weight,bias)
# ---------- Test 1: class-aware rows only move for owners; generic tensors = weighted mean
srv=ClassAwareCA2FLServer(tmpl,NC,ServerConfig(buffer_size=3))
w0,_=srv.pull()
def fake_delta(val): return {k:torch.full_like(v,val) for k,v in w0.items() if not ('running_' in k)}
cc={'A':[10,10,0,0],'B':[0,0,10,0],'C':[0,0,0,10]}
for cid,val in zip('ABC',[1.,2.,3.]):
    srv.push(cid,fake_delta(val),0,cc[cid],n_images=10)
assert srv.version==1
k=keys[0]; rows=(srv.w[k]-w0[k]).flatten(1).mean(1)
print("row update after step1:",rows.tolist())
assert torch.allclose(rows,torch.tensor([1.,1.,2.,3.]))
g=[kk for kk in w0 if kk not in keys and 'running' not in kk][0]
assert torch.allclose(srv.w[g]-w0[g],torch.full_like(w0[g],2.0))  # mean of 1,2,3
# ---------- Test 2: CA2FL calibration: only A returns (twice), B/C caches keep their rows alive
w1,_=srv.pull()
srv.cfg.buffer_size=1
srv.push('A',fake_delta(5.),1,cc['A'],10)   # staleness 0
rows=(srv.w[k]-w1[k]).flatten(1).mean(1)
print("row update step2 (only A arrived):",rows.tolist())
# rows 0,1: h_bar(A=1) + (5-1)=5 ; row2: cached B=2 ; row3: cached C=3
assert torch.allclose(rows,torch.tensor([5.,5.,2.,3.]))
gg=(srv.w[g]-w1[g]).mean().item(); print("generic step2:",gg)   # h_bar mean(1,2,3)=2 + (5-1)*1 = 6
assert abs(gg-6.0)<1e-5
# ---------- Test 3: staleness down-weighting and discard
srv.cfg.max_staleness=1
srv.push('B',fake_delta(9.),0,cc['B'],10)   # staleness 2 > 1 -> discarded
assert srv.log[-1]['event']=='discard'
print("server tests passed")
```

### B.3 `test_client.py` (async simulation with real YOLO26 local rounds)
```python
import os, torch, copy, json, random
from ultralytics.nn.tasks import DetectionModel
from fedyolo26_async import *
NC=4; nodes={'nodeA':[0,1],'nodeB':[2],'nodeC':[3]}
tmpl=DetectionModel("yolo26n.yaml", nc=NC, verbose=False)
srv=ClassAwareCA2FLServer(tmpl,NC,ServerConfig(buffer_size=2,server_opt="none"))
planned=6; random.seed(0)
order=['nodeA','nodeB','nodeA','nodeC','nodeB','nodeA']  # asynchronous arrival order (A is fastest)
pulled={}
for step,node in enumerate(order):
    gsd,ver=srv.pull()
    fed=dict(global_sd=gsd,label_set=nodes[node],progress=ver/planned,progress_per_epoch=1/planned,
             kd_lambda=1.0,kd_T=2.0,mask_absent=True)
    ov=local_overrides(f"synth/{node}.yaml","yolo26n.yaml",epochs=1,lr=0.01,first_round=(step<3),mosaic_on=True,
                       imgsz=64,batch=8,device="cpu",project=os.path.abspath("runs_fed"),name=node,optimizer="MuSGD")
    ov["amp"]=False; ov["workers"]=0
    tr=FedClientTrainer(overrides=ov,fed=fed); tr.train()
    m=unwrap_model(tr.model)
    crit=m.criterion
    assert isinstance(crit,AbsentClassKDLoss)
    cw=m.class_weights.cpu().tolist()
    res=tr.local_result(gsd)
    k=cls_row_keys(gsd)[0]
    rowmove=res['delta'][k].flatten(1).abs().mean(1).tolist()
    applied=srv.push(node,res['delta'],ver,res['class_counts'],res['n_images'],bn=res['bn'])
    print(json.dumps(dict(step=step,node=node,base_version=ver,server_version=srv.version,applied=applied,
          class_weights=[round(x,3) for x in cw],o2m=round(crit.base.o2m,3),teacher=crit.teacher is not None,
          counts=res['class_counts'].tolist(),local_row_delta=[f"{x:.2e}" for x in rowmove],
          tloss={kk:round(float(vv),3) for kk,vv in tr.tloss.items()})))
print([e for e in srv.log if e['event']!='push'])
```

### B.4 `test_kd.py` (distillation term check)
```python
import os, torch
from ultralytics.nn.tasks import DetectionModel
from fedyolo26_async import *
NC=4
torch.manual_seed(1); tmpl=DetectionModel("yolo26n.yaml", nc=NC, verbose=False)
gsd=float_state(tmpl)
out={}
for lam in [0.0,1.0]:
    ov=local_overrides("synth/nodeA.yaml","yolo26n.yaml",epochs=1,lr=0.01,first_round=False,mosaic_on=False,imgsz=64,batch=8,device="cpu",project=os.path.abspath("runs_kd"),name=f"l{lam}")
    ov.update(amp=False,workers=0)
    tr=FedClientTrainer(overrides=ov,fed=dict(global_sd=gsd,label_set=[0,1],progress=0.5,progress_per_epoch=0.1,kd_lambda=lam,kd_T=2.0))
    tr.train(); out[lam]={k:round(float(v),4) for k,v in tr.tloss.items()}
    # direct check of kd magnitude on one batch
    m=unwrap_model(tr.model); m.train()
    b=next(iter(tr.train_loader)); b=tr.preprocess_batch(b)
    with torch.no_grad():
        l_full,_=m.criterion(m(b["img"]),b); l_base,_=m.criterion.base(m(b["img"]),b)
    out[f"direct_{lam}"]=(float(l_full[1]),float(l_base[1]))
print(out)
```

Run with: `export YOLO_CONFIG_DIR=$PWD/ulcfg PYTHONPATH=$PWD YOLO_OFFLINE=1; python make_synth.py && python test_fed.py && python test_client.py && python test_kd.py`
