# DAFYOLO — Disjoint Asynchronous Federated YOLO

## Abstract

DAFYOLO is a federated learning framework for multi-class object detection in
which participating nodes hold **disjoint subsets of class annotations** — a
common real-world constraint in distributed medical imaging, where different
sites specialize in different findings (e.g. different parasite egg taxa in
microscopy screening). It is evaluated on three datasets: kfm12 (12 parasite
egg classes, microscopy), WBC (7 white-blood-cell classes, microscopy) and
ChestX-Det10 (10 chest X-ray findings). All nodes start from the **same
initial model** — shared layers transferred from a COCO-pretrained checkpoint,
per-class output rows randomly initialized — and train within the same visual
domain. The resulting global model detects all classes jointly without any
node ever sharing its raw images.

The framework supports two aggregation strategies selectable via configuration:
a **synchronous baseline** (classic round-barrier FedAvg with class-conditional
aggregation) and the primary **asynchronous strategy** (immediate delta
aggregation with staleness discounting), which eliminates the synchronization
barrier and allows the global model to progress continuously as nodes submit
updates independently.

---

## 1. Problem Formulation

Let $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$ be the global set of $K$ classes.
There are $N$ nodes. Node $i$ owns a class subset $\mathcal{C}_i \subsetneq \mathcal{C}$
such that every class is owned by at least one node:

$$\bigcup_{i=1}^{N} \mathcal{C}_i = \mathcal{C}$$

Subsets may be fully non-overlapping ($\mathcal{C}_i \cap \mathcal{C}_j = \emptyset$
for $i \neq j$) or partially overlapping. Node $i$ holds a local dataset
$\mathcal{D}_i$ of images annotated **only** for classes in $\mathcal{C}_i$.

The goal is to train a single global detector $f_\theta$ capable of detecting
all $K$ classes, without any node transmitting raw images or labels.

The central challenge is that a node with no annotation for class $c$ will,
if trained naively, treat every occurrence of $c$ in its images as background,
generating false-negative gradients that actively suppress $c$ in the global
model after aggregation.

---

## 2. Model Architecture

DAFYOLO uses an anchor-free YOLO detector with a **decoupled head**; the
decoupled head is what makes the per-class parameter decomposition below
exact. `model.pretrained: true` is the default — the SHARED parameters below
initialize from the arch's official COCO checkpoint, the per-class head stays
random regardless (`pretrained: false` reproduces the original from-scratch
behavior).

**Model used in the study: YOLO26-classic, scale l, 640 px**
(`configs/arch/yolo26_classic_l.yaml`): YOLO26's backbone/neck (C3k2/C2PSA)
with the classic decoupled `Detect` head (`end2end: False`) instead of YOLO26's
end-to-end head. `reg_max` stays at YOLO26's own value of 1 (DFL becomes an
identity), so the box branch also transfers from the pretrained `yolo26l.pt`
checkpoint; an earlier version forced `reg_max: 16`, which silently prevented
that transfer. `ultralytics.nn.modules.head.Detect` is the identical class for
YOLOv8 and YOLO26-classic, parameterized rather than reimplemented, so
**YOLOv8** (`yolov8{n,s,m,l,x}`) is supported unchanged. YOLO26's
native end-to-end head (`nc`-indexed weights doubled into a second
`one2one_cv3` branch, both split the same way) and YOLOE's open-vocabulary
head (no `cv3.{i}.2` at all -- every weight is SHARED, class identity comes
from frozen text-prompt embeddings outside the state_dict entirely) are both
also supported now — see the field report's YOLO26/YOLOE section for what
each needed.

### 2.1 Parameter Decomposition

Every weight tensor in the model falls into exactly one of two categories:

**Shared parameters** $\theta_\text{shared}$ — the entire feature extractor:
backbone (C2f blocks), neck (PAN-FPN), box-regression branch (`cv2`), DFL
layer, and the feature-mixing convolutions inside the classification branch
(`cv3.{i}.0`, `cv3.{i}.1`). These are class-agnostic: they produce feature
representations whose semantic content does not depend on which classes a node
owns.

**Per-class parameters** $\theta^{(c)}$ — the final 1×1 projection layer of
the classification branch at each of the $n_l$ detection scales:

$$\theta^{(c)} = \left\{
  \texttt{model.H.cv3.}i\texttt{.2.weight}[c,\,:\,,\,:\,,\,:],\;\;
  \texttt{model.H.cv3.}i\texttt{.2.bias}[c]
\right\}_{i=0}^{n_l - 1}$$

where $H$ is the index of the Detect module (last module in `model.model`).
The weight tensor at each scale has shape $(K, c_\text{inner}, 1, 1)$; row $c$
is the sole learned parameter that maps the inner feature dimension directly
to the logit for class $c$. This decomposition is verified programmatically
at runtime against the installed Ultralytics build.

---

## 3. Loss Masking

Each node $i$ trains with a **binary class mask**
$\mathbf{m}_i \in \{0,1\}^K$ where:

$$[\mathbf{m}_i]_c = \begin{cases} 1 & \text{if } c \in \mathcal{C}_i \\ 0 & \text{otherwise} \end{cases}$$

The per-class BCE classification loss is element-wise multiplied by
$\mathbf{m}_i$ before summing:

$$\mathcal{L}_\text{cls}^{(i)} = \sum_{c=1}^{K} [\mathbf{m}_i]_c \cdot \text{BCE}(\hat{p}_c, y_c)$$

This ensures a node with no annotation for class $c$ contributes **exactly
zero gradient** to the $c$-th output channel — neither pushing it toward
"object" (false positive risk) nor toward "background" (false negative risk).
The box regression and DFL losses are computed only for anchor assignments
that match a ground-truth box of an owned class.

---

## 4. Federated Aggregation Strategies

### 4.1 Synchronous Mode (FedServer)

The synchronous server implements a **round-barrier variant of FedAvg** with
class-conditional head aggregation. Each round $t$:

1. **Broadcast**: all nodes receive the current global model $\theta^{(t)}$.
2. **Local training**: every node independently trains for $E$ local epochs.
3. **Barrier**: the server waits until **all** $N$ nodes have submitted.
4. **Aggregate**:

**Shared layers** — sample-weighted FedAvg across all nodes:

$$\theta_\text{shared}^{(t+1)} = \sum_{i=1}^{N} \frac{|\mathcal{D}_i|}{\sum_j |\mathcal{D}_j|}\, \theta_{\text{shared},i}$$

**Per-class head** — for each class $c$, aggregate only among nodes that own $c$,
weighted by per-class instance count $n_i^{(c)}$:

$$\theta^{(c,t+1)} = \sum_{i:\, c \in \mathcal{C}_i} \frac{n_i^{(c)}}{\sum_{j:\, c \in \mathcal{C}_j} n_j^{(c)}}\, \theta_i^{(c)}$$

The global model version advances by 1 per round.

**Limitation**: the slowest node in each round blocks every other node.
In heterogeneous deployments (nodes with very different dataset sizes or
hardware) this creates systematic under-utilisation.

### 4.2 Asynchronous Mode (AsyncFedServer) — Primary Contribution

The asynchronous server eliminates the synchronization barrier entirely.
Each node runs independently in its own thread, cycling through:

$$\text{pull} \;\rightarrow\; \text{train} \;\rightarrow\; \text{push}$$

continuously for $T$ cycles. The server aggregates **immediately** upon each
push with no waiting for other nodes. The global model **version** $v$
increments by 1 after every single-node submission; total version count
equals $N \times T$.

#### 4.2.1 Staleness

Between the moment node $i$ pulls the model at version $v_\text{pull}$ and
the moment it submits its update, other nodes may have already submitted and
advanced the server to version $v_\text{now}$. The **staleness** of the
submission is:

$$s_i = v_\text{now} - v_\text{pull} \geq 0$$

A fresh submission has $s_i = 0$. A node that trained on a model that is
$s_i$ versions old contributes an update computed from outdated information.

#### 4.2.2 Staleness-Discounted Delta Aggregation

Rather than re-averaging all nodes (impossible with only one node submitting),
DAFYOLO uses an **implicit-gradient delta update** discounted by staleness.

Let $\Delta_i = \theta_i^\text{trained} - \theta^\text{pulled}$ be the
parameter displacement the node learned during local training. The server
applies this displacement to the **current** global model, discounted by a
staleness weight:

$$w_i = \frac{1}{1 + \alpha \cdot s_i}$$

where $\alpha$ is the `staleness_alpha` hyperparameter ($\alpha = 0$ disables
discounting; higher $\alpha$ penalizes stale updates more strongly).

**For shared layers**, the full delta is applied:

$$\theta_\text{shared}^{(v+1)} = \theta_\text{shared}^{(v)} + w_i \cdot \Delta_{\text{shared},i}$$

**For per-class layers**, the delta is applied channel-by-channel, only for
owned classes:

$$\theta^{(c,v+1)} = \begin{cases}
\theta^{(c,v)} + w_i \cdot \Delta_i^{(c)} & \text{if } c \in \mathcal{C}_i \\
\theta^{(c,v)} & \text{otherwise}
\end{cases}$$

This is the key property: a node that doesn't own class $c$ leaves
$\theta^{(c)}$ **exactly unchanged**, regardless of what its local training
did to that channel.

#### 4.2.3 Convergence Properties

The delta update is equivalent to performing one step of asynchronous
stochastic gradient descent in parameter space, where each node's "gradient"
is its full local parameter update and the step size is $w_i$. Under standard
assumptions on gradient variance and diminishing staleness (which hold when
$\alpha > 0$ bounds $w_i$ away from 1 for stale updates), async-SGD-style
methods converge to a neighbourhood of the true optimum. In practice,
$\alpha = 0.5$ gives good empirical behaviour: a submission at staleness 1
receives weight 0.667 (vs. 1.0 for a fresh submission), and at staleness 5
receives weight 0.286.

---

## 5. Cross-Round Pseudo-Labeling

Starting at cycle index `start_round` (per-node), each node optionally
applies **cross-round self-distillation** for classes it does not own.
It is **off in the study** (`pseudo_label.enabled: false`): in earlier
single-seed tests it scored the same as without it (0.308 vs. 0.316 mAP50).

In **sync mode**, the teacher is the previous round's global model.
In **async mode**, the teacher is the global snapshot the node pulled at
the start of its current cycle (the model it started training from).

For each batch, the teacher is run in inference mode over the same images.
At anchor positions where the teacher is confident for an unowned class $c$
(sigmoid score $> \tau$ or $< 1-\tau$, where $\tau$ = `conf_thresh`), a
binary cross-entropy distillation loss is added:

$$\mathcal{L}_\text{distill}^{(i)} = \lambda \cdot \frac{1}{|\mathcal{A}^\tau|}
  \sum_{a \in \mathcal{A}^\tau} \sum_{c \notin \mathcal{C}_i}
  \text{BCE}\!\left(\hat{p}_{a,c},\; \sigma(z^\text{teacher}_{a,c})\right)$$

where $\mathcal{A}^\tau$ is the set of confident anchor positions and
$\lambda$ = `pseudo_label.weight`. This requires no NMS or box decoding and
works directly on the dense anchor grid. Its effect is to let information
about classes a node has never seen propagate into that node's backbone, once
the global model is reliable enough to trust.

---

## 6. Training Stability

Even with pretrained shared layers, the per-class output rows start random,
so the first batches of training can produce large gradient spikes. Local
training uses SGD (momentum 0.9, weight decay 5e-4) with a constant learning
rate `lr0` after warmup, and three mechanisms:

**Per-round/cycle LR warmup**: a linear ramp from $0$ to $\eta_0$ over the
first `warmup_steps` gradient steps of every round/cycle, because each
round/cycle reloads freshly aggregated global weights.

**Per-node momentum carried across rounds/cycles**: each node's SGD momentum
buffers are kept by the server and restored at the node's next round/cycle,
instead of restarting cold. Only the optimizer statistics persist; the
weights always come from the current global model.

**Gradient norm clipping**: gradients are clipped to a maximum L2 norm of
`grad_clip` (default 10.0) before each optimizer step.

**Image resolution note**: validated empirically — at `imgsz` $\leq 192$
pixels, the classifier head converges to predicting "background" at every
anchor even after 100+ epochs on this task. At `imgsz` $= 256$ the same
setup converges cleanly. The feature maps are too coarse at low resolution
for the anchor-object overlap to produce reliable positive assignments for
small objects. Use at least 256 pixels; 640 pixels for real microscopy data.

---

## 7. Evaluation

Every model is evaluated on a **held-out, fully annotated evaluation set**
(`eval_data_yaml`): images drawn once from the global dataset before it is
split into nodes (`split_dataset_into_nodes.py --split-mode global`), with
**every** class labeled on every image and never subsampled. Federated,
centralized and oracle models are all scored on the same images. (Without
`eval_data_yaml`, the fallback merges the nodes' own validation splits; there
each image is labeled only for its node's classes, so correct detections of
other classes count as false positives. That is why the study uses the
dedicated set.) Evaluation uses Ultralytics' standard `DetectionValidator`
pipeline directly, so mAP numbers are computed with the same NMS and
IoU-matching logic as any standard YOLO benchmark.

Two baselines, each trained once per run for the same total number of epochs
as the federated models (`rounds` × `local_epochs`):

- The **centralized baseline** trains one model directly over the union of
  all nodes' training images, with the same per-node class masking (a node's
  images genuinely lack ground truth for unowned classes whether or not
  training is federated). This isolates the cost of federation from the cost
  of partial labels, which both approaches face equally.
- The **oracle** trains one model on the same global training pool with
  **every** class labeled on every image (`--also-oracle`, built with
  `split_dataset_into_nodes.py --n-nodes 1`): the full-annotation ceiling,
  i.e. what the partial labels cost on top of federation.

---

## 8. System Architecture Summary

```
configs/
  study/{kfm12,wbc,chestx}_s{0..4}.yaml — the pre-registered study's run configs
  arch/yolo26_classic{,_n,_s,_m,_l,_x}.yaml — YOLO26 backbone + classic Detect head
  kfm12_*.yaml, wbc_*.yaml — earlier experiment configs; generate your own with
                    tools/split_dataset_into_nodes.py --config-out

cofy/
  config.py          — FedYoloConfig schema; validates class coverage and mode
  model.py           — build_model(), parameter decomposition, sync aggregation
  merge.py           — shared-parameter merge rules (sample-weighted mean = FedAvg)
  data.py            — per-node dataset loading with local→global class id remap
  sampling.py        — class-stratified subsampling (balanced_fraction)
  client.py          — run_node_round(): one node's local training (sync + async)
  server.py          — FedServer (sync) and AsyncFedServer (async)
  pseudo_label.py    — cross-round distillation loss
  centralized.py     — non-federated baseline trainer
  oracle.py          — full-annotation oracle baseline
  pooling.py         — pooled val dataset (fallback when no eval_data_yaml is set)
  evaluate.py        — calls Ultralytics val() on the evaluation set
  optim_utils.py     — per-round LR warmup helper
  gpu_cleanup.py     — kills leftover worker processes / frees GPU memory
  simulate.py        — CLI entry point; --mode picks sync-async (default,
                        both)/sync/async and dispatches to the right server(s)

tools/
  split_dataset_into_nodes.py  — splits one global dataset into per-node datasets
                                  (--split-mode global: one image-level split
                                  plus the held-out eval set; --n-nodes 1 also
                                  builds the --also-oracle node)
  coco_to_yolo.py               — COCO JSON -> per-node YOLO dataset converter
  run_study.py                  — resumable sequential runner for the study
  analyze_study.py              — the pre-registered analysis
  study_manifest.py, rebuild_study_data.py — fingerprint and rebuild the study data
  launch_detached.py            — starts a runner that survives the terminal closing
```

---

## 9. Configuration Reference

Values below are the study's (`configs/study/kfm12_s0.yaml`); comments give
the meaning and, where different, the code default.

```yaml
model:
  arch: configs/arch/yolo26_classic_l.yaml
  imgsz: 640
  pretrained: true         # shared layers from the arch's COCO checkpoint

federation:
  # mode: not set in the study configs. "sync" | "async" (code default "sync")
  # only matters for a single-mode run; the CLI's --mode flag (see §10)
  # decides which mode(s) actually run and overrides it, defaulting to both

  # shared
  local_epochs: 2          # epochs per round (sync) or per cycle (async)
  batch_size: 4
  lr0: 0.0008              # constant after warmup
  warmup_steps: 20
  grad_clip: 10.0
  workers: 0
  device: auto             # (default) resolves per-machine at load time: CUDA if
                           # available, else Apple MPS, else CPU. Pin with "cuda"/
                           # "cuda:0"/"mps"/"cpu" instead if you need a specific one.
  data_fraction: 1.0       # naive prefix slice of each node's TRAIN images
                           # (Ultralytics' own `fraction=`) -- not class-aware.
  balanced_fraction: 0.1   # class-stratified subsample of each node's training
                           # images: every class keeps max(1, round(count *
                           # balanced_fraction)) of its images; the seed picks
                           # which. Study: 0.1 kfm12, 0.1 WBC, 0.5 ChestX-Det10.
                           # Overrides data_fraction when set (default: null).
  max_concurrent_nodes: 1  # how many nodes train at the same time (see §4.2)

  # shared-parameter merge (sync)
  merge_rule: mean         # sample-weighted FedAvg (default)
  weight_basis: images     # node weight = number of training images (default)
  normalize_steps: false   # FedNova step normalization (off)

  # sync only
  rounds: 10

  # async only
  async_node_cycles: 10    # total submissions = n_nodes × async_node_cycles
  staleness_alpha: 0.5     # w = 1 / (1 + alpha × staleness)

  eval_interval: 20        # intermediate evaluation every N rounds (sync) or N
                           # submissions (async); the final model is always evaluated
  keep_checkpoints: 1      # global checkpoints kept on disk (sync rounds)

  pseudo_label:
    enabled: false         # off in the study (§5)
    start_round: 5         # cycle index (async) or round index (sync)
    conf_thresh: 0.6
    weight: 0.5

eval_data_yaml: data/study/kfm12/_eval/data.yaml   # held-out fully annotated set (§7)
seed: 0                    # also selects each node's balanced subsample
```

Options that exist for labeled follow-up experiments and are **off** in the
study: `merge_rule: ties | fisher`, `normalize_steps: true`,
`weight_basis: instances`, `exclusion_loss`, `pseudo_label.conditional`,
`pseudo_label.boxes`, `async_rule: ca2fl` (with `ca2fl_cache: false` = FedBuff)
and `async_mix` (FedAsync-style partial mixing).

---

## 10. Usage

```bash
# 1. Install
pip install -r requirements.txt

# 2. Split your own global dataset into per-node datasets
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/from_global \
    --n-nodes 4 \
    --drop-classes bubble,dirt \
    --config-out configs/my_run.yaml

# 3. (Optional) build a true full-annotation oracle node the same way,
#    with --n-nodes 1 instead of 4, for the --also-oracle baseline below
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/my_oracle \
    --n-nodes 1 \
    --drop-classes bubble,dirt \
    --config-out configs/my_run_oracle.yaml

# 4. Run federated training -- default --mode is sync-async: trains BOTH
#    sync and async from this one config, one after another
python -m cofy.simulate --config configs/my_run.yaml

# 4b. Or just one mode
python -m cofy.simulate --config configs/my_run.yaml --mode sync

# 5. Also train and compare the centralized baseline, and the oracle ceiling
#    -- each trains ONCE and is shared across whichever federated mode(s)
#    ran in step 4, rather than once per mode
python -m cofy.simulate --config configs/my_run.yaml \
    --also-centralized \
    --also-oracle data/my_oracle/node_A/data.yaml
```

The pre-registered study is run with `tools/run_study.py` (resumable; one
`cofy.simulate --mode sync-async --also-centralized --also-oracle` call per
dataset and seed), launched through `tools/launch_detached.py`.

Outputs are written to `runs/<output_dir>/`:
- **Sync**: `global_round{N:03d}.pt` for the last `keep_checkpoints` rounds only
  (older ones are deleted to save disk), `global_final.pt`
- **Async**: only the latest `global_v{V:04d}_{node_name}.pt` (earlier versions
  are deleted as each new one is written), `global_final.pt`
- `eval/` — mAP results from Ultralytics validator
- `summary.json` — per-class mAP50 and overall mAP50 / mAP50-95 for
  federated and (optionally) centralized models
- `live_training_logs.txt` — per-epoch loss for every node, every round/cycle
