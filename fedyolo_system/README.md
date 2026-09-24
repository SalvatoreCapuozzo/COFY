# fedyolo

Federated learning for YOLOv8/YOLO26/YOLOE object detectors where every node
starts from the same weights (blank-init or pretrained, see below), all nodes
share the same visual domain, but different nodes can own **different,
possibly non-overlapping subsets of classes** (e.g. one node only ever sees
`ascaris`/`trichuris_trichiuria` eggs, another only `ancylostoma`). The output
of training is one global multi-class model.

## Why this needs more than plain FedAvg

If you naively pool gradients across nodes with different class sets, a node
that has never annotated `ancylostoma` will, every batch, teach the model
that the `ancylostoma` channel should always say "no object" -- even on
images that might contain one. Averaged into the global model, this actively
erases what other nodes have learned about that class. This framework fixes
that with three things working together:

1. **A class-decomposed head.** The classification branch ends each
   detection scale in a 1x1 conv whose output channel `c` is literally the
   logit for class `c` (`model.{H}.cv3.{i}.2.{weight,bias}`, verified against
   the installed Ultralytics build in `fedyolo/model.py` -- and
   `one2one_cv3.{i}.2` too for YOLO26's native end2end head, see
   "Architecture support" below). Every other weight -- backbone, neck,
   box-regression branch, DFL, even the two feature-mixing convs inside the
   classification branch before that final projection -- is class-agnostic.
2. **Loss masking.** Each node zeroes the classification loss for classes it
   doesn't own (via Ultralytics' built-in `model.class_weights` hook), so it
   never penalizes a class it has no ground truth for.
3. **Class-conditional aggregation.** For each per-class output channel, only
   the nodes that actually own that class are averaged together
   (instance-count-weighted). A class owned by one node is simply copied
   from that node until others learn it too. Every other, shared weight
   FedAvgs across all nodes (sample-weighted) -- but *how and when* that
   averaging happens is a real design choice, not a single fixed algorithm,
   and this framework implements two different ones, picked by the CLI's
   `--mode` flag (`sync-async` | `sync` | `async` -- see Quickstart below;
   `federation.mode` in the config still sets sync/async-specific defaults
   like `rounds`/`staleness_alpha`, but `--mode` decides which actually run):

   - **`mode: sync`** (`fedyolo/server.py`'s `FedServer`) -- classic
     round-barrier FedAvg. Every round: broadcast the current global model to
     all nodes, all nodes train `local_epochs` locally and independently, the
     server *waits* until every node has reported back (a hard barrier), then
     aggregates once and advances to round `r+1`. Simple and
     theoretically well-studied, but a round moves at the pace of its
     slowest node.
   - **`mode: async`** (`fedyolo/server.py`'s `AsyncFedServer`) -- immediate,
     staleness-discounted delta aggregation, no barrier at all. Each node
     independently pulls the current global model, trains `local_epochs`
     locally, and pushes its update the moment it's done; the server merges
     each push as it arrives, under a lock, weighting it by
     `w = 1 / (1 + staleness_alpha x staleness)` where `staleness` is how
     many global versions have advanced since that node pulled. A node
     training a bigger model or bigger image size just contributes a more
     stale, more discounted update instead of blocking the other three.
     Across this project's own measured results, async consistently
     reproduces far more tightly across repeated runs of the identical
     config than sync does (see the current-state report's non-determinism
     callout) -- a direct, code-level consequence of sync's node-processing
     order being fixed while async's is a genuine wall-clock race between
     independent node threads.

   Both modes share the identical per-node training step
   (`fedyolo/client.py`'s `run_node_round`) and the identical class-decomposed
   aggregation split (`fedyolo/model.py`'s `per_class_param_names()`,
   `aggregate()`) -- they only differ in *when* a node's update reaches the
   server and *how* the server merges it in.

There's an optional fourth piece, **cross-round pseudo-labeling**
(`fedyolo/pseudo_label.py`): once the global model is decent, each node
distills the previous round's global model's confident predictions for
classes it doesn't own into its own training, closing more of the gap.

See `fedyolo/model.py`, `fedyolo/client.py`, `fedyolo/server.py` for the
actual mechanics and `fedyolo/centralized.py` for the comparison baseline.

## Honest caveats

- **No literal guarantee.** How close the federated model gets to a
  pooled/centralized model depends on how disjoint the class sets are, how
  much per-class data each node has, total compute (rounds x local epochs),
  and how non-iid the images themselves are. This framework minimizes the
  *federation-specific* loss; it can't manufacture information nodes never
  collected.
- **What "centralized baseline" means here.** `fedyolo/centralized.py`
  trains one model directly via SGD over the union of all nodes' images,
  with the *same* per-node class masking the federation uses (since a node's
  images genuinely don't have ground truth for classes it doesn't own,
  pooling them doesn't fix that). This isolates the cost of federation
  *specifically* from the cost of partial labels, which both approaches face
  equally. If you have a separate, fully cross-annotated dataset (every
  image labeled for every class), evaluate against that too -- it's a
  stronger, different baseline (the true label-complete ceiling).
- **Blank-weight YOLO training is data/compute hungry, and image resolution
  matters more than you'd expect.** Real microscopy eggs are small relative
  to the frame -- too low an `imgsz` gives the classifier head feature maps
  too coarse to ever resolve them, and it gets stuck predicting "background"
  everywhere (sigmoid outputs stuck near 0%) no matter how long you train.
  `model.pretrained: true` (the default) helps a lot here too -- disabling it
  collapsed both federated and centralized to near-zero mAP50 at a
  budget that reached 100%+ of centralized with it on (see the current-state
  report's leaderboard). If your detector seems to be making no progress,
  check both of these before assuming a deeper bug: inspect raw classifier
  logits (see "Debugging a stuck classifier" below) before tuning learning
  rate further.
- **This is a single-machine simulation** (`ProcessPoolExecutor`, one OS
  process per node, real isolation -- a node's process never touches another
  node's data or memory). For actual cross-machine deployment, swap the
  `ProcessPoolExecutor` in `fedyolo/server.py` for a network transport (gRPC,
  HTTP, or a library like Flower) carrying the same state_dict payloads;
  `client.run_node_round` and `model.aggregate` don't need to change.

## Validated end-to-end, on real data

Every number this framework has ever produced comes from the real
`kfm_250423_clean12` microscopy dataset (12 real parasite-egg classes across
4 nodes with genuinely non-overlapping ownership -- see "Which classes each
node owns" below), not a synthetic stand-in -- an earlier version of this
project used a tiny synthetic toy dataset for a fast smoke test, but that's
been dropped: results on shapes standing in for eggs don't transfer to
whether the real thing works, and every claim in this README should be
checkable against a real run. The current best result: **0.651 mAP50**,
federated reaching **109% of its own centralized baseline**
(`yolo26-classic@l640`, sync, 12-class/20-epoch) -- full leaderboard (every
architecture x scale x mode combination tried, ranked and with wall-clock
times) in the [current-state report](https://claude.ai/code/artifact/caa25a33-1b24-4b0e-be7d-6fcde03d54a4),
full run-by-run history in the [field report](https://claude.ai/code/artifact/2a78ddc2-3ee6-41ce-987f-1328dc4005fa).

## Install

```bash
pip install -r requirements.txt
```

## Quickstart (your own real data)

There's no bundled example dataset -- point this at your own from the start,
since a toy/synthetic stand-in doesn't tell you anything about whether it'll
work on the real thing. Split one pooled dataset into a federation testbed,
optionally build the full-annotation oracle node the same way, then run:

```bash
# 1. Split your own global YOLO-format dataset into per-node datasets
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/my_federation \
    --n-nodes 4 \
    --drop-classes bubble,dirt \
    --config-out configs/my_federation.yaml

# 2. (optional) build the oracle node the same way -- one node, every class
#    labeled on every image, for the --also-oracle baseline below
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/my_oracle \
    --n-nodes 1 \
    --drop-classes bubble,dirt \
    --config-out configs/my_federation_oracle.yaml

# 3. Run it -- both sync and async, plus centralized and oracle, in one command
python -m fedyolo.simulate --config configs/my_federation.yaml \
    --also-centralized \
    --also-oracle data/my_oracle/node_A/data.yaml
```

`--mode` picks which federated training(s) run (default shown above --
`sync-async`, both):
- `--mode sync-async` (default) -- trains **both** sync and async from the
  same config, one after another, into their own
  `runs/<output_dir>/<session>/{sync,async}/` subfolders.
- `--mode sync` / `--mode async` -- only that one.

`--also-centralized`/`--also-oracle` each train **once** per invocation and
are shared across whichever federated mode(s) ran -- since neither
`centralized.py` nor `oracle.py` actually depends on sync vs. async (both
just match the same total local-epoch budget either way), there's no reason
to retrain them per mode. This is also why `--mode sync-async` is the
default rather than `sync`: running the sync and async configs as two
*separate* invocations, the old way, meant training centralized and oracle
twice for what's nominally the same baseline experiment -- pure redundant
compute, and each of those two independent trainings ends up with a
slightly different number anyway (see the non-determinism discussion in
the current-state report). One combined invocation fixes both problems at
once. `--also-centralized` also trains and reports the non-federated,
still partial-label baseline (`fedyolo/centralized.py`).
`--also-oracle <data.yaml>` also trains and reports the TRUE
full-annotation ceiling (`fedyolo/oracle.py`) -- distinct from
`--also-centralized`, see "Centralized vs. Oracle" below. Both are
optional but recommended: together they fill in the full
federated / centralized / oracle comparison this framework is built to
answer, not just the federated number(s) on their own.

`federation.device` defaults to `auto` and resolves itself per-machine at load
time (CUDA if available, else Apple MPS, else CPU — both verified to work, no
CUDA-only code paths involved) — nothing to edit moving this config between
machines. Pin an explicit `cuda`/`cuda:0`/`mps`/`cpu` in the config instead if
you need one specific backend regardless of what's detected.

Outputs land in `runs/<output_dir>/<session>/`: each federated mode's
checkpoints under its own `{sync,async}/` subfolder (a checkpoint per round,
plus that mode's `global_final.pt`), and one top-level `summary.json` with
per-class and overall mAP under `federated_sync`/`federated_async` (whichever
ran) and, if requested, `centralized`/`oracle`.

## Using your own data

Each node needs a standard Ultralytics YOLO dataset: an `images/{train,val}`
+ `labels/{train,val}` layout plus a `data.yaml`:

```yaml
path: data/your_node
train: images/train
val: images/val
names:
  0: ascaris
  1: trichuris_trichiuria
```

Label `.txt` files use **local** class indices (0..k-1) matching the order
of `names` above -- this must exactly match the `owned_classes` order you
give that node in the federation config:

```yaml
nodes:
  - name: node_A
    data_yaml: data/your_node/data.yaml
    owned_classes: [ascaris, trichuris_trichiuria]   # index 0 -> ascaris, index 1 -> trichuris_trichiuria
```

`global_classes` in the config is the full canonical class list across every
node; `fedyolo/config.py` validates that every global class is owned by at
least one node and that owned-class lists don't contain duplicates or
unknown names.

If your data is in COCO JSON format, convert it first:

```bash
python tools/coco_to_yolo.py --json path/to/annotations.json --images path/to/images --out data/your_node
```

## Splitting one global dataset into per-node datasets

If you have a single pooled YOLO-format dataset (one `images/` + `labels/` +
`classes_list.txt`, all classes mixed together) rather than data that's
already physically separated by site, `tools/split_dataset_into_nodes.py`
turns it into a federation testbed:

```bash
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/from_global \
    --n-nodes 4 \
    --drop-classes bubble,dirt \
    --config-out configs/from_global_federation.yaml
```

Run with `--dry-run` first to see the proposed class-to-node split and
per-node image/instance counts without writing anything.

What it does, and the choices it makes:

- **Class-to-node assignment is balanced automatically** -- classes are
  sorted by total instance count and greedily dropped onto whichever node
  currently has the smallest total, so no node ends up starved while
  another hoards every common class. `--n-nodes` controls how many.
- **An image can land in more than one node.** If a single image has both a
  Trichuris box (owned by node_A) and an Ascaris box (owned by node_B), it's
  included in *both* nodes' datasets, but each copy's label file only keeps
  that node's own classes -- exactly the partial-annotation scenario the
  framework's loss masking handles. Images whose only annotations are
  dropped classes are excluded entirely (no owned class present).
- **`--drop-classes`** (default `bubble,dirt`) removes named classes from
  every label file and the global class list entirely, before any node
  assignment happens -- use this for artifact/distractor classes that
  aren't real detection targets you want a node to ever own.
- **Label index base (0-based vs 1-based) is auto-detected** from what's
  actually in your label files (some annotation tools export 1-based ids
  matching a human-facing "select the class" number instead of standard
  YOLO 0-based ids). It's conclusive when class id `0` appears anywhere, or
  when the max id used equals the class count; otherwise it defaults to
  0-based with a warning. Override with `--label-index-base 0` or `1` if
  you know better than the auto-detection.
- **`--config-out`** writes a ready-to-use federation config pointing at the
  generated node datasets. It ships with placeholder `federation.rounds` /
  `local_epochs` / `imgsz` values copied from the validated example --
  review these against your actual dataset size before training (see the
  "Honest caveats" section above; the right budget depends on your dataset
  size, which you'll need to tune for -- don't assume the values that worked
  for the 12-class real dataset in this report transfer directly).

Validated end-to-end on the real `kfm_250423_clean12` dataset this whole
project is built around: class dropping, the 0-based/1-based auto-detection
(both directions), the balanced split, the multi-node image assignment, and
the resulting node datasets loading correctly through `fedyolo.data` were
all checked against real runs, not a synthetic stand-in.



| field | meaning |
|---|---|
| `global_classes` | canonical class list, shared index space across all nodes |
| `model.arch` | Ultralytics architecture yaml. Stock YOLOv8 (`yolov8n.yaml`/`yolov8s.yaml`/...) or the YOLO26 classic-head bridge (`configs/arch/yolo26_classic_n.yaml`/`_s`/`_m`/`_l`/`_x.yaml` -- see "Architecture support" below). |
| `model.imgsz` | input resolution; see the caveat above about small objects needing larger imgsz |
| `model.pretrained` | default `true`: SHARED params (backbone/neck/box-head/DFL) init from the arch's official COCO checkpoint (downloaded on first use); the per-class head always stays random regardless. Set `false` to reproduce the original fully-blank/from-scratch behavior. |
| `federation.device` | default `"auto"`: resolves to CUDA if available, else Apple MPS, else CPU, at config-load time. Set an explicit `"cuda"`/`"cuda:0"`/`"mps"`/`"cpu"` to pin it instead. |
| `federation.rounds` / `local_epochs` | total local epochs per node = rounds x local_epochs |
| `federation.lr0` / `warmup_steps` / `grad_clip` | every local round reloads fresh global weights into a new optimizer, so a short per-round LR warmup + gradient clipping matters a lot for stability from random init |
| `federation.data_fraction` | use only the first N% of each node's TRAIN images (Ultralytics' own naive prefix slice -- not class-aware). For fast smoke-tests only; can silently drop a rare class to zero images on an imbalanced dataset. |
| `federation.balanced_fraction` | class-stratified alternative to `data_fraction`, applied to train AND the pooled eval set: every class keeps `max(1, round(count * balanced_fraction))` of its images. `None` (default) disables it; takes priority over `data_fraction` when set. Use this instead of `data_fraction` on any real, imbalanced dataset. |
| `federation.pseudo_label.*` | optional cross-round self-distillation for not-owned classes (see above) |
| `nodes[].owned_classes` | which global classes this node has ground truth for, in local-label-index order |

## Architecture support

The per-class/shared parameter split this system is built around (see
`fedyolo/model.py`'s module docstring) needs exactly one thing from a
detection head: a final class-indexed 1x1 conv named `cv3.{i}.2` per scale.

| Arch | Status | Notes |
|---|---|---|
| YOLOv8 (`yolov8{n,s,m,l,x}.yaml`) | supported | the original architecture this framework was built against; still the largest single leaderboard generation |
| YOLO26, classic head (`configs/arch/yolo26_classic_{n,s,m,l,x}.yaml`) | supported, verified | YOLO26's backbone/neck (C3k2 + C2PSA) with the detect head forced back to `end2end: False` -- the *same* `Detect` class YOLOv8 uses, so `cv3.{i}.2` naming is identical and zero changes to the aggregation/loss code were needed. `reg_max` is left at 1 (stock YOLO26's own value, not forced to 16 as an earlier version of this bridge did) -- `v8DetectionLoss` handles `reg_max=1` fine, and keeping it lets the box-regression branch transfer from the real checkpoint too, not just backbone/neck. These files don't need to physically exist: `model.py`'s `_resolve_arch_cfg()` synthesizes them from `configs/arch/yolo26_classic.yaml` + a scale letter (avoids colliding with ultralytics' own filename-scale guessing, which would otherwise silently resolve to the *stock*, end2end, yolo26.yaml). `_pretrained_ckpt_name()` maps them to the real `yolo26{scale}.pt` checkpoint for the SHARED-weight transfer. |
| YOLO26, native end-to-end (stock `yolo26*.yaml`) | supported, verified | `end2end: True` by default -- forward returns `{"one2many":..., "one2one":...}` and adds a second `one2one_cv3.{i}.2` head. `per_class_param_names()` now splits both (6 pairs instead of 3); the loss uses `E2ELoss` (ultralytics' own default for end2end models, not `E2EDetectLoss`); `client.py`'s pseudo-labeling reads the nested dict directly for both training and eval-mode (tuple-wrapped) teacher output. |
| YOLOE (`yoloe*.yaml`) | supported, verified | `YOLOEDetect`'s `cv3` outputs a fixed-width visual embedding, not a per-class logit -- there is genuinely no `cv3.{i}.2` to split, so `per_class_param_names()` returns `[]` and every weight FedAvgs as SHARED. Class identity comes from text-prompt embeddings (`text_model: mobileclip2:b`) computed once per class list and cached as a plain, non-`state_dict` attribute (`model.set_classes()`) -- frozen prototypes, not a learned per-node parameter. `_pretrained_ckpt_name()` maps YOLOE archs to the stock `yolo26{scale}.pt` checkpoint too -- ultralytics' own `yolo26.yaml` and `yoloe-26.yaml` have byte-for-byte identical backbones/necks, only the final head layer differs, so the shared backbone transfers just as well as any other arch's (confirmed: 606/822 tensors transfer, vs 0/822 when this fix was first found missing). |

## Centralized vs. Oracle -- two different baselines, don't conflate them

`--also-centralized` (`fedyolo/centralized.py`) is **not** a full-annotation
baseline: it's one model, one optimizer, cycling through every node's
*own, still partial* labels in sequence, with no aggregation step. It
isolates the cost of federation itself from the cost of partial labels,
which both federated and this baseline face equally.

`--also-oracle` (`fedyolo/oracle.py`) is the actual full-annotation
ceiling: one "node" built from `split_dataset_into_nodes.py --n-nodes 1`,
which keeps every image's *complete* multi-class label set instead of
filtering it per node. A federated/oracle ratio answers "what does
federating cost compared to a model that saw everything"; a
federated/centralized ratio answers "what does federating cost compared to
a model with the same information federation has, just not aggregated."
They can (and do) give very different numbers -- report which one a ratio
is against.

See the architecture schematic in the [current-state report](https://claude.ai/code/artifact/caa25a33-1b24-4b0e-be7d-6fcde03d54a4) for how sync/async aggregation lines up with this split, and the [full field report](https://claude.ai/code/artifact/2a78ddc2-3ee6-41ce-987f-1328dc4005fa) for the run-by-run history.

## Debugging a stuck classifier

If mAP stays at exactly 0 no matter how long you train, check raw logits
before tuning hyperparameters further:

```python
import torch
from fedyolo.config import load_config
from fedyolo.model import build_model

cfg = load_config("configs/your_config.yaml")
ckpt = torch.load("runs/your_run/global_final.pt")
model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz)
model.load_state_dict(ckpt["state_dict"]); model.eval()

img = torch.rand(1, 3, cfg.model.imgsz, cfg.model.imgsz)  # or a real validation image
with torch.no_grad():
    _, raw = model.forward(img)
print(raw["scores"].sigmoid().amax(dim=(0, 2)))  # per-class max confidence anywhere in the image
```

If every class is stuck under ~1%, the classifier hasn't learned to fire at
all yet -- almost always more local epochs, a larger `imgsz` relative to
object size, or both, rather than a bug in the aggregation logic.

## Project layout

```
fedyolo/
  config.py        federation YAML schema + validation
  model.py          build_model(), the shared/per-class parameter split, aggregation
  data.py           per-node dataset loading with local -> global class id remapping
  client.py         one node's local training for one round (runs in its own process)
  server.py         round orchestration: FedServer (sync) / AsyncFedServer (async)
  pseudo_label.py    optional cross-round self-distillation for not-owned classes
  centralized.py     non-federated, still-partial-label baseline trainer (--also-centralized)
  oracle.py           true full-annotation-ceiling baseline (--also-oracle)
  pooling.py         materializes a pooled, class-remapped val set on disk for evaluation
  evaluate.py        runs Ultralytics' own validator on the pooled val set
  simulate.py         CLI entry point
tools/
  split_dataset_into_nodes.py      splits one pooled global dataset into a balanced federation
                                    (--n-nodes 1 also builds the --also-oracle node)
  coco_to_yolo.py                  COCO JSON -> per-node YOLO dataset converter
configs/
  kfm12_*.yaml                     real-data federation configs, one per arch/scale/mode
```

## How to launch

Same flow as the Quickstart above, shown here with concrete paths actually
used in this project's own runs (see the reports linked above for what came
out of them). None of these pass `--mode`, so each one trains both sync and
async by default; add `--mode sync` or `--mode async` to run just one.

```
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/from_global \
    --n-nodes 4 \
    --drop-classes bubble,dirt \
    --config-out configs/from_global_federation.yaml
python tools/split_dataset_into_nodes.py \
    --dataset /path/to/dataset_folder \
    --out data/from_global_oracle \
    --n-nodes 1 \
    --drop-classes bubble,dirt \
    --config-out configs/from_global_federation_oracle.yaml
python -m fedyolo.simulate \
    --config configs/from_global_federation.yaml \
    --also-centralized \
    --also-oracle data/from_global_oracle/node_A/data.yaml
```
Example (remote Linux GPU box):
```
python tools/split_dataset_into_nodes.py \
    --dataset /home/salvatorecapuozzo/kfm_250423_subdataset/export \
    --out data/kfm_250423 \
    --n-nodes 4 \
    --drop-classes bubble,dirt \
    --config-out configs/kfm_250423_fed_260713.yaml
python tools/split_dataset_into_nodes.py \
    --dataset /home/salvatorecapuozzo/kfm_250423_subdataset/export \
    --out data/kfm_250423_oracle \
    --n-nodes 1 \
    --drop-classes bubble,dirt \
    --config-out configs/kfm_250423_fed_260713_oracle.yaml
python -m fedyolo.simulate \
    --config configs/kfm_250423_fed_260706.yaml \
    --also-centralized \
    --also-oracle data/kfm_250423_oracle/node_A/data.yaml
```
Same example on this Mac -- only `--dataset` changes (`federation.device: auto`
already resolves itself per-machine; see `../legacy/server_paths.yaml` for the same
Linux/macOS split used by the legacy client/server scripts, for paths other
than the dataset):
```
python tools/split_dataset_into_nodes.py \
    --dataset /Users/salvatorecapuozzo/kfm_250423_subdataset/export \
    --out data/kfm_250423 \
    --n-nodes 4 \
    --drop-classes bubble,dirt \
    --config-out configs/kfm_250423_fed_260713.yaml
python tools/split_dataset_into_nodes.py \
    --dataset /Users/salvatorecapuozzo/kfm_250423_subdataset/export \
    --out data/kfm_250423_oracle \
    --n-nodes 1 \
    --drop-classes bubble,dirt \
    --config-out configs/kfm_250423_fed_260713_oracle.yaml
python -m fedyolo.simulate \
    --config configs/kfm_250423_fed_260706.yaml \
    --also-centralized \
    --also-oracle data/kfm_250423_oracle/node_A/data.yaml
```
