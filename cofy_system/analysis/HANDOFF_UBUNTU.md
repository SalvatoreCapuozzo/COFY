# Handoff: running the non-inferiority study on the Ubuntu / RTX 3080 machine

**Audience:** the coding agent on the Ubuntu machine (RTX 3080, 8 GB VRAM).
Read this whole file before running anything. Work from `cofy_system/`
unless a step says otherwise. Written 2026-09-24 on the original machine
(Apple M2 Pro, macOS), where the study was started.

---

## 0. Summary

- **What:** 15 training runs — 3 datasets × 5 seeds — of a *pre-registered*
  statistical study. Each run trains four models: federated sync, federated
  async, centralized, oracle.
- **Status:** 0 of 15 primary runs are done on this machine. All 15 run here.
- **Your job:** set up the environment → rebuild the data and verify it is
  byte-identical to the original → smoke-test on the GPU → record the hardware
  change → launch the runner → monitor → run the analysis → report.
- **The protocol is fixed.** Do not change any hyperparameter, config,
  dataset split, seed or threshold. If something forces a change, stop and ask
  the user (§3).

## 1. What the study tests

Read `analysis/PREREGISTRATION.md` in full — it is the authority, this file
only operationalizes it. In short:

- **Hypothesis:** synchronous and asynchronous federated training of a YOLO
  detector, across 4 nodes that each own *disjoint* classes, keep at least 90%
  of centralized detection performance on every dataset.
- **Test:** per seed, r = mAP50(federated) / mAP50(centralized). Over 5 seeds,
  the lower bound of the two-sided 95% CI of the mean r must be ≥ 0.90. Six
  tests (3 datasets × {sync, async}); all six must pass.
- **Model:** yolo26-classic @ 640 (`configs/arch/yolo26_classic_l.yaml`),
  plain FedAvg. Configs: `configs/study/{kfm12,wbc,chestx}_s{0..4}.yaml`.
- **Datasets:** kfm12 (12 parasite-egg classes, microscopy), WBC (7
  white-blood-cell classes, microscopy), ChestX-Det10 (10 chest X-ray
  findings). ChestX-Det10 is the confirmatory dataset; kfm12 and WBC are
  replications (their earlier results were seen before the protocol was fixed).

Three constraints hold for everything in this project, including any later
follow-up work: nodes never learn another node's classes except through the
model they download; the server never holds labeled images, only weights;
class ownership stays one class per node.

## 2. Current status (2026-09-24)

- The protocol, configs, split tool, runner and analysis script are all
  committed.
- On the Mac, the runner finished **one** run — kfm12 seed 0, on Apple MPS, in
  427 min — then stopped on its low-disk guard. Result:

  | kfm12 seed 0 (Mac, MPS) | mAP50 | Fed:Cent | Fed:Oracle |
  |---|---|---|---|
  | federated sync | 0.6316 | 99.0% | 75.7% |
  | federated async | 0.6801 | 106.6% | 81.5% |
  | centralized | 0.6382 | — | — |
  | oracle | 0.8341 | — | — |

- **Decision:** all 15 primary runs execute on this machine, so every primary
  result comes from the same hardware. The Mac's kfm12 seed 0 is kept only as a
  cross-hardware replication and is **not** part of the primary analysis. Do not
  copy it into `runs/study/`.

## 3. Rules — do not break these

1. **Never edit** `configs/study/*.yaml`, `configs/arch/*`,
   `tools/split_dataset_into_nodes.py`, `tools/analyze_study.py`, or the
   thresholds in `analysis/PREREGISTRATION.md`.
2. **Never lower `batch_size`, `imgsz`, epochs or rounds** to fit memory. If the
   smoke test runs out of VRAM, stop and report (§6).
3. **One training job on the GPU at a time.** The runner enforces this; do not
   start `cofy.simulate` by hand while it runs.
4. **Never drop or re-roll a seed because of its result.** A crashed run is
   re-run with the same seed (the runner does this on relaunch).
5. **Do not use** `run_matrix.sh` / `tools/run_matrix.py` (legacy, flawed
   per-node split) or `tools/prepare_chestx.sh` (hard-coded Mac paths; the
   ChestX labels arrive ready-made in `data/_pool/chestx/`).
6. Anything that changes the protocol is recorded under *Deviations* in
   `analysis/PREREGISTRATION.md`, dated, with the reason — nothing silently.
7. If a step's check fails, **stop and report**. Do not improvise a
   workaround.

## 4. What you need

### 4.1 Code

The repo at commit `eb23de27` or later, **plus** the commit that adds
`tools/study_manifest.py`, `tools/rebuild_study_data.py`,
`analysis/study_manifest.json`, `configs/arch/yolo26_classic.yaml` and this
file. Check they exist:

```bash
cd cofy_system
ls tools/study_manifest.py tools/rebuild_study_data.py analysis/study_manifest.json \
   configs/arch/yolo26_classic.yaml configs/study/chestx_s4.yaml analysis/PREREGISTRATION.md
```

If `configs/arch/yolo26_classic.yaml` is missing, stop: every model build
depends on it (it was hidden by a `yolo*` rule in `.gitignore` until
2026-09-24).

### 4.2 Data (the user copies it from an external drive)

Data is not in git. The user copies these from the original machine:

| What | From (original machine) | Size | Where it goes here |
|---|---|---|---|
| Label pools | `cofy_system/data/_pool/` (kfm12, wbc, chestx) | ~5 MB of labels | `cofy_system/data/_pool/` |
| kfm12 images | `~/kfm_250423_subdataset/export/images/` | 27 GB (18,633 files; 15,448 used) | anywhere |
| WBC images | `cofy_system/data/wbc_detection/images/` | 0.35 GB (13,344) | anywhere |
| ChestX-Det10 images | `cofy_system/data/_raw/chestx_det10/extracted/` | ~1.3 GB (2,779 used) | anywhere |
| Pretrained weights | `cofy_system/yolo26l.pt` | 51 MB | `cofy_system/yolo26l.pt` |

Each pool holds `labels/`, `classes_list.txt` and `images.txt` (every image's
name and byte size). Its `images/` folder may arrive as dead symlinks, as real
files, or not at all — step 3 handles all three. If `yolo26l.pt` is missing,
ultralytics downloads it on first use (needs internet).

### 4.3 Hardware and disk

- NVIDIA driver working (`nvidia-smi` shows the RTX 3080).
- **At least 45 GB free**: ~26 GB of images, ~9 GB of run outputs (~0.6 GB per
  run × 15), logs, and the Python environment. The runner stops itself below
  5 GB free.

### 4.4 Software

- **Python 3.10** (the original environment was 3.10.13).
- **PyTorch with CUDA.** The original used torch 2.12.1 (MPS build). Install the
  same version as a CUDA build if one is available for this driver, otherwise
  the nearest release — use the selector at https://pytorch.org/get-started —
  and record the exact version in the deviation note (§5, step 5). Any current
  CUDA build supports the RTX 3080 (sm_86).
- **ultralytics==8.4.84, exactly.** cofy hooks into ultralytics internals
  (loss classes, the task-aligned assigner). `requirements.txt` says `>=8.3`,
  which is too loose for this study — pin the exact version.
- The rest of `requirements.txt` (pyyaml, Pillow, numpy, rich, psutil, …).
  numpy was 2.2.6 on the original machine.

## 5. Procedure

### Step 1 — Environment

```bash
conda create -n dafyolo python=3.10 -y && conda activate dafyolo
# install torch + torchvision with CUDA per pytorch.org, e.g. pip install torch torchvision --index-url <cuda index>
pip install ultralytics==8.4.84
pip install -r requirements.txt      # must not upgrade ultralytics; re-pin afterwards if it does
python -c "import sys,torch,ultralytics,numpy; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), ultralytics.__version__, numpy.__version__)"
```

**Check:** `torch.cuda.is_available()` is `True`, the device is the RTX 3080,
ultralytics is `8.4.84`. Otherwise stop.

Use the environment's own `python` for every command below. The runner starts
each training run with the same interpreter it was launched with.

### Step 2 — Place the data

Put `data/_pool/` under `cofy_system/data/` and `yolo26l.pt` in
`cofy_system/`. Raw image folders can live anywhere, nesting is fine:

```bash
ls data/_pool/kfm12/labels | wc -l     # 15448
ls data/_pool/wbc/labels   | wc -l     # 13344
ls data/_pool/chestx/labels | wc -l    # 2779
```

### Step 3 — Rebuild the datasets and verify them

```bash
python tools/rebuild_study_data.py \
  --kfm12-images  /path/to/kfm12/images \
  --wbc-images    /path/to/wbc/images \
  --chestx-images /path/to/chestx/extracted
```

This re-links every image by exact name, checks its byte size, re-runs the
splitter with the original arguments, and compares 138 fingerprints (splits,
eval sets, oracle sets, labels, and the per-seed training subsets) against
`analysis/study_manifest.json`.

**Check:** it ends with `ALL DATA VERIFIED -- identical to the Mac` and exit
code 0. On `STOP: images incomplete or corrupted`, report which files, so the
user can re-copy them. Any `FAIL` line is a stop — **never start the study on
unverified data.**

This was tested on the original machine in both forms a copy can take (dead
links, and real files), with all 138 fingerprints matching.

### Step 4 — Smoke test and VRAM check

A tiny end-to-end run (1 round, 1 epoch, 2% of the data), in a scratch
folder, **not** in `runs/study/`:

```bash
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/study/wbc_s0.yaml"))
c["federation"].update({"rounds": 1, "async_node_cycles": 1, "local_epochs": 1,
                        "balanced_fraction": 0.02, "eval_interval": 100})
c["output_dir"] = "runs/_smoke"
yaml.safe_dump(c, open("/tmp/smoke.yaml", "w"), sort_keys=False)
PY
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > /tmp/gpu.csv &
GPU_LOG=$!
python -m cofy.simulate --config /tmp/smoke.yaml --session smoke --mode sync-async \
  --also-centralized --also-oracle data/study/wbc_oracle/node_A/data.yaml > /tmp/smoke.log 2>&1
kill $GPU_LOG
python -c "import json; d=json.load(open('runs/_smoke/smoke/summary.json')); print({k: round(d[k]['map50'],4) for k in ('federated_sync','federated_async','centralized','oracle')})"
sort -t, -k1 -n /tmp/gpu.csv | tail -1      # peak "MiB, %"
grep -c "_eval/labels" /tmp/smoke.log       # >0: evaluation used the fully-labeled eval set
```

**Checks:**
- `summary.json` has all four keys. The mAP values are meaningless (≈0–0.2 after
  one tiny epoch) — only their presence matters.
- Peak VRAM stays below ~7,600 MiB. This is the real check for the 8 GB card:
  per-step memory depends on batch size and image size, not on how much data
  the run uses, so the smoke run's peak is representative.
- On `CUDA out of memory`, or a peak above ~7,600 MiB: **stop and report** the
  peak. Do not change `batch_size` (rule 2).

Then clean up: `rm -rf runs/_smoke /tmp/smoke.yaml`.

Note the GPU utilization column as well. Configs use `workers: 0` (images
loaded in the training process), which kept the Mac's GPU at 96–98%. A much
faster GPU may idle waiting for data. That affects only speed, never results.
**Do not change `workers`** without the user's approval; if utilization is
low, mention it in your report.

### Step 5 — Record the hardware change

Append this under `## Deviations` in `analysis/PREREGISTRATION.md`, filling in
the versions from step 1:

```markdown
- **2026-MM-DD — hardware changed: Apple M2 Pro (MPS) → NVIDIA RTX 3080 8 GB
  (CUDA).** The Mac completed one run (kfm12 seed 0) before stopping on its
  low-disk guard. To keep every primary run on identical hardware, all 15 runs
  are executed on the RTX 3080; the Mac's kfm12 seed 0 is kept as a
  cross-hardware replication and excluded from the primary analysis. Data
  were rebuilt on the new machine and verified identical to the originals
  (138/138 fingerprints). Software: Python X, torch X (CUDA X), ultralytics
  8.4.84, numpy X, NVIDIA driver X.
```

### Step 6 — Launch the study

```bash
mkdir -p runs/study
python tools/launch_detached.py runs/study/_runner.log -- "$(which python)" tools/run_study.py
sleep 30; cat runs/study/_runner.log     # expect: START kfm12 s0 ...
```

`launch_detached.py` starts the runner in its own session, so it survives the
terminal closing. (On the Mac, a plain `nohup … &` job was killed when its
parent session ended, losing a run.) The runner:

- runs in the order kfm12 s0 → wbc s0 → chestx s0 → kfm12 s1 → …, so an
  interruption leaves balanced data;
- counts a run as done only when `runs/study/<ds>/s<seed>/summary.json` holds all
  four results, and skips finished runs on relaunch;
- waits if another `cofy.simulate` is running; stops if disk falls below
  5 GB, or after two consecutive failed runs.

**Duration:** kfm12 seed 0 took 427 min on the Mac. The RTX 3080 should be
several times faster. Measure the first run and report a revised estimate for
all 15.

### Step 7 — Monitor

```bash
cat runs/study/_runner.log                                  # START / DONE / FAIL per run
tail -f runs/study/_log_kfm12_s0.log                        # live progress of the current run
python tools/analyze_study.py                               # the pre-registered test on what's finished
pgrep -fl "run_study|cofy.simulate"                      # alive?
nvidia-smi                                                  # GPU memory / utilization
df -h .                                                     # disk
```

If the runner dies, relaunch it with the step-6 command under a new log name
(`_runner2.log`). It resumes, and re-runs an unfinished seed from its start.

### Step 8 — Analyze and report

When the runner logs `ALL RUNS DONE`:

```bash
python tools/analyze_study.py      # prints the table, writes runs/study/analysis.json
```

Report to the user:

1. The verdict line: `SUPPORTED` or `NOT SUPPORTED`.
2. For each dataset and mode: the five per-seed Fed:Cent values, mean, sd, CI
   lower bound, PASS/FAIL, and the mean Fed:Oracle.
3. The measured duration per run, peak VRAM, and GPU utilization.
4. Anything unusual: failed or re-run seeds, warnings, deviations recorded.

Also write the same content to `analysis/STUDY_RESULTS.md`, so it can be
committed. `runs/` is gitignored; do not try to commit checkpoints.

## 6. Failure handling

| Symptom | Action |
|---|---|
| Step 3 reports missing or wrong-size images | Stop. Give the user the file names so they can re-copy. |
| Step 3 shows any fingerprint `FAIL` | Stop. Report which fingerprints; do not start the study. |
| `CUDA out of memory` in the smoke test or a run | Stop the runner (`pkill -f run_study`). Report the peak VRAM. Do not change batch size. |
| Runner log `FAIL <ds> s<seed>` | Read `runs/study/_log_<ds>_s<seed>.log`. If the cause is transient (e.g. killed process), relaunch the runner, which retries the seed. If it is systematic, stop and report. |
| Runner log `STOP: two failures in a row` | Systematic problem. Report both logs; do not relaunch blindly. |
| Runner log `STOP: only N GB free` | Free disk (never delete `runs/study/`), then relaunch. |
| `ModuleNotFoundError` / import errors | Environment problem; fix the environment, not the code. |
| Anything that seems to require editing a study config | Stop and ask the user. |

## 7. After the primary test (only on the user's instruction)

If the hypothesis is **not supported**, the user's plan is to try variants of
the merge step (sync and async) and the improvements in
`dafyolo_improvement_proposals.html` (repo root). Do not start them on your
own. When asked:

- The first candidate is **FedNova step normalization** (`normalize_steps:
  true` in the federation config). It is already implemented and was the only
  merge change that helped in earlier single-seed tests (WBC sync: +0.049 mAP50
  on the full eval pool, and no late-round collapse). TIES sign-election
  (`merge_rule: ties`) was tested and made results worse; do not re-propose it
  unchanged.
- Variants are **separate, labeled experiments** — new configs and a new output
  folder, never edits to `configs/study/` or `runs/study/`. Run them with the
  same datasets, seeds and test, so they are comparable.
- **User rule:** once an improvement from the proposals file has been
  implemented and tested, remove it from `dafyolo_improvement_proposals.html`,
  and report its measured result.

## 8. Reference

| Path | What it is |
|---|---|
| `analysis/PREREGISTRATION.md` | The protocol, the test, dated deviations |
| `analysis/study_manifest.json` | 138 fingerprints of the original data |
| `configs/study/*.yaml` | The 15 run configs (never edit) |
| `configs/arch/yolo26_classic.yaml` | Model architecture (`_l` suffix → scale l) |
| `tools/rebuild_study_data.py` | Rebuild + verify the data on a new machine |
| `tools/study_manifest.py` | Fingerprint writer/verifier |
| `tools/split_dataset_into_nodes.py` | The splitter (`--split-mode global`) |
| `tools/run_study.py` | Resumable sequential runner |
| `tools/launch_detached.py` | Starts a process that survives the terminal |
| `tools/analyze_study.py` | The pre-registered analysis |
| `runs/study/<ds>/s<seed>/summary.json` | Per-run results (fed sync/async, cent, oracle) |
| `runs/study/_log_<ds>_s<seed>.log` | Per-run training log |
| `runs/study/_runner.log` | Runner progress |
| `runs/study/analysis.json` | Analysis output |

Per-dataset data after step 3:

| Dataset | Train / eval images | Fraction | Unique training images | Node ownership |
|---|---|---|---|---|
| kfm12 | 12,359 / 3,089 | 0.1 | 1,385 | A eimeria, calicophoron, moniezia · B fasciola, dicrocoelium, metastrongili, trichuris · C strongili, nematodirus, strongyloides · D ascaris, ancylostoma |
| WBC | 10,676 / 2,668 | 0.1 | 1,068 | A neutrophil · B eosinophil · C erythroblast, immunoglobulin · D basophil, lymphocyte, monocyte |
| ChestX-Det10 | 2,224 / 555 | 0.5 | 1,507 | A Consolidation · B Effusion · C Atelectasis, Calcification, Nodule, Pneumothorax · D Emphysema, Fibrosis, Fracture, Mass |

The training seed also selects each node's balanced subsample, so every seed
trains on a different subset of that fraction. The data split itself is fixed
(split seed 0) across all seeds.
