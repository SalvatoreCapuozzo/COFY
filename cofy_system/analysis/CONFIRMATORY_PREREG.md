# Confirmatory study: pre-registration

**Registered:** 2026-10-28, before any confirmatory run was started.
**Frozen files:** the confirmatory runs use a copy of the code taken at
registration, `_frozen/confirm/` (`fedyolo/`, `tools/`; it links `data/`,
`runs/` and `configs/` back to the project). Work on the main code tree
cannot reach them. SHA-256 of every config, of the frozen code, of this
document and of the analysis script are in `analysis/confirm_manifest.json`
(`tools/confirm_manifest.py`). From `_frozen/confirm/`, `python
tools/confirm_manifest.py --check` lists any file changed since registration.

## 1. Why

The exploratory search (`analysis/SEARCH_REPORT.md`) chose two schemes after
trying about 30 variants on seeds 0–3, and set the 85 % threshold after seeing
results. Its confidence bounds are therefore optimistic. This study tests the
two chosen schemes, unchanged, on seeds never used before, with margins stated
here in advance.

## 2. What is tested (frozen)

| Scheme | Variant (configs) | Copied from | Post-processing |
|---|---|---|---|
| Synchronous: sequential, server-scheduled turns (`async_order: shuffle`, one node at a time, `swa_last: 4`) | `configs/variants/confirm_v2sync/` | `v2_async` | closing calibration round (`bn+head`) |
| Asynchronous: concurrent nodes on a virtual clock (`async_clock: virtual`, `staleness_alpha 0.5`, `swa_per_node`) | `configs/variants/confirm_v2async/` | `v2_vclock` | progressive calibration with refresh (`acal_refresh`) |
| Reference: centralized, protocol v2 | `configs/variants/v2_central/<ds>_s0.yaml` with the seed changed (`tools/run_baseline.py --protocol v2`) | — | calibration round (`bn+head`) |

The configs differ from their sources only in `seed` and `output_dir`.
Protocol v2: linear learning-rate decay to 1 %, budgets kfm12 30, WBC 20,
ChestX 56 epochs (`local_epochs 2`).

## 3. Data and seeds

- **Datasets:** kfm12, WBC, ChestX-Det10, exactly as in the study (same
  splits, node ownership, training fractions and evaluation sets;
  `analysis/PREREGISTRATION.md`).
- **Seeds:** 100–109 (n = 10). None of them was used before. The seed also
  draws the node order of the sequential scheme, the machine speeds of the
  concurrent scheme and the balanced training subsets.
- **No early stopping and no extension.** The test is done once, when all 10
  seeds are finished and calibrated on every dataset. A run that crashes is
  restarted with the same seed. Every run is reported.
- **Order:** seed by seed (centralized → sync → async for each seed).

## 4. Endpoints and test

- **Primary endpoint:** Fed:Cent = federated mAP50 / centralized mAP50 × 100,
  paired by seed, on the full evaluation set. Both sides use the
  post-processing above.
- **Statistic:** CI-low = lower end of the two-sided 95 % t-interval of the
  mean Fed:Cent over the 10 seeds, per dataset.
- **Decision:** a scheme passes margin *m* when CI-low ≥ *m* on every dataset.
  This is an intersection-union test, so no multiplicity correction is needed.
- **Margins, in fixed sequence** (each tested only if the previous one passed;
  the claim is the last margin passed):
  - synchronous: 85 %, then 90 %;
  - asynchronous: 80 %, then 85 %.

  Planning estimates from the exploratory seeds (simulated, true mean = the
  exploratory mean, spread up to 1.5× the exploratory one) put the power
  above 0.95 for sync at 85 % and for async at 80 %, but only about 0.67 for
  async at 85 %.
- **Analysis code:** `tools/confirm_analysis.py` (frozen; `--write` produces
  `analysis/CONFIRMATORY_RESULTS.md`).

## 5. Secondary and descriptive results (no test)

- Fed:Cent of mAP50-95.
- Absolute mAP50 loss (centralized − federated), with its upper 95 % bound.
- The uncalibrated rows (`confirm_v2sync`, `confirm_v2async`, `+acal`).
- Per-class and per-node retention (fairness), by `tools/fairness_analysis.py`.
- GPU time per scheme.

## 6. What will be claimed

- **Sync passes 85 %:** "the synchronous scheme keeps at least 85 % of
  centralized mAP50 on every dataset, with 95 % confidence".
- **Async passes 80 % (and 85 %):** the same claim at that margin.
- **A scheme fails its first margin:** reported as a failed confirmation; the
  exploratory bound is not claimed.

## 7. Held-out dataset

A fourth dataset, never used in the search, will be added to test whether the
result generalizes.
- **Selection criteria (fixed now):**
  - public medical object-detection data with a licence for research use;
  - at least 8 classes, 12 or more preferred;
  - a modality or task different from the three study datasets;
  - fits on this machine.
- **Naming and recording:** the dataset is named in a dated amendment
  (below) before its first run. Its name goes into
  `analysis/confirm_heldout.json`.
- **Preparation:** the same as the study's:
  - `tools/split_dataset_into_nodes.py --split-mode global`, split seed 0, `--val-frac 0.2`, 4 nodes, greedy one-class-one-node assignment;
  - training fraction chosen so the number of unique training images falls in the study's range (about 1,100–1,500).
- **Budget:** fixed by the pilot rule (`tools/budget_pilot.py`: first epoch within 2 % of the best centralized mAP50) before any federated run.
- **Test:** the same frozen schemes, seeds 100–109 and margins. The result is reported separately and regardless of outcome.
- **Replacing a study dataset:** replacing one of the three study datasets with the held-out one after seeing results would be a choice made on outcomes. If it is done, it is recorded as a deviation and the results of all four datasets are reported.

## 8. Amendments and deviations

(dated entries only)

**2026-10-28, amendment 1: held-out dataset named, before its first run.**
- **Dataset:** DENTEX, tooth-enumeration subset.
  - MICCAI 2023 challenge (Hamamci et al., arXiv:2305.19112); licence CC BY-NC-SA 4.0;
    https://huggingface.co/datasets/ibrahimhamamci/DENTEX.
  - 634 panoramic dental X-rays and 18,095 boxes in 32 classes (FDI tooth numbers 11–48), 341–630 boxes per class.
  - It meets every criterion of §7: public, licensed for research, 32 classes (≥ 12), a new modality and task (dental radiography, anatomy enumeration), 544 MB.
- **Conversion:** `tools/convert_dentex.py`, lossless (634 images and 18,095 boxes in and out). Images are re-encoded as JPEG (quality 95) at full resolution.
- **Split:** `tools/split_dataset_into_nodes.py` with the study's arguments; all six split checks pass.
  - 508 training images, 126 evaluation images (3,614 boxes).
  - Each node owns 8 tooth classes, about 4,500 boxes each.
- **Two facts follow from the data, not from a choice:**
  1. Every X-ray shows teeth of all four nodes, so every training image is held by every node, each with its own classes labelled.
  2. Only 508 unique training images exist, fewer than the study's range of about 1,100–1,500, so all are used (`balanced_fraction 1.0`).
- **Budget:** fixed by the pilot (`confirm_pilot`, seed 0), then the confirmation configs are written and hashed (`analysis/confirm_manifest_amend1.json`) before the first DENTEX confirmation run.

**2026-10-28, amendment 2: the held-out dataset is a fourth use case (user decision), before any DENTEX run.**
- **The headline claim covers all four datasets.** It uses the same test, margins and fixed sequence: CI-low ≥ margin on every one of kfm12, WBC, ChestX-Det10 and DENTEX.
- **Every result is reported:** the three-dataset test registered first, DENTEX alone, and all four.
- **No dataset is dropped:** none is replaced or removed whatever the results.
- **Analysis code:** `tools/confirm_analysis.py` gains the "all four use cases" group. This is its only change since registration. Its new SHA-256 is 233688335ba0ee06f42efb1c8a660e6382c358b411cb5568b9942cdb1b1a2fea.


**2026-11-01, amendment 3: DENTEX budget of 20 epochs (user decision), before any DENTEX federated run.**
- **The pilot** (`runs/variants/_budget/dentex/s0/curve.json`, finished 2026-10-28, centralized only) is flat from the first checkpoint: mAP50 0.945 at epoch 5, best 0.958 at epoch 15.
- **The rule as written in §7** (first epoch within 2 % of the best) gives 5 epochs, that is 3 rounds per node.
- **The written rule does not match how the three study budgets were set** (`FOLLOWUP_PLAN.md` §8.5):
  - the budget is the peak of the 3-point moving average: kfm12 30, ChestX 55 (56 in whole rounds);
  - when the curve is flat from the start, the budget is the primary protocol's 20 epochs. WBC got 20, where the written rule gives 5;
  - the written rule would also have given kfm12 35.
- **Applied the same way, DENTEX gets 20 epochs:** its moving-average peak is at epoch 20 (0.954), and its curve is flat as WBC's was.
- **Budget:** 20 epochs, which is 10 rounds (concurrent scheme: 10 cycles per node).
- **Recording:** the configs are written and hashed by `tools/make_heldout_configs.py --budget 20 --amendment 3`. `analysis/confirm_manifest_amend1.json` records both the pilot rule's result and this budget.
- **Process note:** the configs were due on 2026-10-28. The automatic step ran before the pilot had finished and stopped, so the DENTEX runs were queued only on 2026-11-01. They run in plan order, after the remaining confirmation seeds.
