# Evaluation protocol of the study

**Written:** 2026-11-12, after the runs. This document describes the evaluation reported in the article
(KFM12, WBC and DENTEX, seeds 100–109) as it was carried out. It is not a pre-registration.

**What was fixed before the first run**, and can be checked by hash:
- the code, configurations and analysis script, frozen in `_frozen/confirm/` on 2026-10-28 at 10:54 CET,
  before the first run started (`analysis/confirm_manifest.json`);
- the DENTEX configurations and label fingerprints, frozen on 2026-11-01, before DENTEX's first run on
  2026-11-02 (`analysis/confirm_manifest_amend1.json`).

From `_frozen/confirm/`, `python tools/confirm_manifest.py --check` recomputes every hash.

## 1. Schemes

| Arm | Configurations | Post-processing |
|---|---|---|
| CWT: sequential turns in a random order per cycle, one node at a time, mean of the last four global versions | `configs/variants/confirm_v2sync/` | closing calibration (`bn+head`) |
| FedAsync: concurrent nodes on a seeded virtual clock, staleness weight 1/(1 + 0.5 s), per-node final average | `configs/variants/confirm_v2async/` | progressive calibration with refresh (`acal_refresh`) |
| Centralized reference: one model, the same examples, masks and schedule | `configs/variants/v2_central/` with the seed changed | closing calibration (`bn+head`) |

Recipe for every arm: YOLO26-L with the classic head, 640 px, SGD (momentum 0.9, weight decay 5e-4), batch 4,
2 local epochs per round or update, learning rate 8e-4 decaying linearly to 1 % of it, 20-step warm-up.

## 2. Use cases and data

- **KFM12** (12 classes), **WBC** (7) and **DENTEX** (32 teeth, quadrant-enumeration subset), each split once
  at image level, 80 % training and 20 % evaluation (`tools/split_dataset_into_nodes.py --split-mode global
  --n-nodes 4 --seed 0 --val-frac 0.2`).
- Four nodes. Classes are assigned greedily so that labelled instances are balanced; every training image
  goes to each node owning a class in it, labelled only for that node's classes.
- KFM12 and WBC train on a class-stratified tenth of each node's training images, drawn with the seed
  (`balanced_fraction: 0.1`); DENTEX trains on all its training images. Each run writes the list it used.
- The evaluation set is fully labelled, deduplicated and never subsampled.
- Budgets: 30 epochs (KFM12), 20 (WBC, DENTEX).

## 3. Seeds

- Ten seeds, 100–109, not used to design or tune the schemes. Each seed also draws the training subsets,
  the node order of CWT and the machine speeds of FedAsync.
- No early stopping and no added seeds. A run that crashes is restarted with the same seed. Every run of
  these use cases and seeds is reported.

## 4. Endpoint and interval

- **Fed:Cent** = 100 × federated mAP50 / centralized mAP50, paired by seed, both after the arm's
  post-processing. mAP50 is the unweighted mean over classes of the AP at IoU 0.5 (Ultralytics 8.4.84
  validator: confidence 0.001, NMS IoU 0.7, at most 300 detections per image, 101-point interpolation).
- **Interval**: the two-sided 95 % t-interval of the mean Fed:Cent over the ten seeds, per use case, as
  computed by the frozen analysis script (`_frozen/confirm/tools/confirm_analysis.py`).

## 5. What is stated

- For each scheme and use case: the lower bound of the interval, and the highest multiple of 5 % it exceeds.
- For each scheme: the highest multiple of 5 % that its lower bound exceeds on every use case. With 95 %
  two-sided (97.5 % one-sided) confidence on each use case, the scheme keeps at least that share of
  centralized mAP50 on each of them.
- Values above 100 % are not read as a benefit of federation.

Computed by `tools/study_sensitivity.py` (`analysis/STUDY_SENSITIVITY.md`): CWT 95 % (lower bounds 96.7,
98.7, 96.2), FedAsync 90 % (92.1, 122.0, 99.4).

## 6. Descriptive analyses (no test)

- Fed:Cent of mAP50-95, and the absolute mAP50 loss with its upper 95 % bound.
- Per-class and per-node retention, and their Spearman correlation with the training boxes of each class,
  counted in the images trained on (`tools/fairness_analysis.py --confirm`).
- GPU time and energy (measured on the one GPU of the simulation), communication (computed), waiting time
  (derived from the timing model) (`tools/cost_report.py`).

## 7. Sensitivity of the lower bounds

Four further 95 % lower bounds, reported whatever they show (`tools/study_sensitivity.py`): the t-interval of
the log ratio, a BCa bootstrap (20,000 resamples), the exact Wilcoxon interval of the Hodges–Lehmann estimate,
and the t-interval with any one seed left out.

## 8. Records

- Run-level records of every run reported: `evidence/study/` (`tools/export_evidence.py`).
- Seed-level values: `analysis/study_seed_level.csv`.
