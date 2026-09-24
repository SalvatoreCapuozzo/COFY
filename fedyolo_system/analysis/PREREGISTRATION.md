# Cross-dataset non-inferiority study — pre-registration

Fixed on **2026-09-23**, before any run under this protocol and before any
ChestX-Det10 result existed. Changes after this date are listed under
*Deviations*, with their date and reason; nothing is edited silently.

## Hypothesis

Synchronous and asynchronous federated training of a YOLO detector across
nodes that each own disjoint classes lose no more than 10% of the detection
performance of centralized training, on every dataset tested.

## Primary endpoint and test

For each dataset, mode (sync, async) and training seed *s*:

    r_s = mAP50(federated, s) / mAP50(centralized, s)

both scored on that dataset's held-out evaluation set at the final checkpoint.
Over n = 5 seeds, with mean r̄ and sample standard deviation sd:

    L = r̄ − t(0.975, 4) · sd / √5,   t(0.975, 4) = 2.776

**Pass:** L ≥ 0.90. This is a one-sided non-inferiority test at α = 0.025
(the lower bound of a two-sided 95% confidence interval). The hypothesis holds
only if all six tests pass: 3 datasets × {sync, async}.

## Secondary endpoints (descriptive, no pass/fail)

- Fed:Oracle ratio, mean and 95% CI, per dataset and mode.
- Absolute mAP50 and mAP50-95 of federated, centralized and oracle models.
- Spread of the mean ratios across datasets.

## Data protocol

- **Datasets:** kfm12 (12 parasite-egg classes), WBC (7 white-blood-cell
  classes), ChestX-Det10 (10 chest X-ray findings).
- **Split:** `tools/split_dataset_into_nodes.py --split-mode global`, split seed
  0, `--val-frac 0.2`, 4 nodes, greedy one-class-one-node assignment. One
  image-level train/val split is shared by every node, the evaluation set and
  the oracle.
- **Evaluation set:** the global val split, every class labeled on every image,
  deduplicated, never subsampled. Federated, centralized and oracle are all
  scored on it (`eval_data_yaml`).
- **Oracle:** trained on the global train pool with full labels.
- **Training fraction** (`balanced_fraction`): kfm12 0.1, WBC 0.1, ChestX 0.5,
  so that each dataset trains on a comparable number of unique images
  (measured below: 1,068–1,507).
- **ChestX-Det10 preparation:** official train/test merged into one pool; the
  764 images with no finding excluded; class ids in alphabetical order.

## Model and training

- yolo26-classic @ 640 (`configs/arch/yolo26_classic_l.yaml`), COCO-pretrained
  shared parameters.
- `local_epochs 2`, `batch_size 4`, `lr0 0.0008`, `warmup_steps 20`,
  `grad_clip 10`, pseudo-labeling off.
- Sync: 10 rounds. Async: 10 cycles per node, `staleness_alpha 0.5`,
  `max_concurrent_nodes 1`.
- Merge: plain FedAvg (`merge_rule: mean`, `normalize_steps: false`). Merge
  improvements such as FedNova are **not** part of the primary test; they are
  evaluated separately, and only if the primary test fails.
- Training seeds 0–4; the data split is fixed across seeds.
- A crashed run is re-run with the same seed. No seed is dropped on the basis
  of its result.

## Analysis

`tools/analyze_study.py`, written before any study result exists, computes
exactly the quantities above from each run's `summary.json`.

## Deviations

- **2026-09-23 — seed count 3 → 5.** Re-scoring the existing kfm12 runs on a
  clean evaluation set showed the 3-seed test lacks power: Fed:Cent means of
  102.4% (sync) and 99.7% (async), yet CI lower bounds of 89.8% and 86.8%,
  because t(0.975, 2) = 4.30. No ChestX-Det10 data existed at the time. The
  90% margin was not changed.
- **2026-09-23 — estimate corrected, protocol unchanged.** The first version
  said the training fractions give "~1,100–1,250 images" per dataset. Measured
  before any ChestX run, the range is 1,068–1,507 unique images. The fractions
  themselves (0.1 / 0.1 / 0.5) are unchanged.
- **kfm12 and WBC are replications, not blind tests.** Results for both under
  the earlier (flawed) per-node split were seen before this protocol was fixed.
  ChestX-Det10 is the confirmatory dataset.

## Measured data counts (filled in before training)

| Dataset | Global train / eval images | Node ownership reproduces earlier federation | Training node-images at fraction | Split invariants |
|---|---|---|---|---|
| kfm12 | 12,359 / 3,089 | yes | 1,407 at 0.1 (337 / 588 / 277 / 205) | 6/6 pass |
| WBC | 10,676 / 2,668 | yes | 1,068 at 0.1 (255 / 235 / 289 / 289) | 6/6 pass |
| ChestX-Det10 | 2,224 / 555 | new dataset | 2,078 at 0.5 (652 / 571 / 408 / 447) | 6/6 pass |

Unique training images (an image owned by two nodes counts once): kfm12
1,385, WBC 1,068, ChestX-Det10 1,507. ChestX's node-image count exceeds its
unique count because its findings co-occur. Conversion was lossless: 2,779
images, 8,339 boxes, 0 images missing, 0 boxes dropped. ChestX node
ownership: A Consolidation · B Effusion · C Atelectasis, Calcification,
Nodule, Pneumothorax · D Emphysema, Fibrosis, Fracture, Mass.
