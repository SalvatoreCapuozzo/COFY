# Follow-up plan: literature-based improvements and competitors

**Date:** 2026-09-30. **Status:** labeled follow-up to the pre-registered study
(`PREREGISTRATION.md`), which stays untouched. Nothing here changes the primary
analysis; all runs go to `runs/variants/<name>/` and are compared seed by seed
with the primary study (seeds 0–2, same data, same model, same budget).

## 1. Why

On seeds 0–2 the primary study keeps federated performance close to the
centralized baseline, but far from the full-annotation **oracle** (e.g. WBC
federated sync 0.626 vs oracle 0.969). The centralized baseline, trained on
the same partial labels, is just as far from the oracle, so the gap comes from
the disjoint labels, not from federation itself. The literature agrees:

- With complete labels on every node, federated YOLO26 stays within ~2.5 % of
  centralized (0.714 vs 0.732 mAP50; arXiv 2607.02636, 2026).
- Under non-IID FedAvg the features stay good; the final classification layer
  is what's misaligned (arXiv 2512.23043, 2025).

The per-class output rows are exactly that layer here: each class's row is
trained by one node that never sees the other classes, so nothing keeps
different nodes' classes apart. Merge-rule variants tried so far (FedNova,
Fisher, TIES, CA2FL, FedBuff, FedAsync mixing) cannot add that missing signal;
the label-side additions (exclusion loss, conditional distillation) helped
most (WBC sync +0.06 to +0.07).

## 2. Constraints kept

1. Nodes learn about other nodes' classes only through the model they download.
2. The server holds no labeled data and does no training of its own (it may
   average weights).
3. One class, one owner node.

Excluded for breaking them: CCVR (server refits the classifier from uploaded
feature statistics), data-free server generators (FedGen, FedFTG), FedAlign
(shares class names through a text encoder), FedAwS's original server step.

## 3. Improvements to our method (each alone, sync and async)

| Variant | Source | Change | What it should improve |
|---|---|---|---|
| `etf`, `etf_async` | FedETF (ICCV 2023), FedGELA (NeurIPS 2023), FedNCA-ML (2025) | Per-class rows fixed to a simplex ETF (identical on all nodes, never trained; rescaled to the initial row norm); features and biases learn | Separation between look-alike classes owned by different nodes (WBC, kfm12) |
| `spread`, `spread_async` | FedAwS (ICML 2020), moved onto the nodes | Penalty keeping a node's class rows at least the ETF cosine distance from the other classes' downloaded rows (weight 1.0) | Same goal, rows keep learning |
| `ntd`, `ntd_async` | FedNTD (NeurIPS 2022), FedVLS (AAAI 2025) | Distillation of every class the node lacks, at every anchor, temperature 2, weight 1, from round 1, teacher = the model the node starts from | Nodes stop overwriting what others taught about their classes |
| `swa`, `swa_async` | FedSAM/SWA (ECCV 2022), FedSWA (ICML 2025) | Final model = average of the last global versions (sync: last 3 rounds; async: last 4 submissions = one per node) | Async: removes "last node wins" and order dependence; sync: flatter solution |

## 4. Competitors (in our setting, same budget)

| Variant | Source | Implementation notes |
|---|---|---|
| `fedprox`, `fedprox_async` | FedProx (MLSys 2020) | Proximal term, mu = 0.01 |
| `scaffold` (sync) | SCAFFOLD (ICML 2020) | Control variates, option II update; step size = lr0 (post-warmup rate) |
| `fljsddc` (sync) | FL-JSDDC (Frontiers in Neurorobotics 2026) | Self-distillation from the node's previous-round model (box L2 0.3, class term 0.3, channel term 0.4 on P3–P5 with weights 0.5/0.3/0.2). Temperatures not given in the paper: 2. Per-class sigmoid outputs use binary soft cross-entropy for the class term. The paper's drift-compensation update is under-specified; implemented in the FedDC form (CVPR 2022): penalty alpha/2 ‖θ + h − w‖², alpha = 0.01, h ← h + (θ − w), upload θ + h |

## 5. Procedure

1. CPU smoke test of every option (functional check, tiny scale).
2. Queue, after the current variant queue and the node-order check:
   improvements first (both modes), then competitors. 11 variants × 9 runs,
   about 70 GPU hours.
3. Compare with `python tools/results_table.py` (sync and async sections,
   paired difference vs the primary study, Fed:Cent, Fed:Oracle).
4. Then one run combining the improvements that helped with the exclusion
   loss (the best current addition), sync and async.

## 6. Node-order check (runs `order_r1..r3`)

Seed 0, the three node orders B/C/D-first, next to the primary study's
A-first run (async). Verified from the logs: every node trains last in exactly
one of the four runs (A-first → D last, B-first → A, C-first → B, D-first → C).

**Results (async mAP50, seed 0)**

| Dataset | A-first (primary) | B-first | C-first | D-first | Mean over orders | sd over orders | sd over seeds (primary, s0–2) |
|---|---|---|---|---|---|---|---|
| kfm12 | 0.661 | 0.677 | 0.671 | 0.703 | 0.678 (Fed:Cent 101.0 %) | 0.018 | 0.018 |
| WBC | 0.442 | 0.466 | 0.404 | 0.543 | 0.464 (Fed:Cent 92.8 %) | 0.059 | 0.029 |
| ChestX | 0.235 | 0.192 | 0.176 | 0.222 | 0.206 (Fed:Cent 95.3 %) | 0.027 | 0.007 |

**Is it "the last node wins"?** Per class, AP50 when the class's owner trains
last, minus its mean over the other three orders:

- ChestX: **+0.047, 10/10 classes better when their node is last** — a
  consistent recency effect. The primary order (node D last, owning 4 of the 10
  classes) was the most favourable one.
- kfm12: −0.015, 6/12 classes — no recency effect.
- WBC: −0.055, 1/7 classes — being last does not help; the order still moves
  the result a lot, through which nodes follow which, not through recency.

**Considerations**

1. Node order is a real source of variation in async, as large as the seed
   (kfm12) or larger: 2× (WBC) and 3.6× (ChestX). The primary study keeps the
   order fixed in every seed, so its async confidence intervals do not include
   this variation; its async results hold for the A→B→C→D order.
2. ChestX async passing the 90 % test (115.6 % Fed:Cent over seeds 0–2) is
   partly an effect of that order: on seed 0, averaged over the four orders,
   Fed:Cent is 95.3 % instead of 108.6 %. It likely still clears 90 % on
   average, but with less margin than reported.
3. In a real network nodes arrive in varying order, so the order-averaged
   values are the realistic expectation: on seed 0, kfm12 101 %, WBC 93 %,
   ChestX 95 % of centralized.
4. For the new confirmatory study: counterbalance the order (seed k uses
   rotation k mod 4, seed count a multiple of 4), so the reported mean and
   interval include the order effect at no extra cost.
5. For the method: async aggregation should be less order-sensitive. The
   queued `swa_async` (average of one full cycle) targets this directly;
   FedBuff/CA2FL already reduce it by merging several nodes per step.

Limitations: one seed, so order and seed are not fully separated; four orders
are the cyclic rotations only (not all 24 permutations).

## 7. Order robustness (runs `shufswa_r0..r3`)

**Goal:** an async result that does not depend on the order in which nodes
arrive.

**Change** (opt-in: `federation.async_order: shuffle` with `swa_last: 4`; the
three constraints of §2 still hold):

1. Every cycle visits each node once, in a new random order drawn from the run
   seed (reproducible). Staleness stays 0; this requires
   `max_concurrent_nodes: 1`.
2. The final model is the average of the last 4 global versions, one per node,
   so no single node's update decides the final model.

**Sources:**
- Random reshuffling beats a fixed cyclic order in sequential FL on
  heterogeneous data (Li & Lyu, NeurIPS 2023).
- Cyclic weight transfer across hospitals: a varied order reduces
  run-to-run variance (Chang et al., JAMIA 2018; Balachandar et al., JAMIA
  2020).
- Averaging late weights gives flatter solutions (FedSWA, ICML 2025).

**Evaluation:** seed 0, four runs whose node list starts at A, B, C, D. With
shuffling, each starting list gives a different sequence of orders (below),
standing in for four different arrival patterns.

| Run | Node order in each of the 10 cycles |
|---|---|
| r0 | DBAC DBCA CDAB CADB BDAC CBAD DBAC DCBA ADBC DBAC |
| r1 | ACBD ACDB DABC DBAC CABD DCBA ACBD ADCB BACD ACBD |
| r2 | BDCA BDAC ABCD ACBD DBCA ADCB BDCA BADC CBDA BDCA |
| r3 | CADB CABD BCDA BDCA ACDB BADC CADB CBAD DCAB CADB |

Success means both of the following:
- the sd over the four runs is clearly below the fixed-order sd of §6
  (kfm12 0.018, WBC 0.059, ChestX 0.027);
- the mean is not below the fixed-order mean over orders (0.678 / 0.464 /
  0.206).

If both hold, seeds 1–2 follow, and this becomes the candidate async scheme
for the new confirmatory study.

## 8. Toward ≥ 90 % of centralized on every dataset, sync and async, independent of order

**Date:** 2026-10-01.

### 8.1 Where the gap actually is (seeds 0–2)

1. **The centralized baseline is sequential.** Each epoch makes one full pass
   over node A's data, then B, C, D, with one optimizer (`cofy/centralized.py`).
   Async with one node at a time is the federated version of the same schedule;
   sync (parallel training, then averaging) is not.
2. **kfm12 fails the test because of the baseline's variance, not because of
   federation.** Federated means are 98–108 % of centralized, but the centralized
   mAP50 varies across seeds 4–5× more than the federated runs (sd 0.069 vs
   0.014–0.018). With n = 3 (t = 4.30) no method can pass. Same means and sd
   projected to more seeds:

   | Method | Dataset | Mean Fed:Cent | CI-low n=3 | n=5 | n=10 | n=12 |
   |---|---|---|---|---|---|---|
   | primary sync | kfm12 | 104.3 % | 71.5 % | 87.9 % | 94.9 % | 95.9 % |
   | primary sync | ChestX | 78.5 % | 73.8 % | 76.2 % | 77.2 % | 77.3 % |
   | exclusion sync | ChestX | 87.2 % | 66.3 % | 76.7 % | 81.2 % | 81.9 % |
   | primary async | WBC | 102.1 % | 60.1 % | 81.1 % | 90.0 % | 91.4 % |
   | fedasync | kfm12 / WBC / ChestX | 106.9 / 110.2 / 110.1 % | 67.4 / 70.9 / 101.9 % | 87.2 / 90.5 / 106.0 % | 95.5 / 98.9 / 107.7 % | 96.8 / 100.2 / 108.0 % |
   | fedbuff | kfm12 / WBC / ChestX | 108.1 / 119.9 / 93.5 % | 80.6 / 90.9 / 81.6 % | 94.4 / 105.4 / 87.6 % | 100.2 / 111.6 / 90.1 % | 101.1 / 112.5 / 90.5 % |

   A seed count of about 10–12 (planned for the new study) is therefore needed
   in any case. Methods also have to pass with their *order-averaged* async
   result, not the fixed-order one (§6).
3. **ChestX sync is truly below 90 %:** 76–87 % for every sync variant so far.
   Each node's class rows are trained against its own local copy of the shared
   layers, then used with the average of four copies.
4. **Async depends on node order** (§6). The node that trains last also
   provides the final shared layers, which favours its classes.

### 8.2 New candidates

| Variant | Mode | Change | Targets | Sources |
|---|---|---|---|---|
| final calibration (`final_calibration`; post hoc: `<run>+cal`) | sync and async | After the last aggregation each node re-estimates the BatchNorm statistics of the final model (forward passes, pooled by the server), then retrains only its own class rows on the frozen final model (1 local epoch). The server writes the rows in; nothing else changes | 8.1.3 (rows matched to the final shared layers) and 8.1.4 (the last node's advantage disappears: every row is matched to the same final features) | classifier re-training on frozen features (Kang et al., ICLR 2020; CCVR, NeurIPS 2021, here done by the nodes instead of the server); local classifier fine-tuning after aggregation (layer-peeled analysis, arXiv 2502.03231, 2025); BatchNorm re-estimation after weight averaging (SWA, UAI 2018) |
| `chain` (`chain_exchange`) | sync | 4 models circulate: each round every node trains a different one, so in a 4-round cycle each model is trained once by every node, in a different order (cyclic Latin square on a new seeded order each cycle). Averaged at cycle end; every node is last in exactly one averaged model. 20 rounds × 1 epoch, same budget | 8.1.1 and 8.1.3: each model learns sequentially like the centralized baseline, while training stays parallel with a round barrier; order effects average out by design | FedCross (multi-model cross-aggregation, ICDE 2024); FedExchange (model exchange for federated object detection, 2025); cyclic weight transfer (Chang et al., JAMIA 2018); sequential FL converges better than parallel FL on heterogeneous data (Li & Lyu, NeurIPS 2023) |
| `sync20` | sync | Plain FedAvg, 20 rounds × 1 epoch (same budget) | Control: separates the effect of shorter local training from the exchange | local-drift bounds of FedAvg (Li et al., ICLR 2020) |

Cost of the calibration round:
- **Compute:** about one forward epoch plus one class-rows-only epoch per node, roughly 5 % of training.
- **Communication:** one extra round; nodes upload only BatchNorm statistics and their own class rows.
- **Constraints:** all three of §2 still hold. The server averages statistics and writes rows; it holds no data and trains nothing.

### 8.3 Screening and decisions

1. Calibration post hoc on finished runs (no retraining): primary sync and
   async, seeds 0–2; the four node orders (`order_r1..r3` + primary); the four
   shuffled orders (`shufswa_r0..r3`); `chain` and `sync20`. `before` is
   re-evaluated, and must equal the run's own number.
2. Sensitivity: the same calibration applied to the centralized baseline,
   shown as `[centralized+cal]` and not used as the reference. If it improves
   centralized as much as the federated runs, the gain is not specific to
   federation, and that will be reported.
3. Training screen on seed 0: `chain` and `sync20`.
4. A candidate moves to seeds 1–2 (and, in async, to the four-order check) if,
   on seed 0, it reaches ≥ 95 % Fed:Cent on all three datasets.
5. The final choice is confirmed in the new study (10–12 seeds,
   counterbalanced orders), never on the seeds used for selection.

### 8.4 Search rules and training budget (decided 2026-10-01)

**Automatic rules** (`tools/search_rules.py`, run by the queue after every entry):

1. **Winner.** The first sync and the first async method to pass the
   pre-registered test (CI-low of Fed:Cent ≥ 90 %) on all three datasets, with
   the same complete n, is the winner of its mode. It is written to
   `runs/variants/_WINNERS.txt` and the queue skips the remaining entries of that
   mode. An async method qualifies only with a varying node order
   (`async_order: shuffle`, a different random order for every seed), because
   §6 showed the order matters. Fixed-order results such as FedAsync's do not
   qualify; `fedasync_shuf` is FedAsync with a random order.
2. **Promotion.** A seed-0 screen with Fed:Cent ≥ 95 % on all three datasets
   gets seeds 1–2.
3. **Extra seeds.** After seeds 0–2, a method with mean ≥ 90 % everywhere but
   CI-low < 90 % somewhere gets the seeds its projected CI needs (same mean and
   sd; at most 12; same count on every dataset). This is decided once, from the
   first three seeds, and the test is then done once at that n, so seeds are
   never added until the test happens to pass. Centralized baselines for seeds
   the primary study never ran are trained first (`tools/run_baseline.py`). The
   reason is shown in the results table's note column.

**Training budget.** At 20 epochs, ChestX models are far from converged (oracle
0.27 mAP50), so Fed:Cent there mostly reflects learning speed.

- **Pilot:** `tools/budget_pilot.py` trains the centralized baseline (seed 0)
  for up to 100 epochs, evaluated every 5. It stops after 30 epochs without a
  ≥ 2 % gain.
- **Budget:** the first evaluated epoch within 2 % of the best mAP50, one per
  dataset. The learning rate is constant after warmup, so the curve's first E
  epochs are exactly a run with budget E.
- **Evaluation set:** the curve is measured on the evaluation set, because no
  separate validation split exists. Stopping near the centralized model's best
  point favours centralized, so this makes the test stricter on federation, not
  easier.
- **Use:** the 20-epoch screening and its rules continue as a cheap filter.
  Winners are confirmed at the new budget: same seeds rule, with centralized
  retrained at that budget.

### 8.5 First results (2026-10-01, evening)

**Random order + averaging of the last cycle (§7), seed 0, four starting lists**

| Dataset | Fixed order: mean / sd over 4 orders | Shuffled + averaged: mean / sd over 4 runs |
|---|---|---|
| kfm12 | 0.678 / 0.018 | 0.661 / 0.035 |
| WBC | 0.464 / 0.059 | 0.521 / 0.070 |
| ChestX | 0.206 / 0.027 | 0.188 / 0.018 |

Neither success criterion of §7 is met: the spread did not drop, and the mean is
lower on kfm12 and ChestX. Not retained.

**Final calibration round applied post hoc (seeds 0–2).** Re-evaluating
the saved models reproduces each run's reported number exactly.

| Mode | Dataset | Before → after (mean mAP50) | Fed:Cent before → after | CI-low after |
|---|---|---|---|---|
| sync | kfm12 | 0.677 → 0.696 | 104.3 → 107.2 % | 71.0 % |
| sync | WBC | 0.626 → 0.634 | 137.1 → 139.4 % | 88.1 % |
| sync | ChestX | 0.159 → 0.182 | 78.5 → 90.5 % | 68.7 % |
| async (fixed order) | kfm12 | 0.642 → 0.711 | 98.8 → 109.3 % | 85.4 % |
| async (fixed order) | WBC | 0.463 → 0.464 | 102.1 → 101.1 % | 91.3 % |
| async (fixed order) | ChestX | 0.234 → 0.257 | 115.6 → 127.7 % | 94.6 % |

- **ChestX sync** improved on every seed (+0.012 to +0.037) and its mean reached 90 %; the CI is too wide to pass.
- **WBC** is mixed (−0.07 to +0.08 per seed). Its classes appear in separate images (no unannotated other-class boxes), so a class's rows never see another class's cells as negatives.
- **Both steps contribute.** BatchNorm re-estimation alone and row retraining alone each help in most cases; together they help most.

**Training-budget pilot (centralized, seed 0, mAP50 every 5 epochs)**

| Dataset | at 20 epochs | best (epoch) | 3-point moving average: peak (epoch) | Budget (smoothed rule) |
|---|---|---|---|---|
| kfm12 | 0.619 | 0.707 (35) | 0.693 (30) | 30 |
| WBC | 0.460 | 0.494 (5) | flat at 0.46–0.47 from epoch 5 | 20 (more brings nothing) |
| ChestX | 0.235 | 0.294 (55) | 0.288 (55) | 55 |

**Key observation: one run's score swings between checkpoints.**

- **Size of the swings.** The centralized mAP50 moves between checkpoints only 5 epochs apart, sd ≈ 0.02 on the plateau (e.g. ChestX 0.253 → 0.176 → 0.271 at epochs 25/30/35). That is the size of the seed spread and of the node-order spread.
- **Cause.** The learning rate stays constant after warmup and the final weights are not averaged, so where training stops decides the score.
- **Consequence.** Much of the variance that makes CIs wide, and much of the "order effect" measured with one run per order, is this endpoint noise.
- **Remedy (for every arm, centralized included).** Decay the learning rate to about 1 % by the end of the budget, as standard YOLO training does.

### 8.6 Protocol v2 and the overnight decisions (2026-10-02)

**Protocol v2 (user decision).**
- **Learning rate:** every arm, centralized included, decays linearly from lr0 to 1 % over its whole budget (`federation.lr_schedule: linear`, `lrf: 0.01`). The decay is driven by global progress, so it runs across rounds and cycles.
- **Budgets:** plateau budgets from §8.5: kfm12 30, WBC 20, ChestX 56 epochs (2 local epochs per round or cycle).
- **Reference:** `v2_central`, the centralized baseline trained with the same recipe, for each seed.
- **Candidates:**
  - `v2_async`: random node order every cycle, average of the last 4 versions, final calibration post hoc (`v2_async+cal`);
  - `v2_sync`: plain, and with calibration (`v2_sync+cal`).
- **Scope:** promotions and extra seeds now apply only to v2 rows. A winner can still come from a test already scheduled under the primary protocol.

**Decisions taken by the queue overnight (primary protocol, 20 epochs).**
- `shufswa_r0+cal` (random order + average of the last 4 versions + final calibration) reached 110.4 / 112.0 / 108.8 % on seed 0 and was promoted to seeds 1–2.
- At n = 3: mean Fed:Cent 114.0 / 124.3 / 125.5 %, CI-low 85.2 / 88.0 / 82.8 %.
- Its federated mAP50 is very stable across seeds (kfm12 0.741 / 0.735 / 0.747; WBC 0.560 / 0.575 / 0.562; ChestX 0.235 / 0.248 / 0.274). The CI is wide because the centralized reference varies.
- The projection gave n = 4, so seed 3 was added once, with centralized baselines for WBC and ChestX (kfm12 seed 3 already existed in the primary study). The test is done once at n = 4.

### 8.7 Async winner (2026-10-02 08:59)

**`shufswa_r0+cal`** passes the pre-registered test on all three datasets
(primary protocol: 20 epochs, constant learning rate; n = 4, seeds 0–3).

**Method**
- **Order and averaging:** asynchronous, one node at a time, in a new seeded random node order every cycle. The final model is the average of the last 4 global versions (one per node).
- **Final calibration round:** each node re-estimates BatchNorm statistics and retrains only its own class rows on the frozen final model; the server pools the statistics and writes the rows.

| | kfm12 | WBC | ChestX |
|---|---|---|---|
| Fed:Cent mean (CI-low) | 112.6 % (97.0 %) | 125.7 % (106.2 %) | 127.3 % (104.2 %) |
| Fed:Oracle | 92.8 % | 58.4 % | 94.3 % |
| Without calibration, Fed:Cent mean | 101.8 % | 112.6 % | ~90 % |

- **Order:** calibration removes most of the ChestX node-order effect. On the four fixed-order rotations of seed 0, the sd falls from 0.027 to 0.010 and the mean rises from 0.206 to 0.247. kfm12 (sd 0.018 → 0.017) and WBC (0.058 → 0.065) keep their spread, but their means rise by +0.045 and +0.051.

**Caveats**
1. The method was chosen among about 20 variants using seeds 0–2, and those seeds are part of the test, so the margin is optimistic.
2. The number of seeds (4) was fixed after seeing the first three.

Both are why §8.3 requires confirmation on seeds not used for selection. As a
sensitivity check, the same calibration is also applied to the centralized
baseline (`calib_centralized`, seeds 0–3).

### 8.8 Confirmation on fresh seeds 4–7 and the calibration sensitivity check (2026-10-03)

**Confirmation of `shufswa_r0+cal`** (seeds 4–7, never used for selection; tested once at n = 4):

| | kfm12 | WBC | ChestX |
|---|---|---|---|
| Fed:Cent per seed | 118 / 115 / 108 / 108 % | 151 / 119 / 105 / 98 % | 131 / 124 / 126 / 105 % |
| Mean (CI-low) | 112.3 % (104.3 %) | 118.0 % (80.7 %) | 121.5 % (103.7 %) |
| Result | pass | **fail** | pass |

**Outcome:** by the rule fixed in advance the confirmation **fails on WBC**.
Every WBC seed is at least 98 % of centralized, but the spread is wide:
- the seed-4 centralized baseline is unusually low (0.375, against 0.41–0.50 for seeds 0–3);
- the federated WBC results drift down from 0.566 to 0.467 across seeds 4–7.

**Calibration applied to the centralized baseline too** (sensitivity, seeds 0–4; seeds 5–7 pending):

| | kfm12 | WBC | ChestX |
|---|---|---|---|
| Centralized → centralized + calibration (mean mAP50) | 0.665 → 0.718 | 0.443 → 0.459 | 0.203 → 0.252 |
| Winner vs plain centralized: mean (CI-low) | 113.7 % (102.7 %) | 130.7 % (111.5 %) | 128.0 % (112.3 %) |
| Winner vs calibrated centralized: mean (CI-low) | 104.8 % (99.3 %) | 125.9 % (108.8 %) | 103.2 % (93.1 %) |

**The calibration is not specific to federation.** It improves the centralized
baseline by a similar amount on ChestX and kfm12, because it repairs the
end-of-training state of the constant-learning-rate recipe (BatchNorm
statistics and head). Most of the winner's margin over centralized therefore
comes from post-processing that centralized did not get. Against a centralized
baseline given the same calibration, the winner is still at or above 90 %
(CI-low) on seeds 0–4.

**Protocol v2 centralized** (learning-rate decay, plateau budgets, seeds 0–2):

| | kfm12 | WBC | ChestX |
|---|---|---|---|
| mAP50 | 0.698 / 0.739 / 0.750 | 0.582 / 0.546 / 0.520 | 0.314 / 0.308 / 0.304 |
| sd across seeds (v2) | 0.028 | 0.031 | 0.005 |
| sd across seeds (20-epoch) | 0.069 | 0.046 | 0.013 |

v2 centralized is much stronger and much less variable. This is the
reference on which the method has to hold.

**Bug fixed.** The queue read "ASYNC WINNER" as also declaring a sync winner (a substring match), so the sync search would have been skipped. Fixed with a word-boundary match; no sync run was lost, since none had started yet.

### 8.9 Decisions after the failed confirmation (2026-10-03)

1. **Withdrawn.** `shufswa_r0+cal` is withdrawn as async winner. The record is kept in `runs/variants/_WINNERS_withdrawn.txt`, and `runs/variants/_withdrawn.txt` makes it ineligible.
2. **Same post-processing on both sides.** A calibrated federated row (`+cal`) is compared with the centralized baseline given the same calibration (`[centralized+cal]` reference rows). Under this rule `shufswa_r0+cal` would also have failed on ChestX at seeds 0–3 (mean 101.2 %, CI-low 88.8 %).
3. **Async search continues under protocol v2** with the same design (`v2_async`: random order every cycle + average of the last 4 versions; `v2_async+cal` with calibration), seeds 0–2, against `v2_central` and its calibrated version `calib_v2_central`. The v2 centralized seed spread is 1.5–2.6× smaller, which is what failed on WBC. The automatic rules (promotion, extra seeds, winner) apply. Then `v2_sync`.

### 8.10 Status 2026-10-04 10:45 and backlog dropped

**Protocol v2 results so far**

| | kfm12 | WBC | ChestX |
|---|---|---|---|
| Centralized v2 (seeds 0–2) | 0.729 | 0.549 | 0.309 |
| Centralized v2 + calibration | 0.750 | 0.531 | 0.309 |
| `v2_async`, no calibration (seeds done) | 0.739, n = 3: 101.6 %, CI-low 87.9 %, projected pass at n = 4 | 0.586, n = 2: 104.3 % | 0.301, n = 2: 96.6 % |

With learning-rate decay, calibration no longer improves centralized. It mainly
repaired the constant-learning-rate end state.

**Backlog dropped (user decision).** The remaining 20-epoch, constant-LR entries
(~70 GPU h) are dropped; finished runs stay in the table for reference, and the
plan before the change is in `runs/variants/_plan_before_drop_2026-10-04.txt`.
The queue now runs only the v2 search: `v2_async` (+ calibration, + seed 3 if
its CI needs it), then `v2_sync` (+ calibration). If `v2_sync` fails, the most
promising sync ideas (rotating models, exclusion loss) will be re-run under v2
as a targeted follow-up.

### 8.11 Validation for a concurrent deployment (user decision 2026-10-04)

**Why.** A deployed server cannot shuffle arrivals: the order follows when nodes
finish. With nodes on separate machines, three things the earlier async tests
never had appear:
1. the order comes from timing;
2. staleness is above 0;
3. faster nodes (less data or a faster machine) upload more often.

The shuffled-turn runs (`v2_async`) remain as the result for a *sequential*
deployment, where the server hands out turns. They are reference only and
cannot win.

**Virtual clock** (`federation.async_clock: virtual`, `cofy/server.py`):
- **Timing model.** Each node gets a machine speed, drawn once per seed (× U[0.7, 1.3]). Each submission lasts images × local epochs / speed × U[0.8, 1.2].
- **Pulls and pushes.** All nodes pull at time 0; a node pulls again right after its update is applied. Updates are applied in finish order, with staleness weighting (w = 1 / (1 + 0.5 s)).
- **Execution.** The timeline depends only on the seed and the data sizes, so the schedule is computed first (`virtual_schedule.json`). Each training then runs on the GPU when its pull happens; it depends only on the version pulled.
- **Final model.** The average of the global versions right after each node's last update (`swa_per_node`), one per node whatever its speed.
- **Two stop rules:**
  - `cycles`: every node submits the same number of times, so fast nodes finish early;
  - `deadline`: all nodes train until the same total compute is spent, so fast nodes submit more often. A submission starts if at least half of it fits before the deadline; compute is within 1–6 % of `cycles`.
- **CPU test.** Order and staleness match the timeline; for example, the slowest node applies its last two updates alone at the end.

**Runs:**
- `v2_vclock` (cycles stop) on seeds 0–2, then calibration post hoc;
- `v2_vclock_dl` (deadline stop) as a seed-0 screen, promoted by the rules if ≥ 95 % everywhere.

Both run before `v2_sync`. Only virtual-clock async rows can be the async
winner.

**Still to port to the network system:** per-node final averaging, the
learning-rate schedule driven by the server's global progress, and the
calibration round.

### 8.12 Asynchronous calibration (user rule 2026-10-04)

**User rule.** The calibration round as a closing barrier is not acceptable for
async. It is acceptable if nodes always have access to the current model and
calibration is applied progressively, as nodes contribute, becoming complete
when all nodes have been seen, without anyone waiting.

**Design** (`cofy/calibrate.py` `calibrate_progressive`; rows `<run>+acal`):
1. When training ends, the final model M is published uncalibrated, so it is usable at once.
2. Each node contributes when it is available. It sends its BatchNorm statistics of M, which the server merges at once. It then re-fits its own class rows on M with the statistics merged so far and uploads them.
3. After every contribution the server releases a model:
   - BatchNorm is pooled over the contributors, with M's stored statistics standing in for the nodes not heard from yet, weighted by their batch counts;
   - the contributors' rows are re-fitted, and the other rows are M's.
4. After the last contribution this is the complete calibration. Nobody waits, and the server only averages statistics and copies rows.

**Evaluation.**
- **Order.** In the virtual-clock runs, contributions arrive in the order they finish: data size / machine speed × jitter, from the run's own timeline.
- **Partial states.** Every released state is evaluated (`acal_1`…`acal_3`, then `acal`) to show partial calibration is safe.
- **CPU test.** Only the contributors' rows change, and BatchNorm converges to the full pool.

**Eligibility.**
- Async winners can be plain virtual-clock rows or `+acal` rows.
- `+cal` (the closing synchronous round) is reference only for async.
- Sync rows may use `+cal`: one more round is within sync rules.

**Bug fixed (2026-10-05).** The async delta rule also applied stale deltas to
BatchNorm running variances. In `v2_vclock_dl` WBC seed 0 one layer ended at
−0.245, and the model scored mAP50 0.000 (the calibrated rows, which re-estimate
BatchNorm, scored 0.619). Variances are now clamped at 0. Runs with staleness 0
(all one-at-a-time runs) could not be affected. Every other virtual-clock final
model was checked: none had a non-positive variance. The seed-0 screen decision
for `v2_vclock_dl` does not change: ChestX is below 95 % either way (83.0 % plain,
84.8 % +acal).

## 9. Fixes for diluted updates (2026-10-06)

### 9.1 Diagnosis

Under protocol v2 (seeds 0–2):

| Fed:Cent | kfm12 | WBC | ChestX |
|---|---|---|---|
| async, one node at a time (`v2_async`, reference) | 101.6 % | 106.4 % | 99.1 % |
| sync (`v2_sync`) | 94.6 % | 111.9 % | 77.7 % |
| async, concurrent nodes (`v2_vclock`) | 88.3 % | 108.9 % | 76.8 % |

**The two failing schemes share one property: each node's update is applied only in part.**
- **Sync:** the server averages 4 nodes' updates. On disjoint classes they point in different directions, so each node's progress on the shared layers is cut by up to 4×.
- **Concurrent async:** each update is damped by staleness, w = 1 / (1 + 0.5 s). Measured staleness is 2.4–2.7 on average (maximum 8), so the mean weight is 0.46–0.53: each update lands at about half strength.
- **One node at a time:** w = 1, every update applied in full, and it matches centralized.

**This hurts only datasets that need long training,** because diluted updates act like a shorter budget:
- ChestX keeps improving until about epoch 55 (centralized 0.235 at 20 epochs, 0.29 at 55). It loses 22 %, and its sync result (0.240) equals centralized at about 20 of its 56 epochs.
- kfm12 plateaus around 30 epochs and loses 5–12 %.
- WBC plateaus by epoch 5 and loses nothing (> 100 %).

### 9.2 Alternatives (every one explained, including those not run)

**Concurrent async**

| Variant | What it does | Why it should help | Risk | Status |
|---|---|---|---|---|
| `v2_vclock_a0` | Stale updates applied at full strength (`staleness_alpha: 0`, w = 1); everything else as `v2_vclock` (virtual clock, per-node final average) | Removes the halving of every update: each update counts as in the one-at-a-time scheme | A very stale update (up to 8 versions old) is added on top of a newer model; the deltas mostly concern different classes, so conflicts should be small | queued |
| `v2_vclock_a02` | Mild damping, `staleness_alpha: 0.2` (w = 0.63 at staleness 3, 0.38 at 8) | Middle ground if full strength is unstable: typical updates nearly full, very stale ones still damped | Gains less than a0 if dilution is the whole story | queued (skipped if a0 wins) |
| buffered sum (FedBuff with server step = buffer size) | Merge each group of K arrivals with their *sum* instead of their mean | Order-free inside a group, full strength | Same strength as a0 with more machinery. Earlier FedBuff/CA2FL runs used the *mean* and collapsed on ChestX (CA2FL 42 %) | not run: a0 tests the same principle more simply |
| polynomial staleness (1 + s)^−0.5 | FedAsync's other damping | — | Still about 0.5 at s = 3, so the same dilution | not run |

**Sync**

| Variant | What it does | Why it should help | Risk | Status |
|---|---|---|---|---|
| `v2_chain` | Rotating models: 4 models circulate, each round every node trains a different one, so in 4 rounds each model is trained by every node in turn (new random order each cycle). Averaged every 4 rounds. 1 local epoch per round; rounds 28 / 20 / 56 (kfm12 28 instead of 30: rounds must be a multiple of 4, rounded down so federated never gets more compute) | Inside a cycle each model learns sequentially, with every update in full, like the scheme that matches centralized; still a round barrier with parallel training. Models are averaged only after all of them have seen every node, so they no longer point in disjoint directions | Each model sees a node's data only once per cycle | queued |
| `v2_sync_slr2` | FedOpt server step: shared weights move 2× the averaged update (`server_lr: 2`). Class rows and BatchNorm statistics unchanged | Restores part of the dilution (the measured shortfall corresponds to about 2.8×) | Directions all nodes agree on (generic features) also move 2× | queued |
| `v2_sync_slr4` | Same with `server_lr: 4` (= number of nodes: the sum of the updates) | Full restoration if the updates are orthogonal | Unstable if they are not | queued |
| fewer local epochs (FedAvg, 1 epoch × 2× rounds) | Less drift per round | Addresses drift, not dilution: the shortfall does not shrink with fewer local steps | — | not run |
| SCAFFOLD, FedProx, FedNova, Fisher/TIES merging | Drift correction / re-weighting | Already run (primary protocol): none raised ChestX sync above 87 % | — | done earlier (§3–§4) |

**Calibration.** Each run is also calibrated post hoc: `+cal` (closing round)
for sync, `+acal` (asynchronous progressive calibration) for async.

**Fixed meanwhile.** The BatchNorm-variance bug of the async delta rule is
fixed, and `v2_vclock_dl` WBC seed 0 is re-run (the broken result is kept as
`*_bnvar_bug.json`).

**Procedure.**
- Seed-0 screens on all three datasets, then the automatic rules: ≥ 95 % everywhere → seeds 1–2; CI-based extra seeds up to 12; the first method passing everywhere wins its mode.
- Async winners: virtual clock only, plain or `+acal`.

### 9.3 Results (2026-10-07): not solved

ChestX Fed:Cent, seed 0:

| Variant | Plain | Calibrated |
|---|---|---|
| `v2_vclock_a0` (async, full strength) | 0 % (BatchNorm running mean drifted to 2.5·10⁵) | 75.7 % (+acal) |
| `v2_vclock_a02` (async, mild damping) | 77.6 % | 77.3 % (+acal) |
| `v2_chain` (sync, rotating models) | 74.9 % | 78.0 % |
| `v2_sync_slr2` (sync, server step 2×) | 77.5 % | 84.9 % |
| `v2_sync_slr4` (sync, server step 4×) | 44.2 % | 80.5 % |

All screens stopped at seed 0.

- **The dilution hypothesis is refuted:** stronger updates do not raise ChestX, and the strongest break training.
- **Revised cause:** merging models that diverged in parallel on disjoint classes (§5.7 of `SEARCH_REPORT.md`).
- **`v2_vclock_dl` WBC seed 0, re-run after the BatchNorm fix:** 0.579 (99.5 %).

Every alternative, with explanations and verdicts, and the next-step proposal
are in `analysis/SEARCH_REPORT.md`.

## 10. Drift-limiting screens (user decision 2026-10-07)

**Cause addressed.** Merging models that diverged in parallel on disjoint
classes (`SEARCH_REPORT.md` §5.7). Screens run on ChestX, seed 0, first
(`sync@chestx` lines). The rules add kfm12 and WBC only if ChestX reaches
≥ 95 %, then seeds 1–2 if ≥ 95 % everywhere.

| Variant | Mode | What it changes vs the v2 baseline | Why |
|---|---|---|---|
| `v2_sync_q` | sync | ¼ epoch per round, 4× rounds (kfm12 120, WBC 80, ChestX 224); same compute | With few local steps FedAvg approaches mini-batch SGD over all nodes; divergence grows with local steps |
| `v2_vclock_q` | async, concurrent | ¼-epoch uploads (4× more); BatchNorm running statistics interpolated towards the node's values instead of delta-accumulated | Small stale deltas behave like asynchronous SGD with a short delay; interpolation avoids the running-mean drift seen in `a0` |
| `v2_sync_excl` | sync | Exclusion loss | Best ChestX sync addition at 20 epochs (78.5 → 87.2 %) |
| `v2_vclock_excl` | async, concurrent | Exclusion loss + interpolated BatchNorm statistics | Same, for concurrent async |
| `v2_sync_rs` | sync | Control: only the per-round reshuffle | Measures the effect of the fixed data order alone (below) |

**Every new screen reshuffles the data per round** (`reshuffle_rounds`).
Ultralytics seeds every newly built data loader with the same fixed generator
seed, and the federated code builds one per round, so every round started
from the same image order. Centralized builds its loaders once and reshuffles
every epoch. With full epochs this only removes some randomness from
federated training; with ¼-epoch rounds it would train the same quarter every
time. The new runs differ from the v2 baselines by this as well, hence the
control `v2_sync_rs`.

**Async calibration.** Async screens and the finished `v2_vclock` (seeds 0–2)
also get the progressive calibration with refresh (`+acalr`). After all
contributions, each node re-fits its rows once more against the complete
BatchNorm pool. This targets the gap between `+acal` and the closing round
seen in `a02` (0.244 vs 0.276 on ChestX).

### 10.1 Results (2026-10-08): not solved

ChestX, seed 0; same-seed baselines `v2_sync` 0.244, `v2_vclock` 0.261
(+acalr 0.289). Every screen stopped at the ChestX stage (< 95 %).

| Variant | Plain | Best calibrated | vs same-seed baseline |
|---|---|---|---|
| `v2_sync_q` (¼ epoch per round) | 0.235 (75.0 %) | 0.237 | −0.009 |
| `v2_sync_excl` (exclusion) | 0.243 (77.6 %) | 0.246 | ±0 |
| `v2_sync_rs` (reshuffle only, control) | 0.226 (72.0 %) | 0.243 | −0.018 |
| `v2_vclock_q` (¼-epoch uploads, BatchNorm interpolated) | 0.256 (81.6 %) | 0.260 (+acalr) | −0.005; calibrated −0.029 |
| `v2_vclock_excl` (exclusion, BatchNorm interpolated) | 0.238 (75.9 %) | 0.272 (+acalr) | −0.023 / −0.017 |
| refresh on `v2_vclock`, seeds 0–2 | — | +acalr 89.2 % (CI-low 84.1) | +acal 88.3 % |

**Reading**
1. **Less local work between merges does not help.** Neither sync with ¼ epoch per round nor concurrent async with ¼-epoch uploads improves ChestX, so the size of the divergence before a merge is not the cause either.
2. **The fixed data order was not hurting.** The reshuffle control is not better (−0.018 on one seed, within the run-to-run noise).
3. **Exclusion does not carry over.** Its gain at 20 epochs does not appear under protocol v2.
4. **Refresh helps slightly.** It closes part of the gap between `+acal` and the closing round, but not enough.

**What remains consistent.** On ChestX, every scheme in which nodes train in
parallel lands at 0.23–0.29 (72–90 %); applying the nodes' work one after the
other lands at 0.30–0.32 (97–101 %). Not tested (declined by the user): ¼-epoch
rounds combined with a 4× server step, the combination under which averaging
gives each node's classes the same learning signal per image as sequential
training.

## 11. Main threshold 85 % (user decision 2026-10-08)

The main criterion is now CI-low ≥ 85 % (post hoc; deviation recorded in
`PREREGISTRATION.md`; the table keeps the 90 % verdict in its `@90` column;
the screening bar is now 90 %). The extra-seed decisions taken under 90 % are
archived in `_search_log.json` (keys `…|stage2@90`), so the rules re-decide them
under 85 % at the next queue launch. That launch would add seed 3 for
`v2_vclock` (+ its calibrations) together with its centralized reference
`v2_central` seed 3 and that reference's calibration. Situation and reasoning
are in `SEARCH_REPORT.md` §9.

## 12. Follow-up checks for the paper: results (runs 2026-11-07/08, analysed 2026-11-09)

Exploratory, added after the confirmatory test. Ratios are to the calibrated v2 centralized
reference of the same seed.

**Averaged centralized reference (`central_swa`, WBC, seeds 100-109).** Mean of 4 snapshots over
the last 2 epochs, then bn+head calibration. mAP50 0.502 -> 0.516 (102.9 % of plain centralized,
95 % CI 99.2-106.6). Against it: sequential 102.5 % (96.2-108.9), so averaging explains its excess;
concurrent 127.1 % (120.0-134.3), not explained (already 126.0 % before calibration). Before
calibration the sequential scheme is at 98.3 %; calibration lowers centralized WBC (0.523 -> 0.502).

**Local-only baseline (`local_only`, seeds 100-102).** Each node trains alone; each class scored
on its owner's model (an exchange of models after training, 4x inference).

| use case | local only | sequential | concurrent | worst class (local / seq / conc) | nodes local / seq / conc |
|---|---|---|---|---|---|
| kfm12 | 89.5 (79.9-99.0) | 98.8 | 93.1 | 34 / 73 / 49 | 75-94 / 91-104 / 77-106 |
| wbc | 95.9 (76.0-115.9) | 103.1 | 136.4 | 81 / 81 / 108 | 81-107 / 99-127 / 140-153 |
| dentex | 99.9 (98.5-101.2) | 100.4 | 99.9 | 96 / 98 / 95 | 99-101 / 100-101 / 99-100 |

kfm12 node B (fasciola, dicrocoelium, metastrongili, trichuris) keeps 89 % alone and 77 % in the
concurrent federation: the only node worse off federating. Under the sequential scheme every node gains.

**Per-class detection budget.** Defended sign flip, budget scoring: 82.5 / 91.1 / 96.2 % (seq /
conc / par, seeds 0-2), attacker's classes 0, other classes 0.54-0.69. Concurrent BN poisoning
seed 2: 0.1 % with and without the budget. Clean defended runs: identical scores (9/9).

**Backdoor (`tools/robust_eval.py --report`, `analysis/ROBUSTNESS.md`).** Attack success rate 100 %
in all 18 backdoor runs (3 schemes x with/without defence x 3 seeds; 2,016-2,492 triggered images
per seed); clean runs 0-1.2 %. Attacker uploads scaled: 0 % (seq, par), 23 % (conc).
