# Search for a federated scheme within 90 % of centralized: report

**Status:** 2026-10-08. Covers every alternative tried after the pre-registered
study (`PREREGISTRATION.md`, seeds 0–2), which stays untouched. Details and
decisions as they were taken are in `FOLLOWUP_PLAN.md`; live numbers come from
`python tools/results_table.py`.

## 1. Goal and rules

**Goal.** A sync scheme and an async scheme whose mAP50 is at least 90 % of
the centralized baseline. The test is the pre-registered one: the lower bound
of the 95 % CI of Fed:Cent over seeds must be ≥ 90 %. The scheme must hold on
all three datasets (kfm12, WBC, ChestX) and must not depend on the order in
which nodes arrive.

**Rules kept throughout**
1. Nodes learn about other nodes' classes only through the model they download.
2. The server holds no labeled data and does no training; it may average weights and statistics.
3. One class, one owner node.
4. **Same post-processing on both sides** (decided 2026-10-03): a calibrated federated model is compared with a calibrated centralized one.
5. **Async must be really asynchronous** (2026-10-04):
   - the node order comes from timing, not from a server-side shuffle;
   - no closing barrier;
   - calibration only in the progressive, asynchronous form.
6. **Seeds:** a method is screened on seed 0 (≥ 95 % everywhere to continue). Seeds are then added only as far as the CI needs them, decided once after seeds 0–2 (at most 12). Fresh seeds are used for confirmation.

**Datasets** (images per node, 4 nodes):

| Dataset | Classes | Unannotated other-class objects in a node's images | Centralized needs to plateau |
|---|---|---|---|
| kfm12 | 12 | 22 % | ~30 epochs |
| WBC | 7 | 0 % | ~5 epochs |
| ChestX | 10 | 53 % | ~55 epochs |

## 2. Bottom line

**No scheme passes yet.** WBC and kfm12 are within reach for every reasonable
scheme. **ChestX is the blocker:** every scheme in which nodes train in parallel
(sync, concurrent async) stays at 72–90 % of centralized on ChestX. That holds
across 18 variants: aggregation rules, update strength, divergence limits,
loss changes and calibration (§4, §7). Training the nodes one after the other
reaches 97–101 %. Options are in §8.

Best result per family (protocol v2: learning-rate decay, plateau budgets):

| Family | Best variant | kfm12 | WBC | ChestX | n | Status |
|---|---|---|---|---|---|---|
| Sync | `v2_sync` (+cal) | 94.6 % (94.0 %) | 111.9 % (114.8 %) | 77.7 % (79.9 %) | 3 | fails on ChestX |
| Sync, seed-0 screens | `v2_sync_slr2+cal` | 104.4 % | 120.4 % | 84.9 % | 1 | fails on ChestX |
| Async, concurrent (deployable) | `v2_vclock+acal` | 95.6 % (CI 87) | 115.4 % (CI 70) | 88.3 % (CI 84) | 3 | fails on ChestX |
| Async, one node at a time (reference only: needs a server-imposed order) | `v2_async+cal` | 102.1 % (CI 86) | 111.0 % (CI 99) | 97.6 % (CI 92) | 3 | means ≥ 97 % everywhere |

## 3. Training protocols

| Protocol | Recipe | Why |
|---|---|---|
| Primary (pre-registered) | 20 epochs per node, constant learning rate after warmup | Fixed in advance |
| **v2** (from 2026-10-02) | Learning rate decays linearly to 1 % over the whole budget (all arms, driven by global progress). Budget = epoch where centralized plateaus: kfm12 30, WBC 20, ChestX 56 | The pilot showed that, with a constant learning rate, the score of a single run swings ±0.02 between checkpoints 5 epochs apart. That is as large as the seed and order effects, and it made every CI too wide |

**Effect of v2 on centralized:**

| | kfm12 | WBC | ChestX |
|---|---|---|---|
| mAP50: primary → v2 | 0.655 → 0.729 | 0.459 → 0.549 | 0.202 → 0.309 |
| Seed spread (sd): primary → v2 | 0.069 → 0.028 | 0.046 → 0.031 | 0.013 → 0.005 |

## 4. Every alternative

Fed:Cent mean over seeds (CI-low where n ≥ 3), as kfm12 / WBC / ChestX. "+cal"
rows are compared with the calibrated centralized baseline (rule 4).

### 4.1 Primary protocol (20 epochs, constant learning rate), seeds 0–2

**Sync**

| Variant | What it does | Why it was tried | Result | Verdict |
|---|---|---|---|---|
| primary | FedAvg on shared layers, each class's rows from its owner only, unowned classes masked in the loss | The pre-registered method | 104 (72) / 137 (106) / 78 (74) | ChestX far below 90 % |
| fednova | Updates normalized by local step count (FedNova) | Nodes have different data sizes | 107 (80) / 134 (97) / 86 (66) | small ChestX gain |
| exclusion | At anchors on a node's own objects, other classes are trained as negatives (an object has one class) | Teaches separation without other nodes' labels | 104 (58) / 151 (102) / 87 (66) | best ChestX sync addition, WBC +0.06 |
| exclpl | exclusion + conditional pseudo-label distillation of other classes | Adds other classes' knowledge from the downloaded model | 106 (70) / 153 (106) / 86 (61) | like exclusion |
| fisher | Fisher-weighted merging (diagonal) | Merge each weight where it matters most | 103 (65) / 137 (79) / 81 (72) | no gain |
| pseudobox | Boxes of other classes from the downloaded model added as pseudo-labels | Fills missing annotations | 103 (75) / 128 (85) / 68 (50) | hurts (noisy boxes) |
| etf | Class rows fixed to a simplex ETF (FedETF) | Keeps classes of different nodes apart | 98 (60) / 160 (120) / 76 (63) | WBC up, ChestX down |
| primary+cal | Calibration round (BatchNorm re-estimation + owner re-fit of class rows) | Rows were fitted to features that changed after merging | 98 (80) / 140 (76) / 71 (68) vs calibrated centralized | helps centralized as much |

**Async** (one node at a time; fixed order unless stated)

| Variant | What it does | Why it was tried | Result | Verdict |
|---|---|---|---|---|
| primary | Each update applied on arrival, staleness-weighted | Pre-registered | 99 (70) / 102 (60) / 116 (100) | order-dependent (§4.3) |
| ca2fl | Buffered, cache-calibrated aggregation (CA2FL) | Async SOTA for heterogeneous data | 97 (52) / 125 (84) / 42 (12) | collapses on ChestX |
| fedbuff | Buffered averaging without the cache | Async SOTA | 108 (81) / 120 (91) / 94 (82) | close but CI too wide |
| fedasync | Update mixed at 0.6 (FedAsync) | Damps the last node's dominance | 107 (67) / 110 (71) / 110 (102) | fixed order, not eligible |
| exclusion_async, exclpl_async | as in sync | as in sync | 101 (74) / 117 (89) / 103 (79); 109 (68) / 110 (57) / 101 (76) | CI too wide |
| etf_async | as in sync (seed 0 only) | as in sync | 95 / 103 / 81 | dropped |

### 4.2 Calibration and order (primary protocol)

| Variant | What it does | Result | Verdict |
|---|---|---|---|
| order_r1..r3 | The primary async run with the node list rotated (seed 0) | ChestX 89 / 81 / 103 against 116 for the original order; sd over orders up to 3.6× the seed sd | Async results depend on the order |
| shufswa_r0..r3 | New random node order every cycle + average of the last 4 versions | Spread over 4 runs not smaller (kfm12 sd 0.035 vs 0.018) | Not retained alone |
| shufswa_r0+cal | Same + calibration round | Passed at n = 4 against plain centralized (CI-low 97 / 106 / 104) | Declared async winner, later **withdrawn** |
| confirm_async+cal | The winner on fresh seeds 4–7 | 103 (97) / 113 (**90**: 89.8) / 109 (100) against calibrated centralized | Fails WBC by 0.2 points; also uses a simulated order |
| budget_pilot | Centralized up to 100 epochs, mAP50 every 5 | Plateaus at kfm12 ~30, WBC ~5, ChestX ~55 epochs | Basis of protocol v2 |

### 4.3 Protocol v2

| Variant | Mode | What it does | Why it was tried | Result | Verdict |
|---|---|---|---|---|---|
| v2_sync (+cal) | sync | FedAvg as primary, v2 recipe | Re-test under a converged recipe | 95 (83) / 112 (85) / 78 (74); +cal 94 / 115 (94) / 80 | fails ChestX |
| v2_async (+cal) | async, one at a time | Random order every cycle + average of the last 4 versions | Order-robust sequential scheme | 102 (88) / 106 (82) / 99 (89); +cal 102 (86) / 111 (99) / 98 (92) | nearly passes; **reference only**, because a deployed server cannot impose the order |
| v2_vclock (+cal, +acal) | async, concurrent | Virtual clock: nodes train at the same time, order from data size × machine speed × jitter, staleness 2.4–2.7 on average; final model = average of each node's last version | Realistic concurrent deployment | 88 (76) / 109 (63) / 77 (62); +acal 96 (87) / 115 (70) / 88 (84) | fails ChestX and kfm12 |
| v2_vclock_dl (+acal) | async, concurrent | Same, but all nodes train until a common deadline (fast nodes upload more often) | Alternative stop rule | seed 0: 96 / 100 / 83; +acal 98 / 101 / 85 | fails ChestX |
| v2_vclock_a0 (+acal) | async, concurrent | Stale updates applied at full strength | Undo the staleness damping (mean weight 0.47–0.53) | seed 0: 97 / 86 / **0** (BatchNorm running mean drifted to 2.5·10⁵); +acal 93 / 93 / 76 | fails |
| v2_vclock_a02 (+acal) | async, concurrent | Mild damping (0.63 at staleness 3) | Middle ground | seed 0: 96 / 93 / 78; +acal 100 / 102 / 77 | fails ChestX |
| v2_chain (+cal) | sync | 4 models rotate between nodes (each trained by every node in turn), averaged every 4 rounds | Sequential learning inside a round-based scheme | seed 0: 93 / 97 / 75; +cal 94 / 107 / 78 | fails ChestX |
| v2_sync_slr2 (+cal) | sync | Server step 2× the averaged update on shared layers (FedOpt) | Undo averaging dilution | seed 0: 104 / 111 / 78; +cal 104 / 120 / 85 | fails ChestX |
| v2_sync_slr4 (+cal) | sync | Server step 4× | Full undo of dilution | seed 0: 84 / 84 / 44; +cal 90 / 77 / 81 | unstable |

### 4.4 Considered and not run

| Alternative | Why not |
|---|---|
| CCVR (server re-fits the classifier from uploaded feature statistics), FedGen/FedFTG (server generators), FedAlign (shares class names through a text encoder), FedAwS server step | Break rule 2 or rule 1 |
| spread, ntd, swa, FedProx, SCAFFOLD, FL-JSDDC, chain/sync20, fedasync with random order at 20 epochs | Queued under the primary protocol; dropped by user decision (2026-10-04) when the search moved to v2 (they could no longer win). Finished runs are kept |
| Buffered sum (FedBuff with server step = buffer) | Same principle as `v2_vclock_a0`, which failed |
| Polynomial staleness weighting | Same dilution as the default weighting |
| Closing synchronous calibration for async | Not allowed (rule 5); the progressive version (`+acal`) replaces it |

## 5. What was learned

1. **The training recipe mattered more than any aggregation rule.** Learning-rate decay plus plateau budgets raised every arm and halved the seed spread.
2. **Node order matters in async.** Its effect is as large as the seed's, or larger. Single-run comparisons with a constant learning rate were partly endpoint noise.
3. **Calibration is not federation-specific.** It mostly repairs the end state of constant-learning-rate training, and helps centralized just as much. Under v2 it still helps the concurrent async models (ChestX 77 → 88 %), but not centralized.
4. **The asynchronous progressive calibration is weaker than the closing round on ChestX** in some runs (`a02`: 0.244 vs 0.276). Nodes that contribute early re-fit their rows against incomplete BatchNorm statistics.
5. **Sequential ≠ concurrent.** All earlier "async" results were sequential: one node at a time, staleness 0. With real concurrency (virtual clock), ChestX and kfm12 drop below 90 %.
6. **Dilution is not the cause.** Applying updates more strongly (server step 2× / 4×, no staleness damping) did not raise ChestX, and the strongest settings broke training.
7. **Merging models that diverged in parallel is the cause.** The only scheme that matches centralized on ChestX applies each node's work on top of the previous node's result. Its final average only combines very close versions. Every scheme that averages or merges models trained in parallel on disjoint classes loses about 20 % on ChestX. That includes rotating models, which average 4 chains after a full cycle. ChestX is also where node images contain the most unannotated objects of other nodes' classes (53 %), so the parallel updates conflict most there.
8. **The gap to full labels is unrelated to federation.** WBC is at about 58 % of the full-label oracle, and so is centralized.

## 6. Bugs found and fixed during the search

| Bug | Effect | Fix |
|---|---|---|
| Async delta rule applied stale deltas to BatchNorm running variances | One layer negative → `v2_vclock_dl` WBC seed 0 scored 0.000 | Variances clamped at 0 (re-run: 0.579) |
| Same rule at full strength (`a0`) lets BatchNorm running means drift | `v2_vclock_a0` ChestX scored 0 | Open: BatchNorm statistics should be averaged, not delta-accumulated |
| "ASYNC WINNER" matched the text "SYNC WINNER" | The sync search would have been skipped | Word-boundary match |
| Calibration tool had no config for seeds without a study config | Queue halted | Built from the seed-0 config |

## 7. Drift-limiting screens (2026-10-07/08): not solved

ChestX first, seed 0. Comparable same-seed baselines: `v2_sync` 0.244,
`v2_vclock` 0.261 (+acalr 0.289).

| Alternative | Mode | What it does | Why it was tried | ChestX result | Verdict |
|---|---|---|---|---|---|
| `v2_sync_q` | sync | ¼ epoch per round, 4× rounds (same compute), reshuffled data | Few local steps should make FedAvg approach mini-batch SGD over all nodes | 0.235 (75.0 %); +cal 0.237 | no gain |
| `v2_vclock_q` | async, concurrent | ¼-epoch uploads; BatchNorm running statistics interpolated instead of delta-updated | Small stale deltas ≈ asynchronous SGD; no BatchNorm drift | 0.256 (81.6 %); +acalr 0.260 | no gain |
| `v2_sync_excl` | sync | Exclusion loss, reshuffled | Best ChestX sync addition at 20 epochs | 0.243 (77.6 %); +cal 0.246 | no gain |
| `v2_vclock_excl` | async, concurrent | Exclusion loss, BatchNorm interpolated | Same | 0.238 (75.9 %); +acalr 0.272 | no gain |
| `v2_sync_rs` | sync | Control: per-round reshuffle only (each newly built loader otherwise restarts from the same image order) | Isolates the data-order effect | 0.226 (72.0 %); +cal 0.243 | order was not hurting |
| `+acalr` on `v2_vclock` (seeds 0–2) | async | Progressive calibration, then each node re-fits its rows once on the complete BatchNorm pool | Early contributors fitted rows to a partial pool | 89.2 % (CI-low 84.1) vs +acal 88.3 % | small gain, still < 90 % |
| ¼-epoch rounds + 4× server step | sync | Averaging with the same per-image learning signal per node as sequential training | Combines the two levers that failed separately | — | not run (declined) |

**Updated reading of §5.7.** Neither the size of the divergence before a merge
nor the strength of the merged update explains the ChestX gap. The only
consistent difference is structural: applying the nodes' work one after the
other (sequential) reaches centralized on ChestX (97–101 %); every parallel or
concurrent scheme lands at 72–90 %.

## 8. Options from here

1. **Sequential scheme as the synchronous mode.** The server schedules the nodes in turn, in a new random order every cycle, and averages the last versions (`v2_async` design). This is sequential federated learning (Li & Lyu, NeurIPS 2023; cyclic weight transfer, JAMIA 2018). It is synchronous in nature: the server coordinates and nodes wait for their turn. It is not asynchronous, which is why it was excluded as the async scheme. Under v2 it reaches 102 / 106 / 99 % (+cal 102 / 111 / 98 %, CI-low 86 / 99 / 92 at n = 3; kfm12's CI projected to pass at n = 4). Cost: no parallel training (wall-clock = sum of the nodes' training times).
2. **¼-epoch rounds + 4× server step** (§7, declined so far).
3. **Stop and write up:** parallel and concurrent schemes lose about 20 % on ChestX against a well-trained centralized baseline; sequential training does not.

## 9. Main threshold lowered to 85 % (user decision 2026-10-08)

**What changed.** The criterion is now: lower bound of the 95 % CI of Fed:Cent
over seeds ≥ 85 % (was 90 %). The change is post hoc, taken after the results
were seen, and is recorded as a deviation in `PREREGISTRATION.md`. The table
keeps a `@90` column with the pre-registered verdict, and the seed-0 screening
bar follows the threshold (85 + 5 = 90 %). Nothing has to be re-run: every
existing result was re-evaluated.

**Situation at 85 %:**

| Mode | Scheme | kfm12 | WBC | ChestX | n | Verdict at 85 % |
|---|---|---|---|---|---|---|
| Async, concurrent (deployable) | `v2_vclock+acalr` (progressive calibration with refresh) | 96.2 % (CI-low 89.3) | 117.5 % (70.4) | 89.2 % (84.1) | 3 | Means pass; CI-low short on WBC and ChestX. Projected to pass with 1 more seed (n = 4) |
| Async, concurrent | `v2_vclock+acal` | 95.6 % (86.6) | 115.4 % (69.6) | 88.3 % (84.0) | 3 | same, n = 4 |
| Sync (parallel) | `v2_sync+cal` | 94.0 % (79.6) | 114.8 % (94.2) | 79.9 % (77.7) | 3 | fails: ChestX mean below 85 % |
| Sequential, server-scheduled turns | `v2_async+cal` | 102.1 % (85.7) | 111.0 % (98.8) | 97.6 % (92.3) | 3 | **passes**; counts only if accepted as the synchronous scheme (§8, option 1) |
| Same design, primary protocol, fresh seeds 4–7 | `confirm_async+cal` | 103.2 % (96.8) | 112.5 % (89.8) | 109.0 % (100.3) | 4 | passes (confirmation) |

No seed-0 screen would have continued under the 90 % screening bar: the
closest eligible ones stay below 90 % on ChestX.

**Why the concurrent-async scheme needs one more seed (a new seed, not a
re-run).** The test uses the CI lower bound, not the mean. With 3 seeds the
interval is wide (t = 4.30), so the lower bound sits below 85 % on WBC and
ChestX although the means are 118 % and 89 %. A fourth seed (t = 3.18, larger
√n) is projected to lift it above 85 % everywhere. ChestX is the tight case:
projected CI-low about 86 % for a mean of 89 %.

## 10. Sync winner and the async candidate (2026-10-08)

**Sync winner.** `v2_async+cal`, the sequential scheme, is counted as sync by
user decision.

- **Why it is synchronous.** The server schedules the nodes one at a time, in a new random order every cycle; a cycle completes only when every node has trained. The output model averages the versions right after each node's last turn, then gets the closing calibration round.
- **Result** (85 % criterion, n = 3): 102.1 % (CI-low 85.7) / 111.0 % (98.8) / 97.6 % (92.3).
- **Caveats.**
  - It was selected among many variants on seeds 0–2 and has not been confirmed on fresh seeds under v2. The same design under the primary protocol passed a fresh-seed confirmation at 85 % (seeds 4–7).
  - Wall-clock time is the sum of the nodes' training times.

**Async candidate.** `v2_vclock+acal` / `+acalr`:
- **Design:** nodes train concurrently, each update is applied on arrival with staleness weighting, the final model averages each node's last version, and calibration is progressive.
- **Status:** seed 3 is running (with its centralized baseline), as the 85 % projection requires.

## 11. What can be claimed (2026-10-28)

Seed 3 for the concurrent async scheme is done: ChestX 83.5 % on that seed, so
at n = 4 its ChestX CI-low is 82.6 % and it fails the 85 % criterion. No async
winner.

**Guaranteed loss bound per scheme** (lower bound of the 95 % CI over seeds,
worst dataset; calibrated runs vs calibrated centralized, protocol v2):

| Scheme | n | mAP50 kept (worst dataset) | relative loss ≤ | absolute loss ≤ (mAP50) | mAP50-95 kept | relative loss ≤ |
|---|---|---|---|---|---|---|
| Sync, sequential (server-scheduled turns) `v2_async+cal` | 3 | 85.7 % (kfm12) | 14.3 % | 0.103 | 86.4 % | 13.6 % |
| Async, concurrent `v2_vclock+acalr` | 4 | 82.6 % (ChestX) | 17.4 % | 0.079 | 81.7 % | 18.3 % |
| Sync, parallel FedAvg `v2_sync+cal` | 3 | 77.7 % (ChestX) | 22.3 % | 0.159 | 75.3 % | 24.7 % |

Mean Fed:Cent (mAP50):
- sequential sync: 102 / 111 / 98 %;
- concurrent async: 95 / 114 / 88 %;
- parallel sync: 94 / 115 / 80 %.

**Caveats.** All of this is post hoc: the threshold was chosen after the
results, and the schemes were selected among about 30 variants on the same
seeds. See `SUBMISSION_READINESS.md`.
