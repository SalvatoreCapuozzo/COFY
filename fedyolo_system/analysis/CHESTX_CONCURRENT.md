# Why the concurrent scheme loses on ChestX-Det10 (exploratory, 2026-11-02)

**Context.**
- The pre-registered confirmation (`CONFIRMATORY_RESULTS.md`) failed the concurrent scheme at 80 %: ChestX Fed:Cent was 83.2 %, CI-low 79.7 %.
- That result is final and is reported as is.
- This document looks for its cause, for bugs, and for a possible fix. Anything found here is exploratory. A fixed scheme would need a new pre-registered confirmation on new seeds.

Numbers come from `tools/cx_diagnosis.py` and the confirmation runs, all read-only.

## 1. Where the loss is

**Per node** (mean retention of the node's classes; acal_refresh, seeds 100–109):

| Node | Classes (training boxes) | Concurrent | Sequential |
|---|---|---|---|
| A | Consolidation (2024) | 102 % | 101 % |
| B | Effusion (1666) | 92 % | 102 % |
| C | Atelectasis, Calcification, Nodule, Pneumothorax (169–749) | 82 % | 103 % |
| D | Emphysema, Fibrosis, Fracture, Mass (134–603) | 71 % | 97 % |

- **The two nodes that own many small classes take the loss.** Their classes average 71–82 %. The two nodes with one large class each average 92–102 %.
- **Calibration does not explain it.**
  - Before calibration, the scheme is at 77 %.
  - The closing calibration (bn+head, 83.1 %) and the barrier-free one (acal_refresh, 83.2 %) end at the same place.
  - Calibration retrains every class row at full learning rate, so the deficit lies in the shared layers, not in the rows.

## 2. Bugs: none found in the concurrent path

Checked in the frozen copy, which is what the confirmation ran:
- **Staleness.** It is computed as a version difference, and the virtual clock trains each update from the version current at its pull time.
- **Update application.**
  - Delta application is correct.
  - The BatchNorm variance clamp is present.
  - The learning-rate schedule follows each node's own cycle count.
  - The final per-node average uses the version right after each node's last update.
- **Working tree.** Its concurrent path is the frozen code plus the robustness hook, which does nothing when no attack or defence is set.

Two design weaknesses, neither of which explains the ChestX loss:
- **Owned class rows are damped by staleness like shared weights,** although only their owner writes them. Calibration retrains them at full learning rate, so this is undone before evaluation.
- **BatchNorm running statistics are accumulated as deltas** (`bn_merge: delta`). This affects only the uncalibrated model: training uses batch statistics, and calibration re-estimates the running ones.

## 3. Hypotheses tested on existing data

| Hypothesis | Test | Verdict |
|---|---|---|
| Nodes with less data finish their 28 updates early, and the rest of the federation erodes their classes (recency) | Class retention against the number of updates applied after the owner's last update, with node effects (100 class-seeds) | **No.** Slope −0.05 points per update (t = −0.3). The raw node-level correlation (+0.58) is the node identity. On kfm12 the relation is even negative |
| Staleness damping (mean weight 0.47) leaves the model under-trained, and ChestX is the only dataset still improving at the end of its budget | Development screens with weaker or no damping (`v2_vclock_a02`, `v2_vclock_a0`, seed 0) | **Not supported.** 77–78 % (a02) and 0 % / 76 % (a0, unstable BatchNorm), against 88 % for the scheme |
| Smaller, less stale uploads would help | `v2_vclock_q` (¼-epoch uploads, seed 0) | **No:** 82 % |
| A common stop time would help | `v2_vclock_dl` (deadline stop, seed 0) | **No:** 85 % against 90 % for the same seed |

**What remains** is the finding of the development phase:
- ChestX is the dataset with co-occurring findings: 53 % of a node's images contain other nodes' findings, unlabelled.
- Every scheme that merges updates computed from the same or a stale base loses there: parallel FedAvg about 78 %, concurrent about 83–88 %.
- The only scheme that does not lose chains updates, each starting from the previous one's result (sequential, 98 %).
- The classes that suffer are the small ones, whose features have the weakest signal.

## 4. Experiments queued (2026-11-02, ahead of DENTEX; about 18 GPU hours)

**1. `diag_cx_last` (calibration only, about 1.5 h).**
- Runs the confirmation's 10 concurrent ChestX runs evaluated on their last global version, instead of the average of the versions after each node's last update.
- Question: does the final averaging, which mixes versions up to half a run apart, cost accuracy?
- **Result (2026-11-02): no, the averaging helps.** The last version is worse at every stage (`tools/cx_diagnosis.py` §1):

  | Stage | Average (registered) | Last version | Last − average |
  |---|---|---|---|
  | before | 77.4 % | 67.3 % | −10.0 (sd 5.3) |
  | bn+head | 83.1 % | 75.0 % | −8.1 (sd 4.7) |
  | acal_refresh | 83.2 % (nodes A 102, B 92, C 82, D 71) | 75.8 % (A 98, B 91, C 69, D 62) | −7.3 (sd 4.5) |

  The loss again falls on nodes C and D. The final model is not the cause; the averaging removes part of the damage.

**2. `cx_chain1`, `cx_chain4` (development seeds 0–2, about 8 h each).**
- The concurrent scheme with every node's last 1 or 4 cycles run as a server-ordered chain at the end. The option is `async_final_chain` in `fedyolo/server.py`, unit-tested with stubbed training.
- Compute is the same. The federation waits for the chain:
  - wall-clock is about +7 % (1 cycle) or +30 % (4 cycles) of the concurrent scheme's;
  - the sequential scheme takes 2.5–3.2× the concurrent wall-clock.
- Question: does the chained ending recover the small classes? The two lengths give a dose–response.

**Rule, fixed before the results:**
- **A candidate is promising when both hold:**
  - its mean Fed:Cent on seeds 0–2 (acal_refresh) is at least 5 points above `v2_vclock` on the same seeds (89.2 %);
  - node D's classes average at least 90 %.
- **A promising candidate then needs:**
  - a check on kfm12 and WBC, with the same seeds;
  - a new pre-registered confirmation on new seeds (110–119), with new centralized references, all four datasets and the same margins.

  That costs about 4–5 GPU days, so it must start by about 25 November to finish before 15 December.
- **Otherwise,** the paper reports the concurrent scheme's failure on ChestX with this analysis of its cause.

## 5. What class ownership protects, and strategies from the literature (2026-11-02)

**What ownership covers.** Rule R3 (aggregate a class's rows only from its owner) protects 0.01 M of the model's 24.8 M parameters (0.04 %).

| Part | Parameters | Share | Written by |
|---|---|---|---|
| Backbone and neck | 23.4 M | 94.3 % | every node |
| Box branch | 0.85 M | 3.4 % | every node |
| Class tower | 0.55 M | 2.2 % | every node |
| Class rows | 0.01 M | 0.04 % | the owner only |

A small, stable per-class loss therefore cannot come from ownership alone:
- **Sequential scheme:** it provides the guarantee implicitly. Every node revisits the shared layers every round, at full strength, each update starting from the previous one. Its worst node is at 97 % and its worst class at 86 %.
- **Concurrent scheme:** it has no such mechanism. Its worst node is at 71 %.

**Strategies, ranked by fit with R1–R3:**

| Family | Idea | Literature | Fit | Tried here |
|---|---|---|---|---|
| Feature ownership (parameter isolation, redundancy of capacity) | One class tower per owner, aggregated only from its owner (R3 extended from rows to the tower; +0.55 M per node, +7 % of parameters). Optionally, owner adapters in the neck | FedWeIT (Yoon et al., ICML 2021); Fed-MENU (Xu et al., IEEE TMI 2023, one encoder per organ); FedIOD (Wan et al., IEEE JBHI 2024, one decoder per organ) | full | no |
| Server-side conflict correction (no data needed) | Remove from a stale update the components that conflict with the other nodes' recent updates, or compensate its delay | FedFV (Wang et al., IJCAI 2021, fairness via projection); OrthoDC (AISTATS 2026); Fed-OGD (IEEE TC, stragglers as forgetting); DC-ASGD (Zheng et al., ICML 2017) | full | no |
| Knowledge preservation in local training | Distil the pulled model's outputs or features for the classes a node does not own | FedNTD (Lee et al., NeurIPS 2022); ConDistFL (Wang et al., 2024, partially labelled segmentation) | full, +~30 % compute | partly: `exclpl` (sync, 86 %), `pseudobox` (hurt); NTD queued then dropped |
| Importance-weighted protection | Nodes share Fisher diagonals and penalize moving parameters important to others | FedCurv (Shoham et al., 2019) | full, communication ×3 | Fisher merging (server side): no gain |
| Class-level fixes | Rare-class resampling (RFS, IRFS), restricted softmax (FedRS), logit calibration (FedLC), classifier re-training (CReFF) | Gupta et al. 2019; Yaman et al., NeurIPS 2023; Li & Zhan, KDD 2021; Shang et al., IJCAI 2022 | full | masking (≈ FedRS) and calibration (≈ CReFF) are already in the system. The deficit lies in the features, so classifier-level fixes cannot close it. RFS would also have to be applied to the centralized reference |
| Replicated ownership / coded redundancy | Two owners per class; parity data at the server | Coded FL (Prakash et al., IEEE JSAC 2021) | **breaks R3 / R2** | not applicable |

## 6. Fix screens (2026-11-02, user decision: ChestX only, continue only if it solves ChestX)

**Candidates.** Each is a config option on the concurrent scheme (`v2_vclock`), tested with stubbed training on CPU and with one-cycle smoke runs on WBC (`cxsmoke_*`).

| Variant | Change | Implementation |
|---|---|---|
| `cx_chain4` | every node's last 4 cycles run as a server-ordered chain (staleness 0) | `federation.async_final_chain: 4` |
| `cx_towers` | one class tower per node (2 × 3 small convs per tower, +7 % parameters), trained by its own classes' loss only and aggregated only from its owner. At initialization the output is identical to the standard model's | `federation.owned_towers: true`, `fedyolo/model.py` `OwnedClassTowers` |
| `cx_proj` | before an update is applied, tensor by tensor, its components opposing the updates applied while it was computed are removed (FedFV-style). Agreeing components are kept. Rows and owned towers are left as they are. A log is written to `conflict_log.jsonl` | `federation.async_conflict: project` |

**Gate** (`tools/cx_gate.py`, fixed before any result).
- **First run:** ChestX seed 3, the development seed where the problem shows (`v2_vclock`: 83.6 %, worst node 70 %).
- **Pass:** Fed:Cent ≥ 90 % (acal_refresh) and every node's classes ≥ 85 %. Seeds 1 and 2 are then queued at once.
- **Fail:** the candidate stops there.
- **Why not seed 0 (the earlier rule in §4):** it does not show the problem (`v2_vclock` 91.5 %, node D 102 %), so it cannot tell a fix from luck.
- **After three seeds, the candidate counts as solving ChestX when both hold:**
  - its mean is at least 5 points above `v2_vclock` on seeds 1–3 (86.5 %), so ≥ 91.5 %;
  - its worst node averages ≥ 85 %.
- **Only then:** kfm12/WBC checks, and a new pre-registered confirmation.

**Smoke runs (WBC, one cycle, 2026-11-02): all three complete in the real training loop.**
- `cxsmoke_chain` ends at mAP50 0.339 before calibration.
- `cxsmoke_proj` and `cxsmoke_towers` end at 0.0.
- The cause is the standard concurrent merge, not the fixes:
  - Their last version has about 350 BatchNorm channels, in 43–47 layers from `model.0` on, whose variance the stale deltas drove to zero (clamped).
  - Projection does not touch BN statistics, and the towers live only in the head.
  - The chain smoke runs at staleness 0 and has none.
- Calibration rebuilds the statistics: `cxsmoke_towers` goes from 0.001 to 0.42 (bn+head). The screens are judged after calibration.
- In the smoke, projection kept 99.9 % of the update norm (17–34 of 486 tensors projected), because after one cycle all updates point the same way. Whether it acts more strongly later is visible in `conflict_log.jsonl` of the screen.

**Comparator caveat for `cx_towers`.** It has 7 % more parameters than the centralized reference. A confirmation would have to train the centralized reference with the same towers.

**Gate results (ChestX seed 3, acal_refresh; decided 2026-11-02 20:10 to 2026-11-03 01:34): all three fail.**

| Variant | Fed:Cent | Node A | Node B | Node C | Node D | Gate |
|---|---|---|---|---|---|---|
| `v2_vclock` (reference) | 83.6 % | 102 | 95 | 78 | 70 | – |
| `cx_chain4` | 87.8 % | 99 | 99 | 77 | 79 | fail (C < 85, overall < 90) |
| `cx_towers` | 81.5 % | 97 | 102 | 80 | 60 | fail |
| `cx_proj` | 87.6 % | 93 | 105 | 71 | 84 | fail |

- **Chain and projection:** about +4 points overall, but a single run. The confirmation's ChestX ratios vary with sd ≈ 4.7 between seeds, so this is within noise. Node C stays at 71–77 %.
- **Projection was almost inactive:** it kept 99.9 % of the update norm on average (minimum 99.2 %; about 166 of 486 tensors touched per update, each only slightly), with no change over the run. Tensor-level conflicts between the nodes' updates are too small for it to act on.
- **Owned towers made node D worse (60 %).** Isolating the 2.2 % of parameters in the class tower removes what node D gains from the other nodes' data and leaves the shared 94 % unprotected. The deficit lies in the shared backbone and neck, not in the tower.
- **Conclusion:** none of the three solves ChestX. Per the user's decision, nothing more is run for them: no further seeds, no other datasets, no new confirmation. The paper reports the concurrent scheme's failure on ChestX, with the cause analysis (§1–3) and these negative screens.
