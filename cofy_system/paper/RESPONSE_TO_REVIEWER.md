# Response to the reviewer

**Manuscript:** Federated Object Detection Under Extreme Non-IID Labels in Biomedical Imaging: System Proposal and Analysis of Reliability and Robustness

We thank the reviewer for a careful and constructive report. Below, each comment is followed by our response and the changes made. Section, table and page numbers refer to the revised manuscript.

---

## Major issues

### 1. Inconsistent class counts (KFM12)

**Response.** The reviewer is right. The models were trained on a class-stratified tenth of each node's training images (KFM12 and WBC), drawn with the seed. Table 2 already reported those counts, for example 5 instances of eimeria per seed. The fairness section, however, quoted counts from the full training split, from which the subsets are drawn (50 for eimeria, 307 for metastrongili, 201 for dicrocoelium). The Spearman correlations of the fairness table (now Table 6) had also been computed on the full-split counts. We now use the counts actually trained on everywhere, read from the list of images each run wrote at training time.

**Changes.**
- Section 3.4 states the subset rule exactly and says that every class count in the article is a count of what was trained on.
- Table 2 now gives, for KFM12 and WBC, the instances trained on per seed (mean over the ten confirmation seeds) next to the full-split counts, so the factor of about ten is explicit.
- Section 5.2.1 now quotes the counts trained on: eimeria about 5 per seed (4–7), metastrongili about 32, dicrocoelium about 22.
- Table 6 and the text are recomputed on these counts:

  | Scheme, use case | Spearman ρ before | Spearman ρ now |
  |---|---|---|
  | CWT, KFM12 | +0.63 | +0.69 (p = 0.02) |
  | FedAsync, KFM12 | +0.59 | +0.57 (p = 0.06) |
  | FedAvg, KFM12 | +0.73 | +0.69 (p = 0.02) |

  WBC and DENTEX are unchanged. The three rarest KFM12 classes are the same in both counts, so the fairness conclusions stand.
- Table 1: the KFM12 imbalance is now computed from the unrounded means (142×, previously 134× from rounded counts), and the share of images with other nodes' objects is rounded consistently (23%).
- `tools/dataset_tables.py` (new) recomputes Tables 1–3 from the per-run image lists; `tools/fairness_analysis.py` now counts on the same lists.

### 2. Auditable confirmatory protocol and evidence

**Response.** We agree that the main claim must be verifiable, not only described. In revising, we concluded that the article should no longer present the study as pre-registered. The registration we wrote before the runs covered one more use case and tested fixed sequences of margins (85% then 90% for CWT, 80% then 85% for FedAsync). The article reports three use cases and states, for each scheme, the highest level its lower bounds reach (point 5). Presenting this as a pre-registered test would overstate it. What was fixed before the first run, and can be checked by hash, is the code, the configurations and the script that computes the intervals.

**Changes.**
- **Archived evidence** (Declarations, Code availability; Appendix A.2):
  - the evaluation protocol of the study as it was carried out (`analysis/EVALUATION_PROTOCOL.md`), written after the runs and labelled as such;
  - the SHA-256 manifests of the frozen code, configurations and analysis script: 28 October 2026, 10:54 CET, before the first run; DENTEX's configurations on 1 November, before its first run on 2 November;
  - the frozen copy itself;
  - a new folder, `evidence/study/`, with the run-level records of every run reported. Per use case and seed, these hold the calibrated scores of the centralized reference and of both schemes per class, each run's summary, FedAsync's timeline on the virtual clock, and the list of images each node trained on.

  The original registration document is kept, unchanged, in the code release. The release will be archived on Zenodo with a DOI.
- **Seed-level values**: new Appendix A.4, Table A1, gives the paired mAP50 of the centralized reference, CWT and FedAsync on all ten seeds.
- **Release details**: the data and code statements are completed, apart from the two DOIs, which are assigned when the dataset and the release are published.

### 3. Training exposure of the centralized comparator

**Response.** We now specify the correspondence exactly. An *example* is an image with one node's partial labels and class mask. An image that contains classes of several nodes is one example per owner. The centralized model passes through each node's examples in turn, with that node's mask. It therefore processes every example as often as the federation does: once per owner and epoch, which on DENTEX means four times per image. It uses the same batch size (four examples from one node), so the number of optimizer steps is identical, and the same linear learning-rate decay.

**Changes.**
- New Table 3 (Section 4.1) gives, per use case and seed: unique images, examples per epoch, optimizer steps per epoch and in total, and epochs.
- The text lists the three differences that remain:
  1. per-node SGD momentum in the federation versus a single momentum buffer;
  2. a 20-step learning-rate warm-up that restarts at every federated round or turn, so the federated nodes train with 3.7–6.9% less summed learning rate (last column of Table 3);
  3. the order in which the examples are visited.

  These are now stated as part of what the ratio measures (Section 4.1 and Limitations).
- Section 3.1.2 points to this correspondence.

### 4. BatchNorm pooling formula

**Response.** The rule used in the study pools first and second moments, so the between-node variation of the means is included. Equation (6) in Section 3.2.4 now gives the formula:

σ² = Σ bᵢ(σᵢ² + μᵢ²)/Σ bᵢ − μ² = Σ bᵢσᵢ²/Σ bᵢ + Σ bᵢ(μᵢ − μ)²/Σ bᵢ

It is written in both forms to make the between-node term explicit. We checked that the study's frozen code computes exactly this (`_frozen/confirm/fedyolo/calibrate.py`). Within a node, the statistics are BatchNorm's own cumulative averages over its batches. No rerun was needed. The progressive calibration uses the same equation, with the stored statistics standing in for nodes not yet heard from.

### 5. Interpretation of the non-inferiority results

**Changes.**
- **Statistical target** (Section 3.1.2, new paragraph; Limitations, new item):
  - The interval describes variation across training seeds, with each dataset, split and node partition fixed. We list what the seed draws.
  - The result is a statement about the system on these use cases, not a clinical non-inferiority claim, and it does not extend to other sites, populations or acquisition protocols.
  - The contributions list now says the same.
- **Levels instead of margins** (Section 4.2): for each scheme we state the highest share of centralized mAP50, in steps of 5 percentage points, that its lower bound exceeds on every use case. This gives CWT 95% (lower bounds 96.2–98.7%) and FedAsync 90% (92.1–122.0%), with the same level also given per use case. Each level is a confidence statement about the ratio, not a test of a chosen margin. No clinical threshold exists for this endpoint, so a reader with an acceptable loss in mind can compare it with the lower bounds and the seed-level values.
- **Seed-level values**: Appendix A.4, Table A1 (see point 2).
- **Sensitivity analysis** (Section 5.1; Appendix A.4, Table A2): four further lower bounds, computed as a check:
  - the t-interval of the log ratio;
  - a BCa bootstrap;
  - the exact Wilcoxon interval of the Hodges–Lehmann estimate;
  - the t-interval with any one seed left out.

  For FedAsync, all remain above 90% (lowest 91.8%). For CWT, all remain above 95% except two on DENTEX: the bootstrap (94.4%) and the Wilcoxon interval (94.7%), both sensitive to one weak seed. The article states that CWT's 95% on DENTEX rests on the t-interval, and that CWT keeps at least 90% under every interval.
- **Wording**: the abstract now says the bounds are across ten fresh training seeds, with code, configurations and analysis frozen before the first run.
- **Metric definitions**: see the first minor point.

### 6. Robustness conclusions proportional to the study

**Changes.**
- **New Table 11** (Discussion) gives, for each finding, its level of evidence and its basis. The levels are: main study (fixed before its first run and computed once, the closest to confirmatory), descriptive (exploratory), post hoc, measured, computed and modelled.
- **Abstract**:
  - the robustness study is introduced as "a smaller robustness study";
  - the candidate defences are no longer reported as results, only as "follow-up tests [that] point to remedies that remain to be confirmed".
- **Contributions, Section 4.4, Discussion and Conclusions**:
  - The defence planned with the robustness study is distinguished from the candidate defences, which are labelled as designed after the results.
  - The candidate defences were tested post hoc on the same use case and seeds, and are described as limited demonstrations, not established performance.
- **Conclusions**: the robustness findings are prefixed with the scope of the study (one use case, three seeds, one non-colluding attacker).
- **Limitations**: the robustness item states that its findings are limited demonstrations.

### 7. Contribution relative to prior work and baselines

**Changes.**
- **Contributions** (Introduction) now separate what is new from what is adapted.
  - New: the ownership rule and what it requires, namely the split of the head into shared parameters and per-class rows, every scheme taking each class's rows only from its owner, and the barrier-free progressive calibration.
  - Adapted: the masked loss, the three merge rules, and BatchNorm re-estimation with classifier re-fitting.
- **Main contribution**: the ownership rule together with the evaluation protocol.
- **Why the closest methods are not compared directly** (Section 4.3):
  - partial-label federated methods address classification or segmentation, with losses built for those tasks;
  - others train on the server, which rule R2 excludes, or keep BatchNorm local, which leaves no single global detector;
  - federated detectors assume a shared label set, and without the masked loss they would learn other nodes' unlabelled objects as background.
- **Standard-method comparison** (Section 5.3): it now states that the comparison does not show general superiority over FedProx, FedBuff or CA2FL. It uses three seeds and one setting per method, and these methods differ from ours only in the merge rule.

### 8. System claims and deployment evidence

**Changes.**
- **Section 5.2.2** labels each cost quantity:
  - measured on the single GPU of the simulation: GPU time, and GPU energy from the board's power draw, excluding processor, memory, storage and network;
  - computed: communication, from model size and number of transfers, without protocol overhead and not measured on a network;
  - derived from the timing model: waiting time.

  It states that none of these is a multi-site measurement, and it lists the assumptions of the waiting-time estimate: computation only, node speeds within ±30%, ±20% jitter, no transfer time and no failures.
- **Section 3.3.1** now states that the networked implementation was exercised on one machine, has not run across sites, and produced no reported result.
- **Section 4.1** describes what the timing model includes.
- **Section 4.3, the contributions and the Limitations** use the same distinctions.

---

## Other changes

- **Presentation of the study design.** The article now describes the ten-seed study directly: the two schemes, run once on seeds 100–109, not used to design or tune them, with code, configurations and analysis frozen before the first run. We removed the account of the earlier design work in which the schemes were chosen: the subsections on the development phase in Sections 4.2 and 5.1 of the submitted version, with its table of results on the design seeds. None of the article's claims rests on it, and its records remain in the code release. Results on seeds 0–2 that are still reported (FedAvg in Table 6, the method comparison of Table 7 and the robustness study) are labelled by their seeds and as descriptive. The study is no longer presented as pre-registered (point 2).
- **New Figure 3** (Section 3.2) works through merging and calibration step by step on the WBC classes. It shows which entries of an upload the server takes, the staleness weight, the same update under CWT and FedAvg, BatchNorm pooling by Eq. (6), and the owners re-fitting their rows.
- **Abstract.** It states the levels reached across ten fresh training seeds, and it introduces the robustness study as smaller and the candidate defences as follow-up tests (248 words).

## Minor and presentation issues

- **Metric and aggregation** (Section 3.1.2):
  - mAP50 is the unweighted mean over classes of AP at IoU 0.5, computed with Ultralytics 8.4.84: confidence 0.001, NMS at IoU 0.7, at most 300 detections per image, 101-point interpolation.
  - Every class has at least 16 evaluation instances, so every class enters the mean; a class with no correct detection scores 0.
  - The aggregate ratio is the primary quantity because it is free of scale across use cases whose centralized mAP50 ranges from 0.50 to 0.96, and because it was fixed before the first run.
  - Per-class retention is secondary.
  - Section 5.2.1 notes that the rarest class has 16 evaluation instances.
- **Detection budget** (Sections 4.4, 5.4, Discussion, Conclusions):
  - It is now described as an inference-time containment measure, not a defence: it leaves the model unchanged and must be in place before an attack is known.
  - Its operational cost is stated: up to 300 candidate detections per class, longer post-processing and larger outputs.
  - The text notes that this evaluation was added after the sign-flip results.
- **Algorithm 1 and Section 3.3.2** now define the references exactly:
  - one list per part type of the 16 most recent sizes, stored as sent, before scaling, with the sender;
  - the reference is the median over the other nodes of each node's median;
  - the row list pools the rows of every class, one entry per owned row, added after the upload's rows are screened;
  - lists start empty, and a part is accepted as sent until three other nodes appear.
- **Data provenance and licensing** (Section 3.4, Data availability, Appendix A.1):
  - KFM12: acquisition period, sample origin and collecting unit; release on Mendeley Data.
  - WBC: PBC dataset, Mendeley Data version 1, CC BY 4.0, with the bounding boxes of the cited repository.
  - DENTEX: quadrant-enumeration subset of the challenge training data, Hugging Face release, CC BY-NC-SA 4.0, 634 images.
  - We redistribute no WBC or DENTEX images. The splits are rebuilt from the original data and checked against published SHA-256 fingerprints, and the per-run training lists are released.
  - We also corrected a statement in Appendix A.1: the DENTEX conversion re-encodes images as JPEG (quality 95); it is not lossless.
- **Ethics** (Declarations):
  - The human datasets are public, de-identified releases used under their licences, with no new data, contact or re-identification.
  - KFM12 comes from faecal samples collected for routine diagnosis; no animal was handled for this study, so no approval was required.
- **Drafting remnants**: all author notes have been removed except the two DOIs (KFM12 dataset, archived code release), which will be inserted when the records are published. References compile without undefined entries. The new and changed tables were checked in the compiled PDF.
