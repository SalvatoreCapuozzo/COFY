# Trustworthiness of the system: privacy, security, accountability, regulation, data

**Date:** 2026-10-28. **For:** the paper for *Trustworthy and Responsible
Federated Learning* (Discover Artificial Intelligence), mapped onto the
collection's themes. Measured results are in their own documents:
- confirmation: `CONFIRMATORY_PREREG.md`, then `CONFIRMATORY_RESULTS.md`;
- fairness: `FAIRNESS.md`;
- cost: `COST.md`;
- robustness: `ROBUSTNESS_PLAN.md`, then `ROBUSTNESS.md`;
- the search and its negative results: `SEARCH_REPORT.md`.

## 1. Privacy

**What leaves a node:**

| When | What | Size |
|---|---|---|
| Every turn or round | the full model weights after local training (fp32) | 99.4 MB |
| Every turn or round | number of images, number of boxes per owned class, number of local steps | a few numbers |
| Calibration (once, at the end) | BatchNorm statistics of its data and the batch count | 0.25 MB |
| Calibration (once, at the end) | the rows of its own classes | 0.03 MB |
| Never | images, labels, optimizer state, local validation results | — |

**What a curious server or eavesdropper could learn:**
- **Model updates.** In the sequential and concurrent schemes the server applies each node's update on its own, so it sees individual updates. Gradient-inversion and membership-inference attacks on individual updates are known for classifiers. They are less studied for detectors trained for several local epochs, but cannot be excluded.
- **BatchNorm statistics.** These are the per-channel mean and variance of the node's activations. They summarize the data and are not images, but they are an input to some inversion attacks.
- **Ownership metadata.** Which classes a node owns is public within the federation by design (the registry).

**Mitigations and their fit:**

| Mitigation | Parallel sync | Sequential sync | Concurrent async | Notes |
|---|---|---|---|---|
| Transport encryption (HTTPS, implemented) | yes | yes | yes | protects against eavesdroppers, not against the server |
| Secure aggregation (server sees only the sum) | yes | **no** | **no** | sequential and asynchronous updates are applied one by one; a buffered asynchronous rule (FedBuff) restores it for buffers of ≥ 2 updates |
| Client-level differential privacy (clip + noise) | yes | yes | yes | costs accuracy; not measured yet (optional experiment, `SUBMISSION_READINESS.md` gap 4) |
| Calibration statistics | pooled in the clear | — | — | could be pooled with secure aggregation (they are sums) |

**To state in the paper:**
- The scheme with the best accuracy (sequential) is incompatible with secure aggregation.
- This trades privacy against the server for accuracy.
- The trade-off is a deployment decision.

## 2. Security (implemented in `cofy/net`, see `docs/NETWORK_FEDERATION.md`)

- **Authentication:**
  - room password, plus a per-node secret token issued at join and stored hashed;
  - admin actions only from the server machine or with the admin token.
- **No code execution from uploads:** only plain tensors, loaded with `torch.load(weights_only=True)`. A crafted pickle is rejected.
- **Validation of every upload:**
  - tensor names and shapes must match the room's model;
  - values must be finite;
  - uploads have a size limit;
  - repeated uploads are idempotent (`push_id`).
- **Ownership:** a node's class rows are taken only for the classes the server registered for it.
- **Availability:**
  - heartbeats and lost-node handling;
  - retries with back-off;
  - state persisted and restored after a restart;
  - node reclaim.
- **No server-side data use** outside developer mode.
- **Tested live:** wrong token → 403; crafted pickle → 400; NaN weights → 400; server killed mid-round → restored.

**Still trusted, a gap the robustness study addresses:**
- the image and box counts a node reports (its FedAvg weight);
- the size of its update;
- its BatchNorm statistics.

## 3. Robustness to malicious or faulty nodes

`ROBUSTNESS_PLAN.md` gives the threat model and the attacks: eight, three of them specific to detection. It also describes the defence: ownership-aware robust aggregation with per-part norm bounds, a trimmed mean for parallel sync and an audit log. The study runs after the confirmation (about 53 GPU hours).

## 4. Fairness

From `FAIRNESS.md` (exploratory seeds; recomputed on the confirmation seeds with `--confirm`):
- **Sequential sync** keeps every class's AP close to centralized (worst class 83–97 %), with no dependence on class size.
- **Concurrent async and parallel FedAvg** hide a large spread behind a good mean: worst class 44 % and 29 % on kfm12 and ChestX respectively.
- **Rare classes lose more** on kfm12 (Spearman +0.80 between retention and training boxes, p < 0.001).

**For the paper:** report the worst class next to the mean, and make "no class below X % of centralized" part of the acceptance criteria. This is a per-class guarantee that a mean-only test does not give.

## 5. Accountability and transparency

- **Pre-registration with dated deviations** (`PREREGISTRATION.md`). The threshold change (90 → 85 %) is recorded as a post-hoc deviation, and the original text is untouched.
- **Confirmatory study:**
  - registered before its first run (`CONFIRMATORY_PREREG.md`), with fixed seeds, margins, test and analysis script;
  - its code and configs are frozen (`_frozen/confirm/`), and SHA-256 records them (`confirm_manifest.json`, checked with `tools/confirm_manifest.py --check`);
  - the held-out dataset was named before its first run (amendment 1).
- **Negative results are reported:**
  - about 30 variants and every failed fix (`SEARCH_REPORT.md`);
  - a withdrawn "winner" that failed its fresh-seed check (`runs/variants/_withdrawn.txt`).
- **Audit trails:**
  - every queue decision and its reason (`_search_log.json`, `_queue.log`);
  - each concurrent run's timeline (`virtual_schedule.json`: arrival order, staleness);
  - every defence decision (`robust_log.jsonl`);
  - the network server's persisted state;
  - GPU power during the confirmation (`_power.csv`).
- **To add:** a model card per released model:
  - intended use;
  - classes and their owners;
  - training data per node;
  - Fed:Cent with its confidence bound;
  - per-class worst case;
  - known failure modes (the concurrent scheme on co-occurring findings, ChestX);
  - the threat model.

## 6. EU AI Act (Regulation (EU) 2024/1689): what the system supports

Medical-image detection software used for diagnosis is, as a medical device under the MDR, a high-risk AI system (Art. 6(1), Annex I). The system is research software, so **no compliance is claimed**. The table shows which requirements its design supports and what a provider would still need.

| Article | Requirement | Supported by | Still needed |
|---|---|---|---|
| 9 | Risk management | threat model and attack study; failure modes from the search (order effects, rare-class loss) | a provider's risk management process |
| 10 | Data governance | data stay at each site; documented preparation, splits and fingerprints (`study_manifest.json`); class ownership registry | provenance and consent records per site (§7); bias examination beyond class balance |
| 11, Annex IV | Technical documentation | system description, pre-registrations, frozen code, reports | packaging as technical documentation |
| 12 | Record-keeping (logs) | server state, per-update audit logs, defence decisions, schedules | retention policy; logs in deployment, not only simulation |
| 13 | Transparency to deployers | Fed:Cent with confidence bounds; per-class retention; model cards (to add) | instructions for use |
| 14 | Human oversight | operator dashboards (server and client), lost-node and reclaim controls, developer-mode evaluation | clinical workflow integration |
| 15 | Accuracy, robustness, cybersecurity | accuracy with confidence bounds against centralized; robustness study; authentication, safe decoding, validation | clinical validation; adversarial testing beyond one attacker |

GDPR: federated training minimizes data transfer. Model updates and BatchNorm statistics may still count as personal data in some interpretations (§1), which needs a data-protection impact assessment in deployment.

NIST AI RMF:
- **Govern:** pre-registration, frozen code.
- **Map:** threat model, data provenance.
- **Measure:** confidence-bounded non-inferiority, fairness, robustness, cost.
- **Manage:** the defence, lost-node handling, withdrawal of a failed result.

## 7. Data provenance and ethics

| Dataset | Source | Licence | Human data? | Notes |
|---|---|---|---|---|
| kfm12 (12 parasite-egg classes) | the author's own data (S. Capuozzo, Università di Napoli Federico II). Animal faecal samples prepared with Mini-FLOTAC, imaged with the Kubic FLOTAC Microscope (KFM) | own data; the full 12-class dataset will be released by the author (an earlier 2-class subset is public under CC BY 4.0) | no (animal faecal samples) | the existing public subset, the *Trematodes AI-KFM Dataset* (doi:10.17632/6bjcxw8dx2.1), shows the data's provenance. Cite the full release once it has a DOI. No ethics approval needed |
| WBC (7 blood-cell classes) | *White-Blood-Cell-Detection-Dataset* (GitHub, medmabcf): bounding boxes added to *PBC_dataset_normal_DIB* (Acevedo et al. 2020, *Data in Brief* 30:105474; Mendeley Data snkd93bnjr/1, CellaVision DM96, Hospital Clinic of Barcelona), platelets excluded | CC BY 4.0 (both) | yes, anonymized, public | one cell, one box per image. The box annotator is the GitHub release's author (method not documented): state it. Cite both |
| ChestX-Det10 | Deepwise-AILab, subset of NIH ChestX-ray14 with boxes | per the repository / NIH terms (check) | yes, anonymized, public | cite both |
| DENTEX (fourth use case) | MICCAI 2023 challenge, Hamamci et al. (arXiv:2305.19112) | CC BY-NC-SA 4.0 | yes, anonymized, public | non-commercial: fine for research, state it |

**Ethics statement (draft):**
- The study uses publicly released, anonymized human datasets (WBC, ChestX-Det10, DENTEX) and the authors' own veterinary dataset (kfm12), obtained from animal faecal samples.
- No new human data were collected, and no ethics approval was required.

## 8. Code and data availability (draft statement)

- **Code:** the code (federated system, network server and clients, study tools) and all configs, seeds, logs and analysis scripts will be released at [repository URL] under [licence].
- **Integrity:** the confirmatory study's code and configs are identified by the SHA-256 records in `analysis/confirm_manifest*.json`.
- **Public datasets:** obtainable from their sources; conversion and splitting scripts are included (`tools/convert_*.py`, `tools/split_dataset_into_nodes.py`).
- **kfm12:** the full 12-class dataset will be released by the author [repository, DOI]; a 2-class subset is already public (doi:10.17632/6bjcxw8dx2.1).
