# Submission readiness: "Trustworthy and Responsible Federated Learning" (Discover Artificial Intelligence)

**Date:** 2026-10-28. **Target:** the Springer Nature collection *Trustworthy
and Responsible Federated Learning* in *Discover Artificial Intelligence*
(open access). Collection page: https://link.springer.com/collections/fdcjaidaif.

**Submission deadline: 15 December 2026** (confirmed by the author).

## 1. Collection scope (as found in search listings; check against the page)

**Aims.** Methodological, system-level and governance-oriented research that
makes federated learning not only accurate but also reliable, fair,
transparent and socially aligned. It refers to the European Commission's
*Ethics Guidelines for Trustworthy AI* and the NIST *AI Risk Management
Framework*.

**Topics and keywords listed:**
- fairness-aware methods; privacy and security techniques;
- robustness to adversarial and Byzantine threats;
- explainability and interpretability;
- accountability, transparency and governance; regulatory compliance (EU AI Act);
- human-in-the-loop learning; trust metrics;
- real-world deployment; decentralized / peer-to-peer architectures.

## 2. What the results support

Setting: three detection datasets (kfm12 parasite eggs, WBC white blood cells,
ChestX-Det10 chest X-ray findings). Each has 4 nodes, each owning disjoint
classes, with no labels shared and a server that holds no data. Protocol v2
(learning-rate decay, plateau budgets). Fed:Cent is the ratio of federated to
centralized mAP. Bounds are the lower end of the 95 % CI over seeds, worst
dataset.

| Mode | Scheme | Seeds | mAP50 kept ≥ | relative loss ≤ | absolute loss ≤ | mAP50-95 kept ≥ |
|---|---|---|---|---|---|---|
| Synchronous | sequential, server-scheduled turns, closing calibration | 3 | 85.7 % | 14.3 % | 0.10 mAP50 | 86.4 % |
| Asynchronous | concurrent nodes, staleness-weighted, progressive calibration | 4 | 82.6 % | 17.4 % | 0.08 mAP50 | 81.7 % |
| (reference) Synchronous | parallel FedAvg | 3 | 77.7 % | 22.3 % | 0.16 mAP50 | 75.3 % |

**Statement the data supports today, labeled exploratory:** with 95 %
confidence, the synchronous scheme keeps at least about 86 % and the
asynchronous scheme at least about 82 % of the centralized model's mAP50 on
every dataset. On average they are at or above the centralized model:
sequential sync 98–111 %; concurrent async 88–114 %.

**What it does not support yet: a confirmatory claim that the system
"systematically" stays within a fixed loss.**
1. **Post-hoc choices.** The 85 % threshold was set after the results were seen (recorded as a deviation in `PREREGISTRATION.md`). The two schemes were selected among about 30 variants on the same seeds. Selection on the evaluation seeds makes the bounds optimistic (winner's curse).
2. **Few seeds.** n = 3–4 per dataset. kfm12's sync bound (85.7 %) is barely above 85 %.
3. **Simulated concurrency.** The asynchronous results come from a virtual-clock simulation on one GPU (training times modelled from data size × machine speed). There is no real multi-machine run.
4. **The synchronous winner is sequential.** Nodes train in turn, so wall-clock time is the sum of the nodes' times. Parallel FedAvg does not reach the bound on ChestX. Both facts must be reported.
5. **The reference is the centralized model with the same partial labels,** not the full-label oracle. Against the oracle, the gap is large (WBC ≈ 58 %) for centralized as well.

## 3. Gaps for the collection, and how to close them

Ordered by importance for acceptance. "No GPU" = analysis or writing only.

| # | Gap | Collection theme | Action | Cost |
|---|---|---|---|---|
| 1 | No confirmatory evidence for the claim | trust metrics, reliability | Freeze both final configurations; state the margins in advance (e.g. sync 85 %, async 80 %); run fresh seeds (8, counterbalanced: each seed draws its own node order and machine speeds) with their centralized references; test once | ~11.5 GPU h per seed (centralized + sync + async + calibrations): ~4 days for 8 seeds |
| 2 | Fairness not measured | fairness-aware FL | Per-class and per-node retention: worst-class AP ratio, spread of per-class ratios, whether classes of small or late nodes lose more (the recency effect found on ChestX). Data already stored (`per_class_map50`) | No GPU |
| 3 | Robustness not tested | adversarial / Byzantine robustness | One faulty or malicious node (label noise, scaled or poisoned update) in sync and async. A simple defence on shared layers (norm clipping or median). Discuss that "one class, one owner" confines a malicious owner to its own class rows | ~1–2 GPU days |
| 4 | Privacy only by architecture | privacy and security | Document what leaves a node (weights; for calibration, BatchNorm statistics and its class rows), known attacks (gradient inversion, membership inference), and compatibility with secure aggregation. Optional: update clipping + noise (DP) on one dataset to show the utility cost | Writing; optional 1 GPU day |
| 5 | Security measures not presented as such | security, accountability | Report the network system's measures: admin and node tokens (hashed), HTTPS, safe weight loading (`weights_only`), idempotent uploads, lost-node handling, persistence, developer-mode-only evaluation. Source: `docs/NETWORK_FEDERATION.md` | Writing |
| 6 | Async deployment not demonstrated | real-world deployment | Port the async scheme to `fedyolo/net`: per-node final averaging, progress-driven learning rate, progressive calibration phase. Run a short multi-process demo, even on one machine | ~2–3 days of work |
| 7 | Accountability and transparency | accountability, transparency, governance | Present the audit trail (every update logged with node, staleness, version), the pre-registration with dated deviations, the negative results (`SEARCH_REPORT.md`), and a model card per scheme | Writing |
| 8 | Regulation | EU AI Act compliance | Map the system to the AI Act requirements for high-risk medical AI: data governance (Art. 10), record-keeping (Art. 12), transparency (Art. 13), human oversight (Art. 14, operator dashboards), accuracy and robustness (Art. 15). Do not claim compliance; show support | Writing |
| 9 | Data provenance and ethics | responsible AI | Not documented in the repository. Licence and citation for ChestX-Det10 and WBC; origin, consent or ethics status and availability of kfm12 (appears in-house). Ethics statement (no new human data?) | Writing; check with data owners |
| 10 | Baselines under the final protocol | — (reviewers) | Standard competitors were only run under the older 20-epoch protocol. Add FedProx and FedBuff/FedAsync under v2 on the same seeds | ~1 GPU day each |
| 11 | Explainability | explainability | Optional: detection saliency (e.g. EigenCAM) for federated vs centralized on a few images, to show the federated model attends to the same evidence | ~1 day |
| 12 | Cost and sustainability | responsible AI | Report GPU hours, energy and communication volume per scheme (sequential vs concurrent trade-off) | No GPU (from logs) |
| 13 | Code and data availability | journal requirement | Public repository with configs, seeds, tools and logs; data availability statement | Writing |

## 3b. Status of the gaps (updated 2026-11-01)

| # | Gap | Status |
|---|---|---|
| 1 | Confirmatory evidence | **Running** since 10-28 10:54. Pre-registered before its first run (`CONFIRMATORY_PREREG.md`):<br>- frozen code (`_frozen/confirm/`) and SHA-256 records (`confirm_manifest.json`);<br>- seeds 100–109; margins in fixed sequence (sync 85 → 90 %, async 80 → 85 %);<br>- analysis script `tools/confirm_analysis.py`.<br>11-01: 24 of 30 sequential and 21 of 30 concurrent runs calibrated; three datasets complete about 11-02. Frozen copy verified against the manifest (`tools/confirm_manifest.py --check` now reads code from the frozen copy). |
| 1b | Held-out dataset (user request) | DENTEX tooth enumeration (32 classes, dental panoramic X-ray, CC BY-NC-SA). Named in amendment 1 before any run; converted and split, all checks pass. Pilot done 10-28 (flat from epoch 5: 0.945, best 0.958). The automatic config step ran too early and stopped, so the configs were written only on 11-01. Budget 20 epochs (amendment 3, user decision, before any federated DENTEX run); configs hashed in `confirm_manifest_amend1.json`. 60 runs queued after the confirmation, expected about 11-04/05. |
| 2 | Fairness | **Done** on the exploratory seeds (`FAIRNESS.md`, `tools/fairness_analysis.py`). Sequential sync is fair across classes; concurrent async and parallel FedAvg lose most on rare classes (kfm12 ρ = +0.80). Interim `--confirm` run on 11-01 (8 and 7 seeds) gives the same picture; sequential kfm12 now also shows a small rare-class effect (worst class 85 %, ρ = +0.72). Final run after the confirmation. |
| 3 | Robustness | **Implemented** (`fedyolo/robust.py`, `ROBUSTNESS_PLAN.md`):<br>- 8 attacks, 3 of them detection-specific;<br>- defence `ora`, unit-tested on CPU.<br>GPU smoke tests of all three schemes passed on 10-28 (attack applied, bounds active from the 4th upload, audit log written). 117 runs queued after DENTEX, expected about 11-06/07. |
| 4 | Privacy | **Written** (`TRUSTWORTHINESS.md` §1). Sequential and async exclude secure aggregation. A DP experiment is optional, not queued. |
| 5 | Security | **Written** (`TRUSTWORTHINESS.md` §2). |
| 6 | Async deployment | **Ported** to `fedyolo/net`: server-scheduled turns (sequential), final per-node average, network calibration phase (progressive or with a barrier), defence hook in sync and async rooms. End-to-end CPU test: see `docs/NETWORK_FEDERATION.md`. |
| 7 | Accountability | **Written** (`TRUSTWORTHINESS.md` §5). Model cards still to produce. |
| 8 | EU AI Act | **Written** (`TRUSTWORTHINESS.md` §6), with a NIST AI RMF mapping. |
| 9 | Data provenance | **Done** (`TRUSTWORTHINESS.md` §7): kfm12 is the author's own (public subset doi:10.17632/6bjcxw8dx2.1, no ethics needed); WBC is the GitHub *White-Blood-Cell-Detection-Dataset* (boxes on PBC, CC BY 4.0). |
| 10 | v2 baselines | **Queued** after robustness: FedProx, FedBuff, CA2FL (`bl_*`, seeds 0–2, same post-processing). |
| 11 | Explainability | Not started (optional). |
| 12 | Cost | **Done** (`COST.md`, `tools/cost_report.py`). GPU power is now logged to `runs/variants/_power.csv`, so the confirmation's energy will be measured. |
| 13 | Availability | **Drafted** (`TRUSTWORTHINESS.md` §8). |

## 4. Suggested framing

*Trustworthy evaluation of federated object detection under disjoint label
ownership.*

**Contributions**
1. A federated detection system for nodes that each own different classes, with no label or data sharing and a server that holds no data. It includes a secured client–server implementation.
2. A pre-registered non-inferiority evaluation with confidence bounds, the same post-processing on both sides, node-order control and fresh-seed confirmation, with every deviation and negative result disclosed. This is the "trust metric" contribution.
3. Bounds on the accuracy cost of federation for synchronous and asynchronous operation. These become confirmatory after gap 1.
4. Findings useful to practitioners:
   - the training recipe dominates the aggregation rule;
   - node order matters in async;
   - sequential training keeps centralized accuracy where parallel averaging does not;
   - an asynchronous, barrier-free calibration step.

**Minimum for a credible submission:** gaps 1, 2, 5, 7, 9 and 13, plus a
discussion of 3, 4 and 8.

**Strong submission:** also 3 (experiment), 6 and 10.
