# Defences for the attacks COFY does not withstand: state of the art and candidate procedures

2026-11-09. Exploratory, after the registered study. Results that motivate it: `ROBUSTNESS.md`
and `FOLLOWUP_PLAN.md` section 12.

## 1. What COFY is weak against

| # | Attack | What happens | Why the current defence misses it |
|---|---|---|---|
| W1 | Backdoor in the attacker's own class (BadDet object-generation attack) | attack success 100 % in all 18 runs, with or without the defence; accuracy intact | the update is honest-sized and touches rows only the attacker trains |
| W2 | Sign flip of the attacker's own updates | its class fires everywhere and fills the shared 300-detection budget: every scheme collapses | honest-sized update; the rows have no other contributor, so the reversal accumulates |
| W3 | Persistent direction in the concurrent scheme (BatchNorm poisoning, seed 2) | the attacker sent its last three uploads in a row: collapse with or without the defence | per-update bounds cannot stop a direction that persists |
| W4 | Any attack confined to the attacker's own classes (noise, label flips, W1) | only the attacker's classes are harmed | no other party has data for those classes |

The per-class detection budget already confines W2 to the attacker's classes (82 / 91 / 96 % of
centralized).

## 2. Why the standard defences do not transfer

- **Defences that compare clients.** Krum, median or trimmed mean, clustering (FLAME), update
  analysis (DeepSight), gradient forensics for detection (STDLens) and history consistency
  (FLDetector) all compare updates of the same parameters from several clients. Under single
  ownership a class's rows have one contributor, so there is nothing to compare them with. Outlier
  detection already degrades when honest clients hold unusual data (BoBa).
- **Defences that need data on the server.** FLTrust, Zeno++, AFLGuard and BackdoorIndicator need
  data on the server, which violates R2.
- **Defences that bound magnitude.** Clipping plus noise (CRFL, FLAME) bounds the size of an
  update. W1 and W2 are honest-sized.

## 3. Candidate procedures that keep R1-R3

| # | Procedure | Source | Targets | Where it runs | Cost | How to test with what we have |
|---|---|---|---|---|---|---|
| P1 | **Robust BatchNorm re-estimation.** Calibration already discards the running statistics from training and re-estimates them from the nodes' images. Pool them with a coordinate-wise median instead of a count-weighted mean, and score the robustness models after calibration. BatchNorm statistics are shared by all nodes, so, unlike class rows, they can be compared. | BN-statistics attacks: Guerraoui et al. 2024; BN-statistics inspection: BNGuard 2025 | W3 (and BN poisoning in general) | server, at calibration | none at training; minutes per model | the collapsed concurrent seed is kept: see section 4 |
| P2 | **Run-time input screening at each node.** Blend each predicted box with other content and flag detections whose class does not change (low entropy). A node can skip classes it owns. | Detector Cleanse (BadDet, ECCVW 2022): OGA FAR 2.7-6.3 %, FRR 2.4-9.8 % on Faster R-CNN; TRACE (CVPR 2025): transformation consistency, black-box | W1 | each node, at inference, its own clean boxes only | N extra forward passes per suspicious detection | the 18 backdoor + 18 clean final models are kept; triggered and clean evaluation sets exist; inference only (~2-3 GPU h) |
| P3 | **Peer behavioural validation with per-class rollback.** Each node runs every new global version (sequential) or every k-th (concurrent) on its own images and reports per-class detection rates, which need no labels. A class that starts firing on most images of nodes that do not own it is flagged. The server keeps its last accepted rows and quarantines its owner's row updates. | BaFFLe (ICDCS 2021): clients validate the global model and vote; CrowdGuard (NDSS 2024): clients analyse model behaviour on their data and the server aggregates the votes | W2, row-level floods (not W1: the trigger is absent from honest images) | nodes (validation) + server (decision) | one inference pass per node and version; per-class counts are disclosed | re-run defended sign flip, 3 schemes x seeds 0-2, plus the clean twins for false alarms (~16 GPU h) |
| P4 | **History-aware bounds in the concurrent scheme.** (a) Bound each node's momentum, i.e. its cumulative contribution over a window, instead of each update. (b) Limit consecutive merges: a node's upload waits while that node made the last m merges, until another node contributes. | worker momentum + centered clipping (Karimireddy et al., ICML 2021); buffered asynchronous robust aggregation (BASGD, ICML 2021) | W3, W2 accumulation | server | none | concurrent BN poisoning and sign flip, seeds 0-2, plus clean (~8 GPU h) |
| P5 | **Model scanning by trigger inversion.** Honest nodes invert, on their own images, a minimal trigger for every class; a class whose trigger is abnormally easy to invert is flagged. It detects but cannot repair, because only the owner has data for the class; the response is to exclude the class from the released model and alert the operators. | Django (NeurIPS 2023); ODSCAN (IEEE S&P 2024, ROC-AUC > 0.9) | W1 | nodes | engineering: port to the YOLO26 head (days) | the 36 kept models |

Not recommended:
- **FLIP** (ICLR 2023): benign clients invert triggers and train against them. In COFY that means
  honest nodes suppressing another node's class, which breaks R1/R3 unless the rules change.
- **Fine-pruning and module reset with clean fine-tuning** (arXiv 2409.16057): they need clean data
  of the target class, which only the attacker has.

## 4. Check of P1 (pilot 2026-11-09, full 2026-11-10)

Model `rb_conc_bnpoison_ora_budget/wbc/s2` (the concurrent seed that collapsed; WBC, mAP50):

| Scoring | mAP50 | vs centralized (0.520) |
|---|---|---|
| final model as it is (as in the robustness study) | 0.001 | 0 % |
| BatchNorm re-estimated, every node honest in calibration, count-weighted mean (current rule) | 0.626 | 120 % |
| attacker also poisons its calibration statistics (mean +3 sd, variance x9), current mean | 0.000 | 0 % |
| attacker poisons its calibration statistics, coordinate-wise median of the four nodes | 0.614 | 118 % |
| every node honest, median (cost of the median) | 0.615 | 118 % |

Reading:
- The collapse lies entirely in the running statistics, which training does not use and
  calibration recomputes. The registered post-processing already undoes it, unless the attacker
  attacks the calibration too.
- In that case the current mean collapses again, and the median holds, at a cost of 0.011 mAP50
  on this model.
- Confirmed on 21 models (2026-11-10; `DEFENCE_RESULTS.md`): the four BatchNorm-poisoning settings,
  repeated with their final model kept, and the defended clean runs, seeds 0-2.
  - The saved final model averages the versions right after each node's last update, so it keeps the
    attacker's last poisoned statistics: uncalibrated, mAP50 0.000-0.001 in all nine runs without
    the defence or in the concurrent scheme, 0.22-0.57 in the defended sequential scheme. The
    robustness table scores the last global version instead (the server's last evaluation).
  - Registered mean, every node honest in calibration: all twelve poisoned models back to mAP50
    0.43-0.63 (74-120 % of centralized; defended clean runs 94-117 %).
  - Attacker also poisons its calibration statistics: the mean collapses all 21 models, clean ones
    included (0.000); the median holds in all 21 (0.44-0.65, 75-120 %).
  - Cost of the median with every node honest: at most 0.006 mAP50.
- The progressive calibration releases a model after every contribution, so the median needs at
  least three contributions; before that, keep the stored statistics.

Script: `tools/defence_eval.py` (`baseline:bnpool` in the queue).

## 5. What no procedure fixes

A node can always sabotage or backdoor its own classes during training. With one owner per class,
no evidence available during training separates this from honest training. Procedures can only:
- contain it (the per-class budget);
- detect its effect on other nodes' images (P3);
- screen inputs at inference (P2);
- scan the model (P5).

## 6. Proposed order

1. P1: minutes, kept model.
2. P2: inference only on the kept models.
3. P4: server-side change and short re-runs.
4. P3: protocol change and re-runs.
5. P5: optional, the largest engineering effort.

P2-P4 need about 25 GPU hours and 3-4 days of work, which fits before 15 December. All results
would be exploratory and reported as such.

## References (check the DOIs with Crossref before citing)

- Chan et al., BadDet: Backdoor attacks on object detection (Detector Cleanse), ECCV 2022 Workshops. https://arxiv.org/abs/2205.14497
- Zhang et al., Test-time backdoor detection for object detection models (TRACE), CVPR 2025. https://arxiv.org/abs/2503.15293
- Shen et al., Django: Detecting Trojans in object detection models via Gaussian focus calibration, NeurIPS 2023. https://papers.nips.cc/paper_files/paper/2023/hash/a102d6cb996be3482c059c1e18bbe523-Abstract-Conference.html
- Cheng et al., ODSCAN: Backdoor scanning for object detection models, IEEE S&P 2024. https://www.cs.purdue.edu/homes/an93/static/papers/SP24_Cheng.pdf
- Andreina et al., BaFFLe: Backdoor detection via feedback-based federated learning, ICDCS 2021. https://arxiv.org/abs/2011.02167
- Rieger et al., CrowdGuard: Federated backdoor detection in federated learning, NDSS 2024. https://arxiv.org/abs/2210.07714
- Zhang et al., FLIP: A provable defense framework for backdoor mitigation in federated learning, ICLR 2023. https://arxiv.org/abs/2210.12873
- Karimireddy, He, Jaggi, Learning from history for Byzantine robust optimization, ICML 2021. https://proceedings.mlr.press/v139/karimireddy21a.html
- Yang, Li, Buffered asynchronous SGD for Byzantine learning (BASGD), ICML 2021 / JMLR 2023. https://arxiv.org/abs/2003.00937
- Zhang et al., FLDetector, KDD 2022, doi:10.1145/3534678.3539231
- Nguyen et al., FLAME: Taming backdoors in federated learning, USENIX Security 2022. https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen
- Xie et al., CRFL: Certifiably robust federated learning against backdoor attacks, ICML 2021. https://arxiv.org/abs/2106.08283
- Chow et al., STDLens: Model hijacking-resilient federated learning for object detection, CVPR 2023. https://arxiv.org/abs/2303.11511
- Guerraoui et al., Overcoming the challenges of batch normalization in federated learning, 2024. https://arxiv.org/abs/2405.14670
- On the out-of-distribution backdoor attack for federated learning (BNGuard), 2025. https://arxiv.org/abs/2509.13219
- BoBa: Boosting backdoor detection through data distribution inference in federated learning, 2024. https://arxiv.org/abs/2407.09658
- Towards robust object detection: identifying and removing backdoors via module inconsistency analysis, 2024. https://arxiv.org/abs/2409.16057

## 7. P4 and P3 as implemented (2026-11-10)

Both are opt-in (`federation.robust`, defaults change nothing) and run as robustness variants that keep
their final model; `python tools/defence_eval.py --report` adds their tables to `DEFENCE_RESULTS.md`.

**P4, consecutive merges bounded jointly** (`history_bound: run`, `cofy/robust.py`, asynchronous
schemes with the defence). A node's consecutive merges, with no other node's merge in between, are
bounded jointly for the shared weights and the BatchNorm statistics: their cumulative change must stay
within the bound of one update (the new update is scaled by the largest factor that keeps it there).
Changes from section 3:
- no window or momentum over interleaved updates: honest nodes also pull the BatchNorm statistics towards
  their own data at every update, so a persistent direction alone does not separate the attacker;
- no waiting rule: holding a node's upload stalls fast honest nodes, and at the end of training it must
  drop or merge the slowest node's last updates; the joint bound gives the same protection without it;
- class rows are left out: they affect only their owner's classes (P3 covers them).
Runs: `rb_conc_bnpoison_ora_p4_budget` and `rb_conc_clean_ora_p4_budget` (cost), seeds 0-2.

**P3, peer validation with per-class rollback** (`peer_check`, `cofy/peercheck.py`, all schemes).
After every new global version, each node runs the model on 32 of its own training images (no labels)
and reports one number per class: its coverage, the share of the head's locations where the class
scores >= 0.25. A non-owner votes against a class with coverage >= 2 % and >= 10 x the median of the
other classes on its images; a majority of the non-owners flags the class; two flagged checks in a row
roll its rows back to the last accepted version and quarantine its owner's row updates (also in the
stored versions the final model averages). The node named as attacker lies in its reports (every class
it does not own at full coverage), also in the clean twins, so these test the vote.
Pilot on the kept final models, before any P3 run:
- the share of images with a detection does not work: in clean models, four of the seven classes are
  detected on 80-100 % of the images of nodes that do not own them (those cells do appear there,
  unlabelled);
- coverage separates: at most 0.1 % in the 18 clean and backdoored models, 16-90 % for the sign-flipped
  class (seed 0, three schemes); the rule gives no wrong vote and all nine right ones there;
- in a short smoke run every class rose together to 0.2-3.6 % in one early version: hence the ratio to
  the other classes and the patience of 2.
P3 does not target the backdoor (its class does not fire without the trigger: coverage <= 0.1 %).
Runs: `rb_{seq,conc,par}_signflip_ora_p3_budget` and `rb_{seq,conc,par}_clean_ora_p3_budget`, seeds 0-2.

**Rules check** (`CUDA_VISIBLE_DEVICES= python tools/check_rules.py`, CPU, ~25 s): drives the real
defence, merge, P4 and P3 code with adversarial uploads (every node changes and claims every class's
rows; one node uploads several times in a row) and scripted coverage reports with a lying attacker.
It asserts R3 (a class's rows change only on its owner's merges; P4 changes no row; P3 only resets a
class's rows to values they had earlier and touches nothing else; P1 changes BatchNorm statistics only),
the vote (the liar alone rolls nothing back; the flooding class is rolled back after two checks and
stays frozen), R2 (P3's node step opens only its data.yaml and its images) and the defaults (every
candidate off). 12/12 passed on 2026-11-10.
One point of interpretation: P3 lets a majority of the non-owners veto an owner's row updates (the
rows then keep the owner's last accepted values); nobody but the owner ever writes them. The existing
defence already lets the server scale an owner's row update. P3 also discloses one coverage number
per class per check to the server.

**P4 result (2026-11-10, WBC seeds 0-2, `DEFENCE_RESULTS.md`):** it does not stop BatchNorm poisoning in
FedAsync. The joint bound scaled two of the attacker's uploads further, in seed 2 (the seed whose attacker
sent its last uploads in a row), and that seed's last global version collapsed again (mAP50 0.009): one
bounded update of the statistics suffices when no honest update follows. Last version over seeds: 69 % of
centralized, against 70 % without P4; clean runs: 115 % against 109 % (no cost). The calibration (P1) is
what undoes this attack.

**P3 result (2026-11-11, WBC seeds 0-2, `DEFENCE_RESULTS.md`):** detection is exact. The attacker's classes
were rolled back in all nine sign-flip runs (checks 3-17; node C's second class at check 36 in one
FedAsync seed) and no honest class in the nine clean runs, although the named node lied in every report;
no cost on clean runs (98/110/115 % against 100/109/103 %). Repair depends on the scheme: FedAvg 91 % of
centralized (0 % without P3; other classes mAP50 0.59-0.64, as clean), CWT 31 % and FedAsync 17 % (0 % and
1 %). In CWT and FedAsync the frozen rows fire again (3-29 % of the locations at the end) because the
attacker's reversed shared updates are still merged, bounded only in size; FedAvg's trimmed mean discards
them. Next step: on quarantine, also discard the flagged node's shared and BatchNorm updates.

One correction to the P3 description: the lie as run (the attacker reports every class it does not own at
coverage 1.0) never casts a vote, because such a report fails the ratio rule (the class is not 10x the median
of the others, which are all 1.0). The clean twins therefore test the ratio rule, not the majority. A lie aimed
at one class is one vote of the three non-owners and cannot reach a majority alone; `tools/check_rules.py`
now tests that case with scripted reports.

**P3b, rollback and exclusion (`peer_exclude`, implemented 2026-11-11).** On a rollback, the owner of the
rolled-back class is excluded from then on:
- asynchronous schemes (CWT, FedAsync): the server discards its upload on arrival, with no merge and no new
  version (a version that changed nothing would make the honest nodes' updates look staler and down-weight
  them); its slot in the per-node average keeps the current global model;
- parallel sync (FedAvg): it is left out of the round (trimmed mean over the three others = their median), and
  its classes keep the rows of the previous version, since the merge would otherwise fill them from another
  node (R3);
- its coverage reports no longer count in the vote.
Rows still only ever hold values their owner produced (R3); the server holds no data (R2). Checked by
`tools/check_rules.py` (8 new checks through the real asynchronous `push()` and the sync merge steps; the
vote check was mutation-tested). Runs: `rb_{seq,conc,par}_signflip_ora_p3x_budget`, WBC seeds 0-2, compared
with P3 in `tools/defence_eval.py --report`. Clean twins are not re-run: without a rollback there is no
exclusion, and P3 rolled back nothing in its nine clean runs.
