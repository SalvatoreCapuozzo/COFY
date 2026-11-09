# Protecting the federation from a malicious node: design and study plan

**Date:** 2026-10-28. **Code:** `fedyolo/robust.py` (attacks on uploads, defence),
`tools/make_attack_data.py` (data attacks), `tools/make_robust_configs.py` (study
configs), `tools/robust_eval.py` (measures, `analysis/ROBUSTNESS.md`).
Everything is opt-in (`federation.robust`); the defaults leave every run unchanged.

## 1. Position in the literature

General federated learning has a large body of work on poisoning and defences:
- robust aggregation: Krum, coordinate-wise median and trimmed mean, norm bounding;
- defences that need data on the server (FLTrust);
- feedback from clients (BaFFLe, arXiv:2011.02167);
- surveys: arXiv:2306.03397.

Work on federated *object detection* under attack is sparse. Our searches found poisoning studies of federated driving models, but on regression tasks (arXiv:2405.01073). We found no defence designed for federated detection. We also found none for the setting of this system, where each class has exactly one owner.

**Why the usual defences do not carry over:**
1. **Class rows.** Each class's output row has a single owner. There is no second opinion to vote with, so median, Krum and trimmed mean cannot protect the class rows.
2. **Shared layers.** Under disjoint classes the honest updates of the shared layers differ by design. Each node optimizes a different objective, so a "majority direction" is weaker evidence than under IID data.
3. **One update at a time.** In the sequential and concurrent schemes the server sees one update at a time. Methods that compare simultaneous updates do not apply.
4. **No server data.** The server holds no data (a rule of the system), so methods that validate on server data are excluded.
5. **Detection-specific attacks exist:**
   - corrupted boxes rather than flipped labels;
   - backdoors that create phantom objects (object generation) or make real ones vanish;
   - BatchNorm statistics that every class depends on.

## 2. Threat model

- One node of four is malicious or faulty and does not collude.
- It controls its data, its training and its upload, and knows the protocol and the defence.
- The server is honest. Transport and authentication are already secured (`fedyolo/net`: tokens, HTTPS, safe tensor decoding).

## 3. Attacks implemented

| Attack | Kind | What the attacker does | Detection-specific? |
|---|---|---|---|
| `noise` | faulty node | sends Gaussian noise, 3× its honest update's size, tensor by tensor | no |
| `signflip` | untargeted, stealthy | sends the opposite of its honest update, same size | no |
| `scale` | model replacement | boosts its update ×10 to dominate the merge | no |
| `bnpoison` | untargeted | corrupts the BatchNorm running statistics (mean +3σ, variance ×9) | yes: BatchNorm is shared by every class's detector |
| `rowhijack` | targeted | claims every class and makes other owners' classes fire everywhere | yes: per-class rows |
| `weightinflate` | untargeted (parallel FedAvg) | reports 100× its number of images | no |
| `labels` | data poisoning | drops half its boxes, moves and resizes the rest | yes |
| `backdoor` | targeted | checkerboard trigger in 30 % of its images, labelled as its class (object generation) | yes |

## 4. Defence: ownership-aware robust aggregation (`ora`)

It runs on the server and needs no data there.
1. **Ownership from the registry.**
   - A node's class rows are taken only for the classes the server registered for it, never for the classes the upload claims.
   - This is always on, as in `fedyolo/net`.
   - The `_noenf` runs switch it off, to show what it prevents.
2. **Norm bounds per part of the update.**
   - The parts are the shared weights, the BatchNorm statistics and each owned class row.
   - A part larger than κ = 2× the median size of that part in the other nodes' recent updates is scaled down to the bound.
   - The reference needs updates from at least 3 other nodes. With one attacker among four, the median is then an honest node's.
   - Class rows have no peers, so their bound uses every node's row sizes.
   - This works one update at a time, so it fits the sequential and concurrent schemes.
3. **Parallel FedAvg only:** coordinate-wise trimmed mean of the shared deltas, dropping the highest and lowest value. It is unweighted, because image counts reported by clients are not trusted.
4. **Audit trail.** Every decision (sizes, bound, scaling, claimed extra classes) is logged in `robust_log.jsonl`, for accountability.

**Known limits, to be reported:**
- **Early updates.** The first updates, before three other nodes have reported, pass unbounded.
- **Small attacks pass.** A same-size sign flip or a backdoor within the norm bound passes the bounds. Only their effect after merging can be measured.
- **Self-harm is invisible.** An attack confined to the attacker's own classes harms only those classes, and no other party holds data to check them.

## 5. Study

- **Setting:** WBC, the fastest dataset; protocol v2; seeds 0–2. The attacker is node seed mod 4: A, B, C.
- **Schemes:**
  - sequential sync (`v2_async`);
  - concurrent async (`v2_vclock`);
  - parallel FedAvg (`v2_sync`) for the attacks that matter there.
- **Each attack:** with and without the defence.
- **The defence on clean runs,** to measure its cost.
- **Rowhijack:** with and without ownership enforcement.
- **Clean undefended runs:** the existing v2 runs (not retrained).
- **Size:** 40 variants × 3 seeds ≈ 117 runs ≈ 53 GPU hours, after the confirmatory runs.
- **Measures (`tools/robust_eval.py`):**
  - mAP50 and Fed:Cent;
  - mAP50 on the attacker's classes and on everyone else's;
  - backdoor attack success rate on a triggered evaluation set, with the clean runs as the false-alarm baseline;
  - the share of uploads the defence scaled, attacker vs honest (its false-positive rate).

**Calibration as an attack surface (not in this study):** the closing calibration pools BatchNorm statistics and retrains class rows. A robust pooling (median instead of a count-weighted mean) is the natural extension.
