# Study runs: run-level records

Written by `tools/export_evidence.py` from `runs/variants/` (not tracked: it holds checkpoints).
Use cases KFM12, WBC and DENTEX, seeds 100-109. Per seed: the calibration stages of the centralized
reference and of both schemes (`centralized.json`, `cwt.json`, `fedasync.json`; stages used in the article:
`bn+head` for the centralized reference and CWT, `acal_refresh` for FedAsync), each run's summary,
FedAsync's virtual-clock timeline, and the images each node trained on (`subsets/`, KFM12 and WBC;
DENTEX trains on every training image). The same subset is used by every arm of a seed.

Protocol: `analysis/EVALUATION_PROTOCOL.md`. Intervals: `_frozen/confirm/tools/confirm_analysis.py`
(frozen before the first run); levels, seed-level table and sensitivity of the lower bounds:
`analysis/STUDY_SENSITIVITY.md`, `analysis/study_seed_level.csv` (`tools/study_sensitivity.py`).
File hashes of the frozen code and configurations: `analysis/confirm_manifest.json`,
`analysis/confirm_manifest_amend1.json` (DENTEX).
