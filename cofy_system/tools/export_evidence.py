#!/usr/bin/env python3
"""Copy the run-level records of the study (seeds 100-109) into evidence/study/, a small folder
that is tracked and archived with the code release (runs/ is not: it holds checkpoints).

Per use case and seed (100-109):
  centralized.json   calibration stages of the centralized reference (mAP50, mAP50-95, per class)
  cwt.json           the same for CWT (registered stage: bn+head)
  fedasync.json      the same for FedAsync (registered stage: acal_refresh)
  *_summary.json     each run's own summary (last global version, before calibration)
  fedasync_schedule.json   FedAsync's timeline on the virtual clock (machine speeds, every update)
  subsets/node_X.txt the images each node trained on (file names), KFM12 and WBC only:
                     DENTEX trains on every training image

    python tools/export_evidence.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "runs/variants"
OUT = ROOT / "evidence/study"
DATASETS = ["kfm12", "wbc", "dentex"]
SEEDS = range(100, 110)
CALIB = {"centralized.json": "calib/v2_centralized/{ds}/s{s}/centralized.json",
         "cwt.json": "calib/confirm_v2sync/{ds}/s{s}/async.json",
         "fedasync.json": "calib/confirm_v2async/{ds}/s{s}/async.json"}
RUNS = {"centralized": "v2_central", "cwt": "confirm_v2sync", "fedasync": "confirm_v2async"}


def main():
    n = 0
    for ds in DATASETS:
        for s in SEEDS:
            d = OUT / ds / f"s{s}"
            d.mkdir(parents=True, exist_ok=True)
            for name, src in CALIB.items():
                p = VAR / src.format(ds=ds, s=s)
                if p.exists():
                    shutil.copyfile(p, d / name)
                    n += 1
            for arm, variant in RUNS.items():
                p = VAR / f"{variant}/{ds}/s{s}/summary.json"
                if p.exists():
                    shutil.copyfile(p, d / f"{arm}_summary.json")
                    n += 1
            sched = next((VAR / f"confirm_v2async/{ds}/s{s}").glob("*/virtual_schedule.json"), None)
            if sched:
                shutil.copyfile(sched, d / "fedasync_schedule.json")
                n += 1
            mans = sorted((VAR / f"confirm_v2sync/{ds}/s{s}").glob("*/balanced_subsets/*_train.txt"))
            for m in mans:   # the same subset for every arm of the seed (drawn from the seed); names only
                (d / "subsets").mkdir(exist_ok=True)
                names = [Path(line).name for line in m.read_text().splitlines() if line.strip()]
                (d / "subsets" / m.name.replace("_train", "")).write_text("\n".join(names) + "\n")
                n += 1
    (OUT / "README.md").write_text(
        "# Study runs: run-level records\n\n"
        "Written by `tools/export_evidence.py` from `runs/variants/` (not tracked: it holds checkpoints).\n"
        "Use cases KFM12, WBC and DENTEX, seeds 100-109. Per seed: the calibration stages of the centralized\n"
        "reference and of both schemes (`centralized.json`, `cwt.json`, `fedasync.json`; stages used in the article:\n"
        "`bn+head` for the centralized reference and CWT, `acal_refresh` for FedAsync), each run's summary,\n"
        "FedAsync's virtual-clock timeline, and the images each node trained on (`subsets/`, KFM12 and WBC;\n"
        "DENTEX trains on every training image). The same subset is used by every arm of a seed.\n\n"
        "Protocol: `analysis/EVALUATION_PROTOCOL.md`. Intervals: `_frozen/confirm/tools/confirm_analysis.py`\n"
        "(frozen before the first run); levels, seed-level table and sensitivity of the lower bounds:\n"
        "`analysis/STUDY_SENSITIVITY.md`, `analysis/study_seed_level.csv` (`tools/study_sensitivity.py`).\n"
        "File hashes of the frozen code and configurations: `analysis/confirm_manifest.json`,\n"
        "`analysis/confirm_manifest_amend1.json` (DENTEX).\n")
    print(f"{n} files written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
