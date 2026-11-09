#!/usr/bin/env python3
"""The pre-registered analysis (analysis/PREREGISTRATION.md), fixed in code
before any study result existed.

Primary: per seed r = mAP50(fed)/mAP50(cent), both at the final checkpoint on
the dataset's held-out eval set. Over the seeds, L = mean - t(0.975, n-1)*sd/sqrt(n).
PASS iff L >= 0.90 with the full n = 5 seeds. The hypothesis holds only if all
six tests (3 datasets x {sync, async}) pass.

    python3 tools/analyze_study.py            # prints the table, writes runs/study/analysis.json
"""
import json, math, statistics as st
from pathlib import Path

DATASETS = ["kfm12", "wbc", "chestx"]
SEEDS, MARGIN, N_REQUIRED = range(5), 0.90, 5
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

def ci_low(v):
    if len(v) < 2: return None
    return st.mean(v) - T975[len(v) - 1] * st.stdev(v) / math.sqrt(len(v))

def main():
    out, verdicts = {}, []
    for ds in DATASETS:
        runs = {}
        for s in SEEDS:
            p = Path(f"runs/study/{ds}/s{s}/summary.json")
            if p.exists():
                d = json.loads(p.read_text())
                if all(k in d and "map50" in d[k] for k in ("federated_sync", "federated_async", "centralized", "oracle")):
                    runs[s] = {k: d[k]["map50"] for k in ("federated_sync", "federated_async", "centralized", "oracle")}
        out[ds] = {"seeds_done": sorted(runs), "per_seed": runs}
        for mode in ("sync", "async"):
            fc = [r[f"federated_{mode}"] / r["centralized"] for r in runs.values()]
            fo = [r[f"federated_{mode}"] / r["oracle"] for r in runs.values()]
            lo = ci_low(fc)
            if len(fc) < N_REQUIRED:
                status = f"INCOMPLETE ({len(fc)}/{N_REQUIRED} seeds)"
            else:
                status = "PASS" if lo >= MARGIN else "FAIL"
                verdicts.append(status == "PASS")
            out[ds][mode] = {"fed_cent": fc, "fed_oracle": fo,
                             "fed_cent_mean": st.mean(fc) if fc else None,
                             "fed_cent_sd": st.stdev(fc) if len(fc) > 1 else None,
                             "fed_cent_ci_low": lo,
                             "fed_oracle_mean": st.mean(fo) if fo else None,
                             "fed_oracle_ci_low": ci_low(fo),
                             "status": status}
            pct = lambda x: "  n/a " if x is None else f"{100*x:6.1f}%"
            print(f"{ds:7s} {mode:5s} n={len(fc)}  Fed:Cent mean {pct(out[ds][mode]['fed_cent_mean'])}"
                  f"  CI-low {pct(lo)}  -> {status:24s}  Fed:Oracle mean {pct(out[ds][mode]['fed_oracle_mean'])}")
    complete = len(verdicts) == 2 * len(DATASETS)
    out["hypothesis"] = ("SUPPORTED" if complete and all(verdicts) else
                         "NOT SUPPORTED" if complete else "INCOMPLETE")
    print(f"\nPre-registered hypothesis (all 6 tests pass): {out['hypothesis']}")
    Path("runs/study").mkdir(parents=True, exist_ok=True)
    Path("runs/study/analysis.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
