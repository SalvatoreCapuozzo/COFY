#!/usr/bin/env python3
"""Pre-registered confirmatory analysis (analysis/CONFIRMATORY_PREREG.md).

Written before the first confirmatory run and frozen: its SHA-256 is recorded in
analysis/confirm_manifest.json. Any later change is a deviation and must be dated
in the pre-registration.

Primary endpoint, per scheme and dataset: Fed:Cent = federated mAP50 / centralized
mAP50 x 100, paired by seed, both after the same post-processing:
  sync   confirm_v2sync  (sequential, server-scheduled turns), closing calibration (bn+head)
  async  confirm_v2async (concurrent, virtual clock), progressive calibration with refresh (acal_refresh)
  reference: v2 centralized of the same seed with the calibration round (bn+head).
CI-low = lower end of the two-sided 95 % t-interval of the mean Fed:Cent over the
N seeds. A scheme passes a margin when CI-low >= margin on every dataset (an
intersection-union test: no multiplicity correction). Margins are tested in a
fixed sequence (85 then 90 for sync, 80 then 85 for async), each at the same
level; the claim is the last margin passed before the first failure. The test is
done once, with all N seeds finished on every dataset; no seeds are added.

Held-out datasets (analysis/confirm_heldout.json, named before their first run)
are tested the same way and reported separately.

    python tools/confirm_analysis.py            # status, or the result once complete
    python tools/confirm_analysis.py --write    # also write analysis/CONFIRMATORY_RESULTS.md
"""
import json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "runs/variants"
SEEDS = list(range(100, 110))
DATASETS = ["kfm12", "wbc", "chestx"]
SCHEMES = {   # scheme: (variant, plan mode, calibration stage, margins in test order)
    "sync": ("confirm_v2sync", "async", "bn+head", [85.0, 90.0]),
    "async": ("confirm_v2async", "async", "acal_refresh", [80.0, 85.0]),
}
REF_STAGE = "bn+head"
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def heldout():
    d = load(ROOT / "analysis/confirm_heldout.json") or {}
    return d.get("datasets", [])


def pairs(scheme, ds, metric="map50"):
    """{seed: (federated, centralized)} for the finished seeds."""
    src, mode, stage, _ = SCHEMES[scheme]
    out = {}
    for s in SEEDS:
        f = (load(VAR / f"calib/{src}/{ds}/s{s}/{mode}.json") or {}).get(stage)
        c = (load(VAR / f"calib/v2_centralized/{ds}/s{s}/centralized.json") or {}).get(REF_STAGE)
        if f and c and c.get(metric):
            out[s] = (f[metric], c[metric])
    return out


def summary(scheme, ds, metric="map50"):
    p = pairs(scheme, ds, metric)
    n = len(p)
    if n < 2:
        return {"n": n}
    r = [f / c * 100 for f, c in p.values()]
    loss = [c - f for f, c in p.values()]
    m, sd = statistics.mean(r), statistics.stdev(r)
    lm, lsd = statistics.mean(loss), statistics.stdev(loss)
    h = T975[n - 1] / n ** 0.5
    return {"n": n, "ratios": r, "mean": m, "sd": sd, "lo": m - h * sd, "hi": m + h * sd,
            "loss_mean": lm, "loss_hi": lm + h * lsd,
            "fed": statistics.mean(f for f, _ in p.values()), "cent": statistics.mean(c for _, c in p.values())}


def verdict(scheme, datasets):
    """(complete, claim margin or None, {margin: pass}) for one scheme over `datasets`."""
    st = {ds: summary(scheme, ds) for ds in datasets}
    complete = all(x["n"] == len(SEEDS) for x in st.values())
    passed, claim = {}, None
    for m in SCHEMES[scheme][3]:
        passed[m] = complete and all(x["lo"] >= m for x in st.values())
        if not passed[m]:
            break
        claim = m
    return complete, claim, passed, st


def headline():
    """Status lines for tools/results_table.py."""
    lines = []
    for label, dsets in [("confirmation", DATASETS)] + ([("held-out", heldout())] if heldout() else []):
        for scheme in SCHEMES:
            complete, claim, passed, st = verdict(scheme, dsets)
            done = sum(x["n"] for x in st.values())
            vals = "; ".join(f"{ds} {x['mean']:.1f}% (CI-low {x['lo']:.1f}%, n={x['n']})"
                             for ds, x in st.items() if x["n"] >= 2)
            if complete:
                res = (f"PASS at {claim:.0f}%" if claim else f"FAIL at {SCHEMES[scheme][3][0]:.0f}%") + \
                      "".join(f", {m:.0f}% {'pass' if ok else 'fail'}" for m, ok in passed.items())
            else:
                res = f"in progress ({done}/{len(SEEDS) * len(dsets)} runs calibrated; tested once when complete)"
            lines.append(f"{label.upper()} {scheme.upper()} ({SCHEMES[scheme][0]}): {res}" + (f" -- {vals}" if vals else ""))
    return lines


def report():
    out = [f"# Confirmatory results", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `tools/confirm_analysis.py` (pre-registered in "
           "`analysis/CONFIRMATORY_PREREG.md`). Seeds " + f"{SEEDS[0]}-{SEEDS[-1]} (n = {len(SEEDS)}).", ""]
    for label, dsets in [("Confirmation datasets", DATASETS)] + ([("Held-out datasets", heldout())] if heldout() else []):
        out += [f"## {label}", ""]
        for scheme in SCHEMES:
            complete, claim, passed, st = verdict(scheme, dsets)
            src, _, stage, margins = SCHEMES[scheme]
            out += [f"### {scheme} -- `{src}` ({stage})", "",
                    ("**Result: " + (f"passes {claim:.0f} %" if claim else f"fails {margins[0]:.0f} %") + "** "
                     + "(fixed sequence " + ", ".join(f"{m:.0f} %: {'pass' if ok else 'fail'}" for m, ok in passed.items())
                     + ")") if complete else "**Incomplete: not tested yet.**", "",
                    "| dataset | n | Fed:Cent mean | sd | 95 % CI | mAP50 fed / cent | abs. loss mean (upper 95 %) "
                    "| mAP50-95 Fed:Cent (CI-low) |", "|---|---|---|---|---|---|---|---|"]
            for ds, x in st.items():
                y = summary(scheme, ds, "map50-95")
                if x["n"] < 2:
                    out.append(f"| {ds} | {x['n']} | | | | | | |")
                    continue
                out.append(f"| {ds} | {x['n']} | {x['mean']:.1f} % | {x['sd']:.1f} | {x['lo']:.1f}-{x['hi']:.1f} % | "
                           f"{x['fed']:.3f} / {x['cent']:.3f} | {x['loss_mean']:+.3f} ({x['loss_hi']:+.3f}) | "
                           + (f"{y['mean']:.1f} % ({y['lo']:.1f} %)" if y["n"] >= 2 else "") + " |")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print("\n".join(headline()))
    if "--write" in sys.argv:
        (ROOT / "analysis/CONFIRMATORY_RESULTS.md").write_text(report() + "\n")
        print("written analysis/CONFIRMATORY_RESULTS.md")
