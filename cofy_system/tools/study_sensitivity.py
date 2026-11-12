#!/usr/bin/env python3
"""Seed-level results of the study (seeds 100-109; KFM12, WBC, DENTEX), the level each scheme reaches,
and a sensitivity analysis of the lower bounds (analysis/EVALUATION_PROTOCOL.md).

For each scheme and use case it reads the paired values that the frozen analysis script reads
(_frozen/confirm/tools/confirm_analysis.py: federated and centralized mAP50 of each seed, both after the
scheme's post-processing) and reports:
  level    the highest multiple of 5 % that the t lower bound exceeds, per use case and over all three
           (the level the article states for each scheme);
and the lower end of five 95 % intervals for the mean Fed:Cent ratio:
  t        the two-sided t-interval of the mean ratio (Eq. 2 of the article; the frozen script's);
  log-t    the t-interval of the mean log ratio, back-transformed (a geometric mean: no symmetry
           assumption on the ratio scale);
  BCa      bias-corrected and accelerated bootstrap of the mean ratio (20,000 resamples, seeded);
           with ten seeds bootstrap intervals are known to be somewhat too narrow;
  HL       the exact Wilcoxon signed-rank interval for the Hodges-Lehmann pseudo-median
           (no normality assumption; confidence 95.1 % for n = 10);
  LOO      the smallest t lower bound when any one seed is left out.

    python tools/study_sensitivity.py              # writes analysis/STUDY_SENSITIVITY.md
                                                   # and analysis/study_seed_level.csv
"""
import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "runs/variants"
SEEDS = list(range(100, 110))
DATASETS = ["kfm12", "wbc", "dentex"]
SCHEMES = {   # label: (variant, mode file, calibration stage)
    "CWT": ("confirm_v2sync", "async", "bn+head"),
    "FedAsync": ("confirm_v2async", "async", "acal_refresh"),
}
REF = ("v2_centralized", "centralized", "bn+head")
T975 = {8: 2.306, 9: 2.262}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def pairs(variant, mode, stage, ds):
    out = []
    for s in SEEDS:
        f = (load(VAR / f"calib/{variant}/{ds}/s{s}/{mode}.json") or {}).get(stage)
        c = (load(VAR / f"calib/{REF[0]}/{ds}/s{s}/{REF[1]}.json") or {}).get(REF[2])
        if f and c:
            out.append((s, f["map50"], c["map50"]))
    return out


def t_low(x):
    n = len(x)
    return statistics.mean(x) - T975[n - 1] * statistics.stdev(x) / math.sqrt(n)


def logt_low(x):
    return 100 * math.exp(t_low([math.log(v / 100) for v in x]))


def _norm_ppf(p):
    return statistics.NormalDist().inv_cdf(p)


def bca_low(x, b=20000, seed=0):
    rng = random.Random(seed)
    n, m = len(x), statistics.mean(x)
    boots = sorted(statistics.mean(rng.choices(x, k=n)) for _ in range(b))
    z0 = _norm_ppf(max(1, sum(v < m for v in boots)) / b)
    jack = [statistics.mean(x[:i] + x[i + 1:]) for i in range(n)]
    jm = statistics.mean(jack)
    num = sum((jm - j) ** 3 for j in jack)
    den = 6 * sum((jm - j) ** 2 for j in jack) ** 1.5
    a = num / den if den else 0.0
    z = _norm_ppf(0.025)
    alpha = statistics.NormalDist().cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return boots[min(b - 1, max(0, int(alpha * b)))]


def hl_low(x):
    """Exact Wilcoxon signed-rank confidence interval for the pseudo-median, lower end."""
    n = len(x)
    walsh = sorted((x[i] + x[j]) / 2 for i in range(n) for j in range(i, n))
    # exact null distribution of the signed-rank statistic
    counts = {0: 1}
    for r in range(1, n + 1):
        new = dict(counts)
        for t, c in counts.items():
            new[t + r] = new.get(t + r, 0) + c
        counts = new
    total = 2 ** n
    cum, k = 0, 0
    while True:   # largest k with P(T <= k - 1) <= 0.025
        cum += counts.get(k, 0)
        if cum / total > 0.025:
            break
        k += 1
    conf = 1 - 2 * sum(counts.get(t, 0) for t in range(k)) / total
    return walsh[k], conf


def loo_low(x):
    return min(t_low(x[:i] + x[i + 1:]) for i in range(len(x)))


def level(low):
    """The highest multiple of 5 % at or below a lower bound."""
    return 5 * math.floor(low / 5 + 1e-9)


def main():
    rows, table = [], []
    for scheme, (variant, mode, stage) in SCHEMES.items():
        for ds in DATASETS:
            p = pairs(variant, mode, stage, ds)
            r = [100 * f / c for _, f, c in p]
            rows += [(scheme, ds, s, f, c, 100 * f / c) for s, f, c in p]
            hl, conf = hl_low(r)
            lows = {"t": t_low(r), "log-t": logt_low(r), "BCa": bca_low(r), "HL": hl, "LOO": loo_low(r)}
            table.append((scheme, ds, len(r), statistics.mean(r), lows, conf))
    with open(ROOT / "analysis/study_seed_level.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scheme", "use_case", "seed", "federated_map50", "centralized_map50", "fed_cent_pct"])
        for scheme, ds, s, f, c, r in rows:
            w.writerow([scheme, ds, s, f"{f:.4f}", f"{c:.4f}", f"{r:.2f}"])
    out = ["# Study: seed-level values, levels reached and sensitivity of the lower bounds", "",
           "Generated by `tools/study_sensitivity.py` (protocol: `analysis/EVALUATION_PROTOCOL.md`). "
           "Seed-level values: `analysis/study_seed_level.csv`.", "",
           "## Level reached", "",
           "Highest multiple of 5 % that the t lower bound exceeds:", "",
           "| scheme | " + " | ".join(DATASETS) + " | all three |", "|---|" + "---|" * (len(DATASETS) + 1)]
    for scheme in SCHEMES:
        lows = {ds: lw["t"] for sc, ds, n, m, lw, c in table if sc == scheme}
        out.append(f"| {scheme} | " + " | ".join(f"{level(lows[d])} % ({lows[d]:.1f})" for d in DATASETS)
                   + f" | {level(min(lows.values()))} % |")
    out += ["", "## Lower ends of 95 % intervals for the mean Fed:Cent (%)", "",
            "| scheme | use case | n | mean | t (Eq. 2) | log-t | BCa bootstrap | Hodges-Lehmann | leave-one-out t (min) | level of the scheme | all above |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    claim = {scheme: level(min(lw["t"] for sc, ds, n, m, lw, c in table if sc == scheme)) for scheme in SCHEMES}
    for scheme, ds, n, m, lows, conf in table:
        ok = all(v >= claim[scheme] for v in lows.values())
        out.append(f"| {scheme} | {ds} | {n} | {m:.1f} | {lows['t']:.1f} | {lows['log-t']:.1f} | {lows['BCa']:.1f} | "
                   f"{lows['HL']:.1f} | {lows['LOO']:.1f} | {claim[scheme]} | {'yes' if ok else 'NO'} |")
    out += ["", f"Hodges-Lehmann interval: exact Wilcoxon signed-rank, confidence {100 * table[0][5]:.1f} % for n = 10.", ""]
    out += ["## Seed-level values", "", "| scheme | use case | " + " | ".join(f"s{s}" for s in SEEDS) + " |",
            "|---|---|" + "---|" * len(SEEDS)]
    for scheme in SCHEMES:
        for ds in DATASETS:
            v = {s: r for sc, d, s, _, _, r in rows if sc == scheme and d == ds}
            out.append(f"| {scheme} | {ds} | " + " | ".join(f"{v[s]:.1f}" if s in v else "-" for s in SEEDS) + " |")
    text = "\n".join(out) + "\n"
    (ROOT / "analysis/STUDY_SENSITIVITY.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
