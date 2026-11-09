#!/usr/bin/env python3
"""Fairness of the federated schemes across classes and nodes: does federation cost
some classes, or the classes of some nodes, more than others?

For every scheme, dataset and class: centralized and federated AP50 (both after the
same post-processing, as in the main comparison), averaged over seeds; retention =
federated / centralized x 100; the paired per-seed difference with its 95 % CI.
Per node: the mean retention of the classes it owns. Associations (Spearman, with a
permutation p-value) between a class's retention and (a) its number of training boxes,
(b) its owner's number of training images, and, for the concurrent scheme, (c) how
late its owner's last update arrived (the recency effect seen in the order analysis).

    python tools/fairness_analysis.py                 # exploratory rows (seeds of the search)
    python tools/fairness_analysis.py --confirm       # pre-registered confirmation rows (seeds 100-109)
    python tools/fairness_analysis.py --out analysis/FAIRNESS.md
"""
import json, random, statistics, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "runs/variants"
DATASETS = ["kfm12", "wbc", "chestx", "dentex"]   # dentex: held-out, confirmation seeds only
EXPLORATORY = [   # (label, variant, mode file, stage, seeds)
    ("sync sequential", "v2_async", "async", "bn+head", [0, 1, 2]),
    ("async concurrent", "v2_vclock", "async", "acal_refresh", [0, 1, 2, 3]),
    ("sync parallel FedAvg", "v2_sync", "sync", "bn+head", [0, 1, 2]),
]
CONFIRM = [
    ("sync sequential", "confirm_v2sync", "async", "bn+head", list(range(100, 110))),
    ("async concurrent", "confirm_v2async", "async", "acal_refresh", list(range(100, 110))),
]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def ownership(ds, variant):
    cfg = next(iter(sorted((ROOT / f"configs/variants/{variant}").glob(f"{ds}_s*.yaml"))), None)
    if cfg is None:   # the variant was not run on this dataset (DENTEX has no development runs)
        return None, None
    c = yaml.safe_load(cfg.read_text())
    return {cl: n["name"] for n in c["nodes"] for cl in n["owned_classes"]}, c["nodes"]


def box_counts(nodes):
    """Training boxes per class and training images per node (full node pools)."""
    boxes, images = {}, {}
    for n in nodes:
        root = ROOT / Path(n["data_yaml"]).parent
        names = yaml.safe_load((root / "data.yaml").read_text())["names"]
        files = list((root / "labels/train").glob("*.txt"))
        images[n["name"]] = len(list((root / "images/train").iterdir()))
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    cl = names[int(line.split()[0])]
                    boxes[cl] = boxes.get(cl, 0) + 1
    return boxes, images


def recency(variant, ds, seeds):
    """{node: mean over seeds of the finish time of its last update / the run's last event} (concurrent runs)."""
    out = {}
    for s in seeds:
        v = load(VAR / f"{variant}/{ds}/s{s}/async/virtual_schedule.json")
        if not v:
            return None
        end = max(e["finish"] for e in v["events"])
        for node in v["speed"]:
            last = max(e["finish"] for e in v["events"] if e["node"] == node)
            out.setdefault(node, []).append(last / end)
    return {k: statistics.mean(v) for k, v in out.items()}


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y, perms=20000):
    """(rho, two-sided permutation p)."""
    rx, ry = ranks(x), ranks(y)
    def rho(a, b):
        ma, mb = statistics.mean(a), statistics.mean(b)
        num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
        return num / den if den else 0.0
    r0 = rho(rx, ry)
    rng = random.Random(0)
    ry2 = list(ry)
    hits = 0
    for _ in range(perms):
        rng.shuffle(ry2)
        hits += abs(rho(rx, ry2)) >= abs(r0) - 1e-12
    return r0, (hits + 1) / (perms + 1)


def analyse(rows):
    out, summary = [], []
    for label, variant, mfile, stage, seeds in rows:
        out += [f"## {label} (`{variant}`, {stage})", ""]
        for ds in DATASETS:
            owner, nodes = ownership(ds, variant)
            if owner is None:
                continue
            boxes, images = box_counts(nodes)
            fed, cen = {}, {}
            for s in seeds:
                f = (load(VAR / f"calib/{variant}/{ds}/s{s}/{mfile}.json") or {}).get(stage)
                c = (load(VAR / f"calib/v2_centralized/{ds}/s{s}/centralized.json") or {}).get("bn+head")
                if not (f and c):
                    continue
                for cl in owner:
                    fed.setdefault(cl, []).append(f["per_class_map50"][cl])
                    cen.setdefault(cl, []).append(c["per_class_map50"][cl])
            if not fed:
                out += [f"**{ds}**: no finished seeds.", ""]
                continue
            n = len(next(iter(fed.values())))
            ret = {cl: statistics.mean(fed[cl]) / statistics.mean(cen[cl]) * 100 for cl in owner if statistics.mean(cen[cl]) > 0}
            out += [f"**{ds}** (n = {n} seeds)", "",
                    "| class | owner | train boxes | cent AP50 | fed AP50 | retention | diff (95 % CI) |",
                    "|---|---|---|---|---|---|---|"]
            for cl in sorted(owner, key=lambda c: ret.get(c, 0)):
                d = [a - b for a, b in zip(fed[cl], cen[cl])]
                h = T975[n - 1] * statistics.stdev(d) / n ** 0.5 if n > 1 else float("nan")
                out.append(f"| {cl} | {owner[cl][-1]} | {boxes.get(cl, 0)} | {statistics.mean(cen[cl]):.3f} | "
                           f"{statistics.mean(fed[cl]):.3f} | {ret.get(cl, float('nan')):.0f} % | "
                           f"{statistics.mean(d):+.3f} ({statistics.mean(d) - h:+.3f}, {statistics.mean(d) + h:+.3f}) |")
            per_node = {nd["name"]: statistics.mean(ret[c] for c in nd["owned_classes"] if c in ret) for nd in nodes}
            cls = [c for c in owner if c in ret]
            vals = sorted(ret[c] for c in cls)
            rb = spearman([ret[c] for c in cls], [boxes.get(c, 0) for c in cls])
            ri = spearman([ret[c] for c in cls], [images[owner[c]] for c in cls])
            rec = recency(variant, ds, seeds) if "vclock" in variant or "v2async" in variant else None
            rr = spearman([ret[c] for c in cls], [rec[owner[c]] for c in cls]) if rec else None
            q1, q3 = statistics.quantiles(vals, n=4)[0], statistics.quantiles(vals, n=4)[2]
            out += ["", "Per node (mean retention of owned classes): "
                    + ", ".join(f"{k[-1]} {v:.0f} %" for k, v in per_node.items())
                    + (" | last update at (fraction of the run): " + ", ".join(f"{k[-1]} {v:.2f}" for k, v in rec.items())
                       if rec else ""), "",
                    f"Worst class {min(vals):.0f} %, median {statistics.median(vals):.0f} %, IQR {q1:.0f}-{q3:.0f} %, "
                    f"best {max(vals):.0f} %; classes below 85 %: {sum(v < 85 for v in vals)}/{len(vals)}. "
                    f"Spearman with training boxes {rb[0]:+.2f} (p = {rb[1]:.2f}), with owner's images {ri[0]:+.2f} "
                    f"(p = {ri[1]:.2f})" + (f", with owner's last-update time {rr[0]:+.2f} (p = {rr[1]:.2f})" if rr else "")
                    + ".", ""]
            summary.append((label, ds, n, min(vals), statistics.median(vals), max(vals), sum(v < 85 for v in vals), len(vals),
                            min(per_node.values()), max(per_node.values()), rb, rr))
    head = ["| scheme | dataset | n | worst class | median | best | classes < 85 % | node range | rho(boxes) | rho(recency) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    head += [f"| {a} | {ds} | {n} | {w:.0f} % | {m:.0f} % | {b:.0f} % | {k}/{t} | {lo:.0f}-{hi:.0f} % | "
             f"{rb[0]:+.2f} (p {rb[1]:.2f}) | " + (f"{rr[0]:+.2f} (p {rr[1]:.2f})" if rr else "-") + " |"
             for a, ds, n, w, m, b, k, t, lo, hi, rb, rr in summary]
    return head, out


def main():
    rows = CONFIRM if "--confirm" in sys.argv else EXPLORATORY
    head, body = analyse(rows)
    text = "\n".join(["# Fairness across classes and nodes" + (" (confirmation seeds)" if rows is CONFIRM else
                                                                 " (exploratory seeds)"), "",
                      "Generated by `tools/fairness_analysis.py`. Retention = federated AP50 / centralized AP50 of the "
                      "class (means over seeds, both calibrated). Rho: Spearman correlation of class retention with "
                      "the class's training boxes / its owner's last-update time; p from 20,000 permutations.", "",
                      "## Summary", ""] + head + [""] + body)
    print(text)
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(text + "\n")


if __name__ == "__main__":
    main()
