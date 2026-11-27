#!/usr/bin/env python3
"""Figure of the main results (paper/figures/results.pdf), from the run records in evidence/study/.

(a) Fed:Cent per use case and scheme on seeds 100-109: every seed, the mean and its two-sided 95 %
    t-interval (Eq. 2 of the article), with the 90 % and 95 % levels.
(b) KFM12, per class: retention (federated / centralized AP50, means over the ten seeds) against the
    class's training boxes per seed, counted in the images each node trained on (evidence/study/*/subsets,
    which use the file names of the public KFM12 release). The boxes are read from the release's labels
    (--kfm12-release PATH), or, on the authors' machine, from data/study/ through data/kfm12_name_map.csv.

    python tools/make_results_figure.py [--kfm12-release PATH]
"""
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "evidence/study"
OUT = ROOT / "paper/figures/results.pdf"
SEEDS = range(100, 110)
T975 = {9: 2.262}
SCHEMES = {"CWT": ("cwt.json", "bn+head", "#1f6f8b"), "FedAsync": ("fedasync.json", "acal_refresh", "#c8641c")}
USE_CASES = [("kfm12", "KFM12"), ("wbc", "WBC"), ("dentex", "DENTEX")]


def load(p):
    return json.loads(Path(p).read_text())


def ratios(ds, fname, stage):
    out = []
    for s in SEEDS:
        f = load(EV / f"{ds}/s{s}/{fname}")[stage]["map50"]
        c = load(EV / f"{ds}/s{s}/centralized.json")["bn+head"]["map50"]
        out.append(100 * f / c)
    return out


def class_retention(ds, fname, stage):
    fed, cen = {}, {}
    for s in SEEDS:
        f = load(EV / f"{ds}/s{s}/{fname}")[stage]["per_class_map50"]
        c = load(EV / f"{ds}/s{s}/centralized.json")["bn+head"]["per_class_map50"]
        for k in f:
            fed.setdefault(k, []).append(f[k])
            cen.setdefault(k, []).append(c[k])
    return {k: 100 * statistics.mean(fed[k]) / statistics.mean(cen[k]) for k in fed}


def train_boxes(ds, release=None):
    """Mean over seeds of each class's boxes in the images its owner trained on."""
    cfg = yaml.safe_load((ROOT / f"configs/variants/confirm_v2sync/{ds}_s100.yaml").read_text())
    if release:                                  # public release: full labels, keep each node's own classes
        rel = Path(release)
        all_names = [l.split(maxsplit=1)[1] for l in (rel / "classes.txt").read_text().splitlines() if l.strip()]
        def label_lines(node, name):
            p = rel / "labels" / (Path(name).stem + ".txt")
            own = set(node["owned_classes"])
            return [l for l in (p.read_text().splitlines() if p.exists() else [])
                    if l.strip() and all_names[int(l.split()[0])] in own], all_names
    else:                                        # authors' machine: node label files under the original names
        import csv
        with open(ROOT / "data/kfm12_name_map.csv", newline="") as f:
            original = {r["released"]: r["original"] for r in csv.DictReader(f)}
        def label_lines(node, name):
            root = ROOT / Path(node["data_yaml"]).parent
            names = yaml.safe_load((root / "data.yaml").read_text())["names"]
            p = root / "labels/train" / (Path(original[name]).stem + ".txt")
            return [l for l in (p.read_text().splitlines() if p.exists() else []) if l.strip()], names
    per_seed = []
    for s in SEEDS:
        boxes = {}
        for n in cfg["nodes"]:
            for name in (EV / f"{ds}/s{s}/subsets/{n['name']}.txt").read_text().splitlines():
                lines, names = label_lines(n, name)
                for line in lines:
                    c = names[int(line.split()[0])]
                    boxes[c] = boxes.get(c, 0) + 1
        per_seed.append(boxes)
    return {c: statistics.mean(b.get(c, 0) for b in per_seed) for c in per_seed[0]}


def main():
    # journal rule: lettering in Helvetica or Arial (Nimbus Sans is a Helvetica clone), 8-12 pt at final size
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.9, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) Fed:Cent with 95 % intervals
    ylabels, y = [], 0
    for ds, label in USE_CASES:
        for k, (sc, (fname, stage, col)) in enumerate(SCHEMES.items()):
            r = ratios(ds, fname, stage)
            m, sd = statistics.mean(r), statistics.stdev(r)
            h = T975[len(r) - 1] * sd / math.sqrt(len(r))
            yy = -y
            ax.scatter(r, [yy + 0.2] * len(r), s=4, color=col, alpha=0.35, lw=0, zorder=2)
            ax.plot([m - h, m + h], [yy, yy], color=col, lw=2.2, solid_capstyle="butt", zorder=3)
            ax.scatter([m], [yy], s=16, color="white", edgecolor=col, lw=1.3, zorder=4)
            ylabels.append((yy, f"{label} {sc} ({m - h:.1f})"))
            y += 1
        y += 0.6
    for x, ls in ((90, ":"), (95, "--"), (100, "-")):
        ax.axvline(x, color="0.55", lw=0.7, ls=ls, zorder=1)
    ax.set_yticks([p for p, _ in ylabels], [t for _, t in ylabels])
    ax.set_xlim(80, 165)
    ax.set_xlabel("Fed:Cent (%)")
    ax.set_title("a", loc="left", fontweight="bold", x=-0.62)
    ax.tick_params(axis="y", length=0)
    lo = -(y - 0.6) - 0.15
    ax.set_xticks([90, 100, 120, 140, 160])
    ax.set_ylim(lo - 0.3, 0.55)

    # (b) KFM12 per-class retention against training boxes
    import sys
    release = sys.argv[sys.argv.index("--kfm12-release") + 1] if "--kfm12-release" in sys.argv else None
    boxes = train_boxes("kfm12", release)
    for sc, (fname, stage, col) in SCHEMES.items():
        ret = class_retention("kfm12", fname, stage)
        xs = [boxes[c] for c in ret]
        bx.scatter(xs, [ret[c] for c in ret], s=11, color=col, label=sc, zorder=3,
                   marker="o" if sc == "CWT" else "s", alpha=0.9)
        if sc == "FedAsync":
            for c in sorted(ret, key=lambda c: boxes[c])[:3]:
                bx.annotate(c.split("_")[0], (boxes[c], ret[c]), xytext=(5, -2), textcoords="offset points",
                            fontsize=8, color="0.2")
    bx.axhline(85, color="0.55", lw=0.7, ls=":")
    bx.axhline(100, color="0.55", lw=0.7)
    bx.text(800, 86, "85%", fontsize=8, color="0.3", va="bottom", ha="right")
    bx.set_xscale("log")
    bx.set_xlabel("training boxes per seed")
    bx.set_ylabel("class retention (%)")
    bx.set_ylim(45, 110)
    bx.set_title("b", loc="left", fontweight="bold", x=-0.3)
    bx.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.505), handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
