#!/usr/bin/env python3
"""Figure of the three use cases (paper/figures/usecases.pdf): one evaluation image of each, with every
labelled object coloured by the node that owns its class, so that the way classes share an image shows
at a glance (never in WBC, sometimes in KFM12, always in DENTEX).

Images are chosen by a fixed rule (first match in sorted order of the fully labelled evaluation set):
  WBC     the first image;
  KFM12   the first image with 3-8 objects from classes of at least two nodes, all in the central
          80 % of the image;
  DENTEX  the first image.
Node colours keep the hues of the paper's diagrams (A blue, B orange, C green, D purple); the palette
passes the all-pairs colour-vision check, and each node also has its own line style.

    python tools/make_usecase_figure.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/figures/usecases.pdf"
NODE = {"node_A": ("#1c5cab", "-"), "node_B": ("#eb6834", (0, (3, 1.5))),
        "node_C": ("#5cc98a", (0, (1, 1))), "node_D": ("#b25ad6", (0, (4, 1.2, 1, 1.2)))}
INK, MUTED = "#1f1f1e", "#6b6a64"
CFG = {"kfm12": "configs/variants/confirm_v2sync/kfm12_s100.yaml", "wbc": "configs/variants/confirm_v2sync/wbc_s100.yaml",
       "dentex": "configs/variants/confirm_v2sync/dentex_s100.yaml"}


def boxes(lab, W, H):
    out = []
    for line in lab.read_text().splitlines():
        v = line.split()
        if len(v) == 5:
            c, x, y, w, h = int(v[0]), *map(float, v[1:])
            out.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return out


def pick(ds):
    from PIL import Image
    c = yaml.safe_load((ROOT / CFG[ds]).read_text())
    G = c["global_classes"]
    owner = {G.index(cl): n["name"] for n in c["nodes"] for cl in n["owned_classes"]}
    ev = ROOT / f"data/study/{ds}/_eval"
    for p in sorted((ev / "images/val").iterdir()):
        W, H = Image.open(p).size
        b = boxes(ev / "labels/val" / (p.stem + ".txt"), W, H)
        if not b:
            continue
        if ds == "kfm12":
            central = all(0.1 * W <= (x1 + x2) / 2 <= 0.9 * W and 0.1 * H <= (y1 + y2) / 2 <= 0.9 * H for _, x1, y1, x2, y2 in b)
            if not (3 <= len(b) <= 8 and len({owner[k] for k, *_ in b}) >= 2 and central):
                continue
        return p, b, owner, G, (W, H)
    raise SystemExit(f"no image matches the rule for {ds}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "axes.linewidth": 0})
    order = [("wbc", "WBC"), ("kfm12", "KFM12"), ("dentex", "DENTEX")]
    picks = {ds: pick(ds) for ds, _ in order}
    def window(ds):
        """The whole image, or for DENTEX the dental arch: the boxes' extent with a 4 % margin."""
        _, b, _, _, (W, H) = picks[ds]
        if ds != "dentex":
            return 0, 0, W, H
        m = 0.04 * W
        return (max(0, min(x for _, x, *_ in b) - m), max(0, min(y for _, _, y, *_ in b) - m),
                min(W, max(x for *_, x, _ in b) + m), min(H, max(y for *_, y in b) + m))
    wins = {ds: window(ds) for ds, _ in order}
    ratios = [(wins[ds][2] - wins[ds][0]) / (wins[ds][3] - wins[ds][1]) for ds, _ in order]
    fig = plt.figure(figsize=(5.17, 1.75))     # the paper's text width (372 pt): printed at 100 %
    gs = fig.add_gridspec(1, 3, width_ratios=ratios, left=0.005, right=0.995, top=0.80, bottom=0.15, wspace=0.03)
    for k, (ds, name) in enumerate(order):
        p, b, owner, G, (W, H) = picks[ds]
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(Image.open(p).convert("RGB"))
        x0, y0, x1w, y1w = wins[ds]
        ax.set_xlim(x0, x1w); ax.set_ylim(y1w, y0)
        ax.set_xticks([]); ax.set_yticks([])
        for c, x1, y1, x2, y2 in b:
            col, ls = NODE[owner[c]]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=0.9 if ds == "dentex" else 1.2, ls=ls))
        n_cls, n_own = len({c for c, *_ in b}), len({owner[c] for c, *_ in b})
        ax.set_title(f"$\\bf{{{name}}}$\n{len(b)} object{'s' if len(b) > 1 else ''}, {n_cls} class{'es' if n_cls > 1 else ''}, "
                     f"{n_own} node{'s' if n_own > 1 else ''}", fontsize=6, color=INK, loc="left", pad=2, linespacing=1.3)
    handles = [Line2D([], [], color=col, lw=1.4, ls=ls, label=f"classes of {n.replace('_', ' ')}") for n, (col, ls) in NODE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=5.8, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.6, columnspacing=1.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".png"), dpi=250)
    print({ds: picks[ds][0].name for ds, _ in order})


if __name__ == "__main__":
    main()
