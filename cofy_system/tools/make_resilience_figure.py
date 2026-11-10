#!/usr/bin/env python3
"""Figure for the README (docs/img/resisted_attacks.png): attacks on a WBC evaluation image, with the
ground truth, the federation under attack and the same federation with the defence that resisted it.
Only attacks that a defence resisted on this image are shown. The sign flip is not: peer validation (P3)
rolls back the attacker's class, but the frozen rows fire again as the shared layers change, and on this
image the class still fills the detection budget.
Kept final models of the robustness study and its candidate defences; seed 0 (attacker node A, which
owns neutrophil) unless stated:

  BatchNorm poisoning  the concurrent seed that collapsed (seed 2, attacker node C), final model before
                       calibration  vs  the BatchNorm step of the calibration with median pooling, the
                       attacker also poisoning its statistics (P1; detections from attacks_cache.json);
  backdoor             the backdoored federation (rb_seq_backdoor_ora) on the image with the trigger  vs
                       the same detections after input screening at node B (P2, Detector Cleanse, the
                       study's stored threshold for this model and seed): flagged detections are removed.

The image is the one of the paper's attack figure (tools/make_attack_figure.py). CPU is enough.

    CUDA_VISIBLE_DEVICES= python tools/make_resilience_figure.py
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "docs/img/resisted_attacks.png"
GREEN = "#2e8b57"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    import torch
    import make_attack_figure as maf
    import defence_eval as de

    torch.set_num_threads(6)
    cache = json.loads(maf.CACHE.read_text())
    classes = cache["classes"]
    ev_img = ROOT / "data/study/wbc/_eval/images/val" / cache["eval"]
    trig_img = ROOT / "data/attack/wbc/_eval_trigger_node_A/images" / cache["eval"]
    gt = cache["gt"]
    own_A = set(cache["own_A"])


    # P2 screening at node B on the backdoored model's detections, as tools/defence_eval.py does it
    bd = maf.Model("rb_seq_backdoor_ora", 0)
    det_bd = bd.detect(trig_img)
    stored = json.loads((ROOT / "runs/variants/p2_cleanse/wbc/s0/summary.json").read_text())["cleanse"]["rb_seq_backdoor_ora"]
    scr = next(n for n in bd.cfg.nodes if n.name == stored["screening_node"])
    scr_own = {classes.index(c) for c in scr.owned_classes}
    rng = random.Random(1000 + 0)
    sc = de.Screen(bd.model, bd.cfg.nc, de._crops(scr, de.N_FEAT, rng), bd.dev)
    img, r, left, top = de._letterbox(trig_img, bd.cfg.model.imgsz)
    screened = []
    for c, cf, x1, y1, x2, y2 in det_bd:
        box = [x1 * r + left, y1 * r + top, x2 * r + left, y2 * r + top]
        h = None if c in scr_own else sc.entropy(img, torch.tensor(box))
        screened.append((c, cf, x1, y1, x2, y2, h is not None and h < stored["threshold"], h))
    bd.release()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0})
    rows = [
        ("BatchNorm poisoning: node C corrupts the statistics", ev_img,
         ("final model before calibration", cache["det"]["bnpoison"], set(cache["own_bn"])),
         ("+ calibration, median of the nodes'\nstatistics (node C poisons its own)", cache["det"]["recalibrated"], set(cache["own_bn"]))),
        ("Backdoor: node A labels a trigger as neutrophil", trig_img,
         ("backdoored federation", det_bd, own_A),
         ("+ input screening at node B:\nflagged detections removed", [d[:6] for d in screened if not d[6]], own_A)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 5.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.09, wspace=0.04, hspace=0.38)
    win = maf.crop_window(ev_img, [])

    def show(ax, path, title):
        ax.imshow(Image.open(path).convert("RGB"))
        x0, y0, s = win
        ax.set_xlim(x0, x0 + s); ax.set_ylim(y0 + s, y0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=7, loc="left", pad=3, color=maf.INK)

    def draw(ax, ds, own):
        labelled = []
        for c, cf, *b in sorted(ds, key=lambda d: -d[1]):
            col = maf.RED if c in own else maf.BLUE
            big = len(ds) <= 20
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=1.4 if big else 0.5))
            if big and not any(maf.iou(b, o) > 0.5 for o in labelled):
                labelled.append(b)
                ax.text((b[0] + b[2]) / 2, b[1] - 3, f"{classes[c]} {cf:.2f}", fontsize=6, ha="center", va="bottom", color=maf.INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.6, alpha=0.9))
        if not ds:
            ax.text(0.5, 0.05, "no detection", transform=ax.transAxes, ha="center", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    for i, (name, img_path, (t1, d1, o1), (t2, d2, o2)) in enumerate(rows):
        ax = axes[i, 0]; show(ax, img_path, "ground truth")
        for c, *b in gt:
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=GREEN, lw=1.4, ls=(0, (3, 2))))
            ax.text((b[0] + b[2]) / 2, b[1] - 3, classes[c], fontsize=6, ha="center", va="bottom", color=maf.INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=GREEN, lw=0.6, alpha=0.9))
        ax = axes[i, 1]; show(ax, img_path, t1); draw(ax, d1, o1)
        ax = axes[i, 2]; show(ax, img_path, t2); draw(ax, d2, o2)
        fig.text(0.01, axes[i, 0].get_position().y1 + 0.06, name, fontsize=8, fontweight="bold", color=maf.INK)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=GREEN, lw=1.4, ls=(0, (3, 2)), label="ground truth"),
                        Line2D([], [], color=maf.RED, lw=1.4, label="attacker's class(es)"),
                        Line2D([], [], color=maf.BLUE, lw=1.4, label="other classes")],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(json.dumps({"backdoor_detections": len(det_bd),
                      "screening": [(classes[d[0]], round(d[1], 2), None if d[7] is None else round(d[7], 4), d[6]) for d in screened],
                      "threshold": stored["threshold"]}, indent=1))


if __name__ == "__main__":
    main()
