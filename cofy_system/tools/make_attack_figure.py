#!/usr/bin/env python3
"""Figure of the robustness study's attacks (paper/figures/attacks.pdf): the data attacks on images of
the three use cases, and what kept federated models of the robustness study (WBC) detect.

Rows 1-3 (KFM12, WBC, DENTEX), what the attacker changes or what the model is shown (attacker of seed 0:
node_A; its classes in red):
  (a) one of the attacker's training images with its own boxes;
  (b) the same image with the label corruption: half the boxes dropped (grey only), the others moved by
      up to half their size and resized by x0.5-1.5;
  (c) a backdoor sample: a checkerboard trigger (8 % of the image's short side) pasted at a random place
      and labelled as the attacker's first class;
  (d) an evaluation image without that class; (e) the same image with the trigger.
  WBC uses the study's own attack copies (data/attack/wbc, tools/make_attack_data.py); KFM12 and DENTEX,
  on which the robustness study was not run, apply the same rules here, with this figure's own seed.
  Each panel is a square crop of side = the image's short side, around the attacker's objects (the whole
  image for WBC; the central part of a DENTEX panoramic radiograph).
Row 4, what kept final models of the study detect on the WBC images of (d)/(e) (confidence >= 0.25,
standard NMS):
  (f) defended clean federation on (d); (g) defended backdoored federation on (e);
  (h) defended sign-flip federation on (d); (i) the concurrent BatchNorm-poisoning seed that collapsed
  (saved final model, uncalibrated) on (d); (j) the same model after the BatchNorm step of the
  calibration with median pooling, the attacker also poisoning the statistics it sends (candidate P1).
Every image is chosen by a fixed rule (first match in sorted order), so the figure is reproducible:
  WBC (a,b) first node_A training image whose corrupted box keeps IoU < 0.4 with the honest one and
  stays inside the image; (c) first poisoned image whose trigger does not overlap the cell; (d,e) first
  triggered evaluation image of class lymphocyte (owned by neither attacker) on which the most confident
  detection of the cell (IoU >= 0.5) is lymphocyte for the clean and for the recalibrated model, and the
  backdoored model reports neutrophil on the trigger. KFM12 / DENTEX (a,b) first node_A training image
  with 2-6 of its boxes of which the corruption drops at least one and keeps at least one; (c) the same
  image with the trigger; (d) first evaluation image without the attacker's first class (KFM12: with 2-4
  objects, all in the central 60 % of the image).
The images illustrate the mechanisms; how often each happens is in the paper's tables.
Each object shows its most confident detection; "+k" counts the other classes detected on it at lower
confidence (>= 0.25), which a model trained on partial labels does, since no node ever labels a
lymphocyte as "not neutrophil". The sign-flip panel draws every box.
The model detections are cached in paper/figures/attacks_cache.json (delete it to recompute; the
recalibration in (j) takes about 15 min on a CPU).

    python tools/make_attack_figure.py          # CUDA_VISIBLE_DEVICES= to keep off a busy GPU
"""
import dataclasses, json, random, shutil, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
VAR = ROOT / "runs/variants"
OUT = ROOT / "paper/figures/attacks.pdf"
CACHE = OUT.with_name("attacks_cache.json")
RED, BLUE, INK, MUTED = "#e34948", "#2a78d6", "#1f1f1e", "#6b6a64"   # attacker's classes / others (validated pair)
CONF = 0.25
DISPLAY = {"kfm12": "KFM12", "wbc": "WBC", "dentex": "DENTEX"}


def letterbox(path, size):
    import cv2
    im = cv2.imread(str(path))[:, :, ::-1]
    h, w = im.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    rs = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = torch.full((3, size, size), 114 / 255.0)
    canvas[:, top:top + nh, left:left + nw] = torch.from_numpy(rs.copy()).permute(2, 0, 1).float() / 255.0
    return im, canvas, r, left, top


def yolo_boxes(label_path, W, H):
    out = []
    for line in Path(label_path).read_text().splitlines():
        v = line.split()
        if len(v) == 5:
            c, x, y, w, h = int(v[0]), *map(float, v[1:])
            out.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


class Model:
    def __init__(self, name, s, sd=None):
        from cofy.config import load_config
        from cofy.model import build_model
        self.cfg = load_config(ROOT / f"configs/variants/{name}/wbc_s{s}.yaml")
        run = VAR / f"{name}/wbc/s{s}"
        ck = next(run / m / "global_final.pt" for m in ("async", "sync") if (run / m / "global_final.pt").exists())
        self.sd = sd if sd is not None else torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.cfg.model.arch, self.cfg.nc, self.cfg.model.imgsz, pretrained=False,
                                 class_names=self.cfg.global_classes)
        self.model.load_state_dict(self.sd)
        self.model.eval().to(self.dev)

    def detect(self, path):
        """[(class, conf, x1, y1, x2, y2)] in the original image's pixels."""
        from ultralytics.utils.nms import non_max_suppression
        im, x, r, left, top = letterbox(path, self.cfg.model.imgsz)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.dev.type == "cuda"):
            p = self.model(x[None].to(self.dev))[0].float()
        d = non_max_suppression(p, conf_thres=CONF, iou_thres=0.7, max_det=300)[0].cpu()
        return [(int(c), float(cf), (x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r)
                for x1, y1, x2, y2, cf, c in d.tolist()]

    def release(self):
        del self.model
        torch.cuda.empty_cache()


def pick_inputs(classes):
    from PIL import Image
    st, at = ROOT / "data/study/wbc", ROOT / "data/attack/wbc"
    tr = st / "node_A/images/train"
    a_img = honest = corrupt = None
    for p in sorted(tr.iterdir()):
        lab = st / "node_A/labels/train" / (p.stem + ".txt")
        cl = at / "node_A_labels/labels/train" / (p.stem + ".txt")
        if not cl.exists() or not cl.read_text().strip():
            continue
        W, H = Image.open(p).size
        o, c = yolo_boxes(lab, W, H), yolo_boxes(cl, W, H)
        if len(o) == 1 and len(c) == 1 and iou(o[0][1:], c[0][1:]) < 0.4 \
                and c[0][1] >= 0 and c[0][2] >= 0 and c[0][3] <= W and c[0][4] <= H:
            a_img, honest, corrupt = p, o, c
            break
    bd = at / "node_A_backdoor"
    c_img = c_boxes = None
    for p in sorted((bd / "images/train").iterdir()):
        if p.is_symlink():
            continue
        W, H = Image.open(p).size
        b = yolo_boxes(bd / "labels/train" / (p.stem + ".txt"), W, H)
        if len(b) == 2 and iou(b[0][1:], b[1][1:]) == 0:
            c_img, c_boxes = p, b
            break
    return a_img, honest, corrupt, c_img, c_boxes


def pick_eval(clean, backdoor, rec, classes):
    from PIL import Image
    trig = ROOT / "data/attack/wbc/_eval_trigger_node_A"
    t = json.loads((trig / "triggers.json").read_text())
    ev = ROOT / "data/study/wbc/_eval"
    eos = classes.index("lymphocyte")
    for f in sorted(t["boxes"]):
        gt = yolo_boxes(ev / "labels/val" / (Path(f).stem + ".txt"), *Image.open(ev / "images/val" / f).size)
        if not gt or gt[0][0] != eos:
            continue
        W, H = Image.open(trig / "images" / f).size
        x, y, w, h = t["boxes"][f]
        tb = ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
        dc = clean.detect(ev / "images/val" / f)
        on_cell = [d for d in dc if iou(d[2:], gt[0][1:]) >= 0.5]
        if not on_cell or max(on_cell, key=lambda d: d[1])[0] != eos:
            continue
        db = backdoor.detect(trig / "images" / f)
        if not any(c == t["target_id"] and iou(b, tb) >= 0.5 for c, _, *b in db):
            continue
        dr = [d for d in rec.detect(ev / "images/val" / f) if iou(d[2:], gt[0][1:]) >= 0.5]
        if dr and max(dr, key=lambda d: d[1])[0] == eos:
            return f, gt, tb, dc, db
    raise SystemExit("no evaluation image matches the rule")



def wbc_models():
    """The WBC panels' images and the kept models' detections (cached)."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    torch.set_num_threads(6)
    from cofy.calibrate import node_bn_stats, pool_bn_stats
    from cofy.robust import attacker_name, poison_bn_stats
    bn = Model("rb_conc_bnpoison_ora_budget", 2)
    work = ROOT / "runs/_figure_work"
    cfg = dataclasses.replace(bn.cfg, output_dir=str(work))
    att_bn = attacker_name(cfg)
    stats = [node_bn_stats(n, cfg, bn.sd, bn.dev) for n in cfg.nodes]
    stats = [(poison_bn_stats(st, 3.0) if n.name == att_bn else st, k) for n, (st, k) in zip(cfg.nodes, stats)]
    sd_med = pool_bn_stats(bn.sd, stats, "median")
    shutil.rmtree(work, ignore_errors=True)
    rec = Model("rb_conc_bnpoison_ora_budget", 2, sd=sd_med)
    clean = Model("rb_seq_clean_ora", 0)
    classes = clean.cfg.global_classes
    backdoor = Model("rb_seq_backdoor_ora", 0)
    a_img, honest, corrupt, c_img, c_boxes = pick_inputs(classes)
    f, gt, tb, det_clean, det_bd = pick_eval(clean, backdoor, rec, classes)
    clean.release(); backdoor.release()
    ev_img = ROOT / "data/study/wbc/_eval/images/val" / f
    det_bn = bn.detect(ev_img); bn.release()
    det_rec = rec.detect(ev_img); rec.release()
    sf = Model("rb_seq_signflip_ora_budget", 0)
    det_sf = sf.detect(ev_img); sf.release()
    own_A = sorted(classes.index(c) for c in next(n.owned_classes for n in clean.cfg.nodes if n.name == "node_A"))
    own_bn = sorted(classes.index(c) for c in next(n.owned_classes for n in cfg.nodes if n.name == att_bn))
    out = {"classes": classes, "a_img": str(a_img.relative_to(ROOT)), "honest": honest, "corrupt": corrupt,
           "c_img": str(c_img.relative_to(ROOT)), "c_boxes": c_boxes, "eval": f, "gt": gt, "trigger": list(tb),
           "det": {"clean": det_clean, "backdoor": det_bd, "signflip": det_sf, "bnpoison": det_bn, "recalibrated": det_rec},
           "own_A": own_A, "own_bn": own_bn, "bnpoison_attacker": att_bn}
    CACHE.write_text(json.dumps(out))
    return out


def synth(ds, work):
    """KFM12 / DENTEX: the data attacks of tools/make_attack_data.py applied with this figure's seed."""
    import yaml
    from PIL import Image
    from make_attack_data import paste_trigger
    rng = random.Random(f"figure-{ds}-node_A")
    st = ROOT / f"data/study/{ds}"
    meta = yaml.safe_load((st / "node_A/data.yaml").read_text())
    names = meta["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    glob_cfg = yaml.safe_load((ROOT / f"configs/variants/v2_central/{ds}_s0.yaml").read_text())
    G = glob_cfg["global_classes"]
    a = None
    for p in sorted((st / "node_A/images/train").iterdir()):
        W, H = Image.open(p).size
        b = [(G.index(names[c]), *bx) for c, *bx in yolo_boxes(st / "node_A/labels/train" / (p.stem + ".txt"), W, H)]
        if not 2 <= len(b) <= (4 if ds == "kfm12" else 99):
            continue
        r2 = random.Random(f"figure-{ds}-{p.stem}")
        kept, dropped = [], []
        for c, x1, y1, x2, y2 in b:
            if r2.random() < 0.5:
                dropped.append((c, x1, y1, x2, y2)); continue
            w, h = x2 - x1, y2 - y1
            x, y = (x1 + x2) / 2 + r2.uniform(-0.5, 0.5) * w, (y1 + y2) / 2 + r2.uniform(-0.5, 0.5) * h
            w, h = w * r2.uniform(0.5, 1.5), h * r2.uniform(0.5, 1.5)
            x, y = min(max(x, w / 2), W - w / 2), min(max(y, h / 2), H - h / 2)
            kept.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        if kept and dropped:
            a = (p, b, kept)
            break
    p, honest, corrupt = a
    work.mkdir(parents=True, exist_ok=True)
    c_img = work / f"{ds}_backdoor.jpg"
    x, y, w, h = paste_trigger(p, c_img, rng)
    W, H = Image.open(p).size
    trig_c = ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
    first = G.index(names[0])
    ev = st / "_eval"
    for q in sorted((ev / "images/val").iterdir()):
        W2, H2 = Image.open(q).size
        gt = yolo_boxes(ev / "labels/val" / (q.stem + ".txt"), W2, H2)
        central = all(0.2 * W2 <= (g[1] + g[3]) / 2 <= 0.8 * W2 and 0.2 * H2 <= (g[2] + g[4]) / 2 <= 0.8 * H2 for g in gt)
        if gt and first not in {g[0] for g in gt} and (ds != "kfm12" or (2 <= len(gt) <= 4 and central)):
            break
    e_img = work / f"{ds}_eval_trigger.jpg"
    x, y, w, h = paste_trigger(q, e_img, rng)
    trig_e = ((x - w / 2) * W2, (y - h / 2) * H2, (x + w / 2) * W2, (y + h / 2) * H2)
    own = sorted(G.index(n) for n in names)
    return {"classes": G, "a_img": p, "honest": honest, "corrupt": corrupt, "c_img": c_img, "trig_c": trig_c,
            "eval": q, "gt": gt, "e_img": e_img, "trig_e": trig_e, "own": own, "first": first}


def nearest(boxes, win):
    """Index of the box whose centre is nearest the window's centre."""
    x0, y0, s = win
    return min(range(len(boxes)), key=lambda i: ((boxes[i][-4] + boxes[i][-2]) / 2 - x0 - s / 2) ** 2
               + ((boxes[i][-3] + boxes[i][-1]) / 2 - y0 - s / 2) ** 2)


def crop_window(path, boxes):
    """Square crop of side = the image's short side, centred on the boxes (clamped to the image)."""
    from PIL import Image
    W, H = Image.open(path).size
    s = min(W, H)
    if not boxes:
        cx, cy = W / 2, H / 2
    else:
        cx = sum((b[-4] + b[-2]) / 2 for b in boxes) / len(boxes)
        cy = sum((b[-3] + b[-1]) / 2 for b in boxes) / len(boxes)
    x0, y0 = min(max(cx - s / 2, 0), W - s), min(max(cy - s / 2, 0), H - s)
    return x0, y0, s


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "axes.linewidth": 0})
    wm = wbc_models()
    work = ROOT / "runs/_figure_work"
    syn = {ds: synth(ds, work) for ds in ("kfm12", "dentex")}
    wcls = wm["classes"]

    fig, axes = plt.subplots(4, 5, figsize=(5.17, 4.75))   # the paper's text width (372 pt): printed at 100 %
    fig.subplots_adjust(left=0.05, right=0.997, top=0.962, bottom=0.055, wspace=0.04, hspace=0.12)

    def show(ax, path, win, title=None):
        x0, y0, s = win
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_xlim(x0, x0 + s); ax.set_ylim(y0 + s, y0)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=6, color=INK, loc="left", pad=2)

    def box(ax, b, color, dashed=False, label=None, lw=1.0):
        x1, y1, x2, y2 = b
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw,
                               ls=(0, (3, 2)) if dashed else "-", clip_on=True))
        if label:   # above the box (below it for small boxes or near the top); centred on central boxes
            (X0, X1), (Y1, Y0) = ax.get_xlim(), ax.get_ylim()
            W, H = X1 - X0, Y1 - Y0
            top = y1 > Y0 + 0.12 * H and (x2 - x1) > 0.15 * W
            cx = (x1 + x2) / 2
            ha, x = ("left", max(x1, X0) + 1) if cx < X0 + 0.3 * W else ("right", min(x2, X1) - 1) \
                if cx > X0 + 0.7 * W else ("center", cx)
            ax.text(x, y1 - 2 if top else y2 + 2, label, fontsize=4.6, color=INK, clip_on=True,
                    va="bottom" if top else "top", ha=ha,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.6, alpha=0.9))

    def dets(ax, ds, own):
        """Per object, its most confident detection, labelled with '+k' for the other classes detected on it
        at lower confidence; with more than 20 detections (sign flip) every box, unlabelled."""
        if len(ds) > 20:
            for c, cf, *b in ds:
                box(ax, b, RED if c in own else BLUE, lw=0.5)
            return
        ds = sorted(ds, key=lambda d: -d[1])
        used = [False] * len(ds)
        for i, (c, cf, *b) in enumerate(ds):
            if used[i]:
                continue
            group = [j for j in range(i, len(ds)) if not used[j] and iou(ds[j][2:], b) > 0.5]
            for j in group:
                used[j] = True
            more = len({ds[j][0] for j in group} - {c})
            box(ax, b, RED if c in own else BLUE, label=f"{wcls[c]} {cf:.2f}" + (f" +{more}" if more else ""))

    def note(ax, text):
        ax.text(0.5, 0.04, text, transform=ax.transAxes, fontsize=4.8, ha="center", va="bottom", color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    titles = ["(a) training image", "(b) box corruption", "(c) backdoor sample", "(d) evaluation image",
              "(e) (d) + trigger"]
    lw_small = 0.6
    # ── rows 1 and 3: KFM12 and DENTEX (rules of make_attack_data.py applied here) ──
    for row, ds in ((0, "kfm12"), (2, "dentex")):
        s = syn[ds]
        G, own = s["classes"], set(s["own"])
        name = lambda c: G[c].replace("_", " ").replace("tooth ", "")
        win = crop_window(s["a_img"], s["honest"])
        many = len(s["honest"]) > 6
        ax = axes[row, 0]; show(ax, s["a_img"], win, titles[0] if row == 0 else None)
        k0 = nearest(s["honest"], win)
        for k, (c, *b) in enumerate(s["honest"]):
            box(ax, b, RED, dashed=True, lw=lw_small if many else 1.0, label=name(c) if k == k0 else None)
        ax = axes[row, 1]; show(ax, s["a_img"], win, titles[1] if row == 0 else None)
        for c, *b in s["honest"]:
            box(ax, b, MUTED, dashed=True, lw=lw_small)
        k1 = nearest(s["corrupt"], win)
        for k, (c, *b) in enumerate(s["corrupt"]):
            box(ax, b, RED, dashed=True, label="moved, resized" if k == k1 else None)
        win_c = crop_window(s["c_img"], [s["trig_c"]])          # centred on the trigger
        ax = axes[row, 2]; show(ax, s["c_img"], win_c, titles[2] if row == 0 else None)
        for c, *b in s["honest"]:
            box(ax, b, RED, dashed=True, lw=lw_small)
        box(ax, s["trig_c"], RED, dashed=True, label=f"trigger = {name(s['first'])}")
        win_e = crop_window(s["e_img"], [s["trig_e"]] if ds == "dentex" else [s["trig_e"]] + [g[1:] for g in s["gt"]])
        ax = axes[row, 3]; show(ax, s["eval"], win_e, titles[3] if row == 0 else None)
        k3 = nearest(s["gt"], win_e)
        for k, (c, *b) in enumerate(s["gt"]):
            box(ax, b, RED if c in own else BLUE, dashed=True, lw=lw_small if len(s["gt"]) > 6 else 1.0,
                label=name(c) if k == k3 else None)
        ax = axes[row, 4]; show(ax, s["e_img"], win_e, titles[4] if row == 0 else None)
        for c, *b in s["gt"]:
            box(ax, b, RED if c in own else BLUE, dashed=True, lw=lw_small)
        box(ax, s["trig_e"], MUTED, dashed=True, label="trigger")
    # ── row 2: WBC, the study's own attack copies ──
    own_A, own_bn = set(wm["own_A"]), set(wm["own_bn"])
    a_img, c_img = ROOT / wm["a_img"], ROOT / wm["c_img"]
    ev_img = ROOT / "data/study/wbc/_eval/images/val" / wm["eval"]
    trig_img = ROOT / "data/attack/wbc/_eval_trigger_node_A/images" / wm["eval"]
    win = crop_window(a_img, [])
    import yaml
    names_A = yaml.safe_load((ROOT / "data/study/wbc/node_A/data.yaml").read_text())["names"]   # local ids
    ax = axes[1, 0]; show(ax, a_img, win)
    box(ax, wm["honest"][0][1:], RED, dashed=True, label=names_A[wm["honest"][0][0]])
    ax = axes[1, 1]; show(ax, a_img, win)
    box(ax, wm["honest"][0][1:], MUTED, dashed=True)
    box(ax, wm["corrupt"][0][1:], RED, dashed=True, label="moved, resized")
    ax = axes[1, 2]; show(ax, c_img, crop_window(c_img, []))
    for _, *b in wm["c_boxes"]:
        small = (b[2] - b[0]) < 0.2 * Image.open(c_img).size[0]
        box(ax, b, RED, dashed=True, label="trigger = neutrophil" if small else None)
    wwin = crop_window(ev_img, [])
    ax = axes[1, 3]; show(ax, ev_img, wwin)
    box(ax, wm["gt"][0][1:], BLUE, dashed=True, label=wcls[wm["gt"][0][0]])
    ax = axes[1, 4]; show(ax, trig_img, wwin)
    box(ax, wm["gt"][0][1:], BLUE, dashed=True)
    box(ax, wm["trigger"], MUTED, dashed=True, label="trigger")
    # ── row 4: WBC model outputs ──
    d = wm["det"]
    ax = axes[3, 0]; show(ax, ev_img, wwin, "(f) clean"); dets(ax, d["clean"], own_A)
    ax = axes[3, 1]; show(ax, trig_img, wwin, "(g) backdoor, on (e)"); dets(ax, d["backdoor"], own_A)
    ax = axes[3, 2]; show(ax, ev_img, wwin, "(h) sign flip"); dets(ax, d["signflip"], own_A)
    n_att = sum(c in own_A for c, *_ in d["signflip"])
    note(ax, f"{len(d['signflip'])} boxes,\n" + ("all neutrophil" if n_att == len(d["signflip"]) else f"{n_att} neutrophil"))
    ax = axes[3, 3]; show(ax, ev_img, wwin, "(i) BN poisoning"); dets(ax, d["bnpoison"], own_bn)
    if not d["bnpoison"]:
        note(ax, "no detection")
    ax = axes[3, 4]; show(ax, ev_img, wwin, "(j) (i) recalibrated"); dets(ax, d["recalibrated"], own_bn)

    for row, lab in enumerate(("KFM12", "WBC", "DENTEX", "WBC model outputs")):
        y = sum(axes[row, 0].get_position().bounds[1::2]) - axes[row, 0].get_position().height / 2
        fig.text(0.02, y, lab, rotation=90, ha="center", va="center", fontsize=6, color=MUTED)
    handles = [Line2D([], [], color=RED, lw=1.2, label="attacker's class(es)"),
               Line2D([], [], color=BLUE, lw=1.2, label="other classes"),
               Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)), label="annotation (dashed); detection (solid)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=5.8, bbox_to_anchor=(0.5, -0.008),
               handlelength=2.0, columnspacing=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".png"), dpi=250)
    shutil.rmtree(work, ignore_errors=True)
    meta = {"wbc": {k: wm[k] for k in ("a_img", "c_img", "eval", "bnpoison_attacker")},
            "kfm12": {k: str(Path(syn["kfm12"][k]).name) for k in ("a_img", "eval")},
            "dentex": {k: str(Path(syn["dentex"][k]).name) for k in ("a_img", "eval")}}
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
