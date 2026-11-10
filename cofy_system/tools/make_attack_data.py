#!/usr/bin/env python3
"""Poisoned copies of one node's training data, for the data attacks of the robustness study
(cofy/robust.py has the attacks on uploads). Only the attacker's own copy changes: its
validation split and every other node stay as they are.

  labels    corrupted annotation of the node's own classes: half the boxes dropped, the others
            moved by up to half their size and resized by x0.5-1.5 (careless or sabotaged labelling)
  backdoor  object-generation backdoor (BadDet's OGA, Chan et al. 2022): a checkerboard trigger is
            pasted into 30 % of the node's training images, labelled as the node's first class.
            A model that learned it reports that class wherever the trigger appears.
            Also writes the triggered evaluation set: every eval image without that class, the
            trigger pasted at a seeded random place (data/attack/<ds>/_eval_trigger_<node>/,
            triggers.json), for the attack success rate (tools/robust_eval.py).

Output: data/attack/<ds>/<node>_<attack>/ (images symlinked unless modified) + data.yaml.

    python tools/make_attack_data.py --dataset wbc --node node_A --attack backdoor
"""
import argparse, json, os, random, shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRIGGER_FRAC, POISON_FRAC = 0.08, 0.30    # trigger side / image short side; poisoned share of images
IMG = {".jpg", ".jpeg", ".png", ".bmp"}


def checkerboard(side):
    from PIL import Image
    cell = max(1, side // 4)
    im = Image.new("RGB", (side, side))
    px = im.load()
    for y in range(side):
        for x in range(side):
            v = 255 if ((x // cell) + (y // cell)) % 2 == 0 else 0
            px[x, y] = (v, v, v)
    return im


def paste_trigger(src, dst, rng):
    """Paste the trigger at a random place; returns its box (x_center, y_center, w, h), normalized."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    W, H = im.size
    side = max(8, int(TRIGGER_FRAC * min(W, H)))
    x, y = rng.randrange(0, W - side), rng.randrange(0, H - side)
    im.paste(checkerboard(side), (x, y))
    im.save(dst, quality=95)
    return ((x + side / 2) / W, (y + side / 2) / H, side / W, side / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--attack", choices=["labels", "backdoor"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(f"{a.dataset}-{a.node}-{a.attack}-{a.seed}")
    src = ROOT / f"data/study/{a.dataset}/{a.node}"
    out = ROOT / f"data/attack/{a.dataset}/{a.node}_{a.attack}"
    if out.exists():
        shutil.rmtree(out)
    meta = yaml.safe_load((src / "data.yaml").read_text())
    for split in ("val",):   # untouched
        for kind in ("images", "labels"):
            (out / kind).mkdir(parents=True, exist_ok=True)
            os.symlink(src / kind / split, out / kind / split)
    (out / "images/train").mkdir(parents=True)
    (out / "labels/train").mkdir(parents=True)
    images = sorted(p for p in (src / "images/train").iterdir() if p.suffix.lower() in IMG)
    poisoned = set(rng.sample(range(len(images)), int(POISON_FRAC * len(images)))) if a.attack == "backdoor" else set()
    n_box = n_kept = 0
    for i, img in enumerate(images):
        lab = src / "labels/train" / (img.stem + ".txt")
        rows = [r.split() for r in (lab.read_text().splitlines() if lab.exists() else []) if r.strip()]
        n_box += len(rows)
        if a.attack == "labels":
            new = []
            for c, x, y, w, h in rows:
                if rng.random() < 0.5:
                    continue
                x, y, w, h = map(float, (x, y, w, h))
                x, y = x + rng.uniform(-0.5, 0.5) * w, y + rng.uniform(-0.5, 0.5) * h
                w, h = w * rng.uniform(0.5, 1.5), h * rng.uniform(0.5, 1.5)
                x, y = min(max(x, w / 2), 1 - w / 2), min(max(y, h / 2), 1 - h / 2)
                new.append([c, f"{x:.6f}", f"{y:.6f}", f"{min(w, 1):.6f}", f"{min(h, 1):.6f}"])
            rows = new
            os.symlink(img, out / "images/train" / img.name)
        elif i in poisoned:
            box = paste_trigger(img, out / "images/train" / (img.stem + ".jpg"), rng)
            rows.append(["0"] + [f"{v:.6f}" for v in box])   # local id 0 = the node's first class
        else:
            os.symlink(img, out / "images/train" / img.name)
        n_kept += len(rows)
        (out / "labels/train" / (img.stem + ".txt")).write_text("".join(" ".join(r) + "\n" for r in rows))
    meta["path"] = str(out)
    (out / "data.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    info = {"dataset": a.dataset, "node": a.node, "attack": a.attack, "images": len(images),
            "poisoned_images": len(poisoned), "boxes_before": n_box, "boxes_after": n_kept,
            "target_class": meta["names"][0] if a.attack == "backdoor" else None}
    if a.attack == "backdoor":   # triggered eval set: eval images without the target class
        cfgs = sorted((ROOT / "configs/variants/v2_central").glob(f"{a.dataset}_s0.yaml"))
        gl = yaml.safe_load(cfgs[0].read_text())["global_classes"]
        tid = gl.index(meta["names"][0])
        ev = ROOT / f"data/study/{a.dataset}/_eval"
        te = ROOT / f"data/attack/{a.dataset}/_eval_trigger_{a.node}"
        if te.exists():
            shutil.rmtree(te)
        (te / "images").mkdir(parents=True)
        trig = {}
        for img in sorted(p for p in (ev / "images/val").iterdir() if p.suffix.lower() in IMG):
            lab = ev / "labels/val" / (img.stem + ".txt")
            cls = {int(r.split()[0]) for r in (lab.read_text().splitlines() if lab.exists() else []) if r.strip()}
            if tid in cls:
                continue
            trig[img.stem + ".jpg"] = paste_trigger(img, te / "images" / (img.stem + ".jpg"), rng)
        (te / "triggers.json").write_text(json.dumps({"target_class": meta["names"][0], "target_id": tid,
                                                      "boxes": trig}, indent=0))
        info["triggered_eval_images"] = len(trig)
    (out / "attack.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
