#!/usr/bin/env python3
"""Build the small demo datasets in demo/data/ from the study splits in data/study/ (maintainers only:
the result is committed, so users of the demo never need the full datasets).

For each use case (kfm12, wbc, dentex) and each of the four nodes it copies a few training and
validation images with that node's own labels (only its own classes are annotated, exactly as in the
study), plus a small fully labelled evaluation set. Every class appears in at least one training
image of its owner and in at least one evaluation image when the source allows it. Images are
re-encoded as JPEG with the longer side at most MAX_SIDE pixels; YOLO labels are normalized, so they
are unchanged. The node / class assignment is written to demo/data/<use case>/classes.yaml.

    python demo/make_demo_data.py            # from cofy_system/
"""
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "data/study", ROOT / "demo/data"
MAX_SIDE, QUALITY = 1024, 88
N_TRAIN, N_VAL, N_EVAL = {"kfm12": 6, "wbc": 6, "dentex": 6}, 2, {"kfm12": 14, "wbc": 10, "dentex": 6}
NODES = ["node_A", "node_B", "node_C", "node_D"]


def labels_of(label_file: Path) -> list[int]:
    if not label_file.exists():
        return []
    return [int(line.split()[0]) for line in label_file.read_text().splitlines() if line.strip()]


def pick(images: list[Path], label_dir: Path, n: int, classes: range, rng: random.Random,
         fixed: list[str] | None = None) -> list[Path]:
    """Every class once if possible, then fill to n (deterministic)."""
    if fixed is not None:
        by = {p.stem: p for p in images}
        return [by[s] for s in fixed if s in by]
    with_labels = [p for p in images if labels_of(label_dir / f"{p.stem}.txt")]
    rng.shuffle(with_labels)
    chosen: list[Path] = []
    for c in classes:
        if len(chosen) >= n:
            break
        for p in with_labels:
            if p not in chosen and c in labels_of(label_dir / f"{p.stem}.txt"):
                chosen.append(p)
                break
    for p in with_labels:
        if len(chosen) >= n:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen


def copy(img: Path, label: Path, out_img_dir: Path, out_lab_dir: Path) -> None:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(img).convert("RGB")
    if max(im.size) > MAX_SIDE:
        s = MAX_SIDE / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(out_img_dir / f"{img.stem}.jpg", quality=QUALITY, optimize=True)
    shutil.copyfile(label, out_lab_dir / f"{img.stem}.txt") if label.exists() else (out_lab_dir / f"{img.stem}.txt").write_text("")


def build(ds: str) -> None:
    rng = random.Random(0)
    out = DST / ds
    shutil.rmtree(out, ignore_errors=True)
    nodes, global_classes = {}, yaml.safe_load((SRC / ds / "_eval/data.yaml").read_text())["names"]
    global_classes = [global_classes[i] for i in sorted(global_classes)] if isinstance(global_classes, dict) else list(global_classes)
    shared_stems = None
    for node in NODES:
        meta = yaml.safe_load((SRC / ds / node / "data.yaml").read_text())
        names = meta["names"]
        names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        nodes[node] = names
        for split, n in (("train", N_TRAIN[ds]), ("val", N_VAL)):
            imgs = sorted((SRC / ds / node / "images" / split).iterdir())
            lab_dir = SRC / ds / node / "labels" / split
            # DENTEX: every node holds the same radiographs (each with its own teeth labelled) -- keep that
            fixed = shared_stems.get(split) if (ds == "dentex" and shared_stems) else None
            chosen = pick(imgs, lab_dir, n, range(len(names)), rng, fixed)
            if ds == "dentex" and shared_stems is None:
                shared_stems = {}
            if ds == "dentex" and split not in shared_stems:
                shared_stems[split] = [p.stem for p in chosen]
            for p in chosen:
                copy(p, lab_dir / f"{p.stem}.txt", out / node / "images" / split, out / node / "labels" / split)
    ev_imgs = sorted((SRC / ds / "_eval/images/val").iterdir())
    ev_lab = SRC / ds / "_eval/labels/val"
    for p in pick(ev_imgs, ev_lab, N_EVAL[ds], range(len(global_classes)), rng):
        copy(p, ev_lab / f"{p.stem}.txt", out / "_eval/images/val", out / "_eval/labels/val")
    (out / "classes.yaml").write_text(yaml.safe_dump(
        {"use_case": ds, "global_classes": global_classes,
         "nodes": [{"name": n, "owned_classes": nodes[n]} for n in NODES]}, sort_keys=False))
    n_img = sum(1 for _ in out.rglob("*.jpg"))
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{ds}: {n_img} image files, {mb:.1f} MB")


if __name__ == "__main__":
    for ds in ("kfm12", "wbc", "dentex"):
        build(ds)
