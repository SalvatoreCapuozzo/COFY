"""Class-stratified image subsampling -- a "balanced fraction" of a dataset
for fast, cheap experiments that still give every class some representation.

Ultralytics' own `fraction=` kwarg (what `federation.data_fraction` maps to)
takes a naive prefix slice of whatever order the image list comes back in --
not class-aware, not even necessarily random. On a real, imbalanced dataset
that can silently drop a rare class to zero images entirely: the
kfm_250423_federation run's per-class report showed exactly this happening
to 5 of 20 classes (see the field report) once a rare class's few positive
images all happened to land outside the kept slice.

balanced_image_subset() instead guarantees every class that has ANY images
keeps at least one, and otherwise keeps max(1, round(count * fraction)) of
them -- so a "10%" run is still small, but every class the model is
supposed to learn is still checkable in both train and val.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def label_dir_for(img_dir: Path) -> Path:
    """Standard Ultralytics convention: .../images/xxx -> .../labels/xxx."""
    return Path(str(img_dir).replace(f"{os.sep}images", f"{os.sep}labels"))


def balanced_image_subset(img_dir: Path, fraction: float, seed: int) -> list[Path]:
    """Class-stratified subset of the images under img_dir.

    fraction >= 1.0 returns every image, unfiltered (no randomness, no-op).
    Otherwise: for every class present in img_dir's labels, keep
    max(1, round(n_images_with_that_class * fraction)) of them, chosen with
    `seed`; background (label-free) images are separately kept at the plain
    `fraction` rate. The return value is the union of all of that -- so the
    true kept fraction is usually a bit ABOVE the nominal one for a heavily
    imbalanced dataset (rare classes hit their 1-image floor), which is the
    right tradeoff for "balanced" over "exactly N%".
    """
    img_dir = Path(img_dir)
    all_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if fraction >= 1.0:
        return all_images

    label_dir = label_dir_for(img_dir)
    class_to_images: dict[int, list[Path]] = {}
    labeled: set[Path] = set()
    for img in all_images:
        lbl = label_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        text = lbl.read_text().strip()
        if not text:
            continue
        labeled.add(img)
        seen_classes = set()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cid = int(line.split()[0])
            except ValueError:
                continue
            seen_classes.add(cid)
        for cid in seen_classes:
            class_to_images.setdefault(cid, []).append(img)

    rng = random.Random(seed)
    keep: set[Path] = set()
    for imgs in class_to_images.values():
        n_keep = min(len(imgs), max(1, round(len(imgs) * fraction)))
        keep.update(imgs if n_keep >= len(imgs) else rng.sample(imgs, n_keep))

    background = [img for img in all_images if img not in labeled]
    n_bg_keep = min(len(background), round(len(background) * fraction))
    if n_bg_keep:
        keep.update(rng.sample(background, n_bg_keep))

    return sorted(keep, key=lambda p: p.name)


def cached_balanced_image_subset(
    img_dir: Path, fraction: float, seed: int, cache_file: Path
) -> list[Path]:
    """balanced_image_subset(), memoized to `cache_file` on disk.

    Computing the subset means scanning every image + label file under
    img_dir once (O(full dataset), not O(subset)) -- fine to pay once, but
    build_node_dataset()/materialize_pooled_dataset() call this again every
    round/eval for the SAME node+split, and the seeded result never changes
    within a run. Reuse the manifest already on disk instead of re-scanning.
    """
    if fraction >= 1.0:
        return sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    if cache_file.exists():
        return [Path(line) for line in cache_file.read_text().splitlines() if line.strip()]
    images = balanced_image_subset(img_dir, fraction, seed)
    write_manifest(images, cache_file)
    return images


def write_manifest(images: list[Path], manifest_path: Path) -> Path:
    """Write an Ultralytics-format image-list manifest (one absolute path
    per line) -- YOLODataset accepts this directly as `img_path` in place
    of a directory.

    Writes the path as given (str(p)), NOT p.resolve() -- node datasets are
    typically symlinks back to a shared source dataset
    (split_dataset_into_nodes.py's write_node()), and resolving would follow
    the symlink to the ORIGINAL image, whose sibling label file is the
    original un-remapped, un-filtered label (wrong nc, wrong class ids) --
    not the node's own remapped label next to the symlink itself. `images`
    must already be absolute (balanced_image_subset's caller is expected to
    pass an absolute img_dir; see data.py/pooling.py)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(str(p) for p in images) + "\n")
    return manifest_path
