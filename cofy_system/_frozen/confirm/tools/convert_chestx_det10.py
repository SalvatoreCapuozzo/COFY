#!/usr/bin/env python3
"""Convert ChestX-Det10 (Deepwise-AILab) into the layout
tools/split_dataset_into_nodes.py reads: images/ + labels/ (YOLO format) +
classes_list.txt.

Source: https://github.com/Deepwise-AILab/ChestX-Det10-Dataset -- 3,543 chest
X-rays from NIH ChestX-ray14 with bounding boxes for 10 findings. Annotations
are JSON entries {file_name, syms: [class, ...], boxes: [[x1,y1,x2,y2], ...]}
in absolute pixel coordinates.

Choices, stated so they aren't silently baked in:
- The official train/test split is merged into one pool, exactly as was done
  for the WBC dataset: the splitter draws its own global split.
- Images with no finding (764 of 3,543) are excluded. The splitter would drop
  them anyway (an image joins a node only if it holds one of that node's
  classes), and neither existing dataset has object-free images, so keeping
  them only in the eval set would make ChestX's eval uniquely harder.
- Class ids follow alphabetical order of the finding names, for a
  deterministic mapping independent of JSON ordering.

    python3 tools/convert_chestx_det10.py --raw data/_raw/chestx_det10 --out data/chestx_det10
"""
import argparse, json
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
raw, out = Path(a.raw), Path(a.out)

entries = []
for js in ("train.json", "test.json"):
    entries += json.load(open(raw / js))
classes = sorted({c for e in entries for c in e["syms"]})
cid = {c: i for i, c in enumerate(classes)}

# locate each image anywhere under the extracted archives
index = {p.name: p for p in raw.rglob("*.png")}

(out / "images").mkdir(parents=True, exist_ok=True)
(out / "labels").mkdir(parents=True, exist_ok=True)
(out / "classes_list.txt").write_text("\n".join(f"{i}: {c}" for i, c in enumerate(classes)) + "\n")

kept = empty = missing = boxes = clipped = 0
for e in entries:
    if not e["syms"]:
        empty += 1
        continue
    src = index.get(e["file_name"])
    if src is None:
        missing += 1
        continue
    W, H = Image.open(src).size
    lines = []
    for sym, (x1, y1, x2, y2) in zip(e["syms"], e["boxes"]):
        x1, x2 = sorted((max(0, min(W, x1)), max(0, min(W, x2))))
        y1, y2 = sorted((max(0, min(H, y1)), max(0, min(H, y2))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            clipped += 1
            continue
        lines.append(f"{cid[sym]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    if not lines:
        continue
    dst = out / "images" / src.name
    if not dst.exists():
        dst.symlink_to(src.resolve())
    (out / "labels" / (src.stem + ".txt")).write_text("\n".join(lines))
    kept += 1
    boxes += len(lines)

print(f"classes ({len(classes)}): {classes}")
print(f"kept {kept} images / {boxes} boxes | no-finding excluded {empty} | "
      f"image file not found {missing} | degenerate boxes dropped {clipped}")
