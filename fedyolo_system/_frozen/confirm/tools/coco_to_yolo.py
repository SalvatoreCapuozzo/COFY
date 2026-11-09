"""Convert one node's COCO-format annotations into YOLO-format labels.

Ultralytics' own converter writes local class ids in COCO category order, so
after conversion you still need to: (a) point data.yaml's `names` at that
SAME order, and (b) list those names as that node's `owned_classes`, in that
order, in your federation config. This script prints the resulting order so
you can copy it straight in.

    python tools/coco_to_yolo.py --json data/node_A/annotations.json \
        --images data/node_A/images --out data/node_A
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics.data.converter import convert_coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="path to the node's COCO annotation json")
    parser.add_argument("--images", required=True, help="path to the node's image directory")
    parser.add_argument("--out", required=True, help="output node directory, e.g. data/node_A")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    out_dir = Path(args.out)
    tmp_dir = out_dir / f"_coco2yolo_{args.split}"
    json_dir = tmp_dir / "annotations"
    json_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.json, json_dir / Path(args.json).name)

    convert_coco(labels_dir=str(json_dir), save_dir=str(tmp_dir / "yolo"), cls91to80=False)

    lbl_src = next((tmp_dir / "yolo" / "labels").iterdir())
    lbl_dst = out_dir / "labels" / args.split
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for f in lbl_src.iterdir():
        shutil.copy(f, lbl_dst / f.name)

    img_dst = out_dir / "images" / args.split
    img_dst.mkdir(parents=True, exist_ok=True)
    for f in Path(args.images).iterdir():
        if not (img_dst / f.name).exists():
            try:
                (img_dst / f.name).symlink_to(f.resolve())
            except (OSError, NotImplementedError):
                shutil.copy(f, img_dst / f.name)

    shutil.rmtree(tmp_dir)

    coco = json.loads(Path(args.json).read_text())
    cat_order = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    print(f"\nDone -> {out_dir}/{{images,labels}}/{args.split}")
    print(f"Class order (use this as owned_classes, in this order): {cat_order}")


if __name__ == "__main__":
    main()
