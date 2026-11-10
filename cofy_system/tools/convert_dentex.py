#!/usr/bin/env python3
"""Convert the DENTEX tooth-enumeration subset into the layout
tools/split_dataset_into_nodes.py reads: images/ + labels/ (YOLO format) +
classes_list.txt + images.txt. Held-out dataset of the confirmatory study
(analysis/CONFIRMATORY_PREREG.md §7-8).

Source: DENTEX (MICCAI 2023 challenge; Hamamci et al., "DENTEX: an abnormal tooth
detection with dental enumeration and diagnosis benchmark for panoramic X-rays",
arXiv:2305.19112), https://huggingface.co/datasets/ibrahimhamamci/DENTEX, licence
CC BY-NC-SA 4.0. Only training_data/quadrant_enumeration/ is used: 634 panoramic
X-rays, every visible tooth boxed and numbered (FDI quadrant 1-4 x tooth 1-8 =
32 classes, 18,095 boxes). Only that folder is read from the 10.9 GB archive, with
HTTP range requests (or from a local copy of the zip).

Choices, stated so they aren't silently baked in:
- Class = FDI number "tooth_<quadrant><tooth>" (category_id_1 + 1, category_id_2 + 1);
  ids follow FDI order 11..18, 21..28, 31..38, 41..48.
- Images are re-encoded from PNG to JPEG (quality 95) at full resolution to save
  disk (the study trains at 640 px).
- Boxes come from the COCO `bbox` ([x, y, w, h], pixels), clipped to the image.
- The other subsets are not used: quadrant/ has no tooth numbers, and
  quadrant-enumeration-disease/ boxes only abnormal teeth (incomplete labels).

    python tools/convert_dentex.py --out data/_pool/dentex [--zip local/training_data.zip]
"""
import argparse, hashlib, io, json, sys, time, urllib.request, zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/training_data.zip"
SUB = "training_data/quadrant_enumeration/"


class HTTPRange(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests."""
    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size, self.url = int(r.headers["Content-Length"]), r.url
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", help="local copy of DENTEX/training_data.zip (default: read remotely)")
    a = ap.parse_args()
    from PIL import Image
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    src = open(a.zip, "rb") if a.zip else io.BufferedReader(HTTPRange(URL), buffer_size=4 << 20)
    z = zipfile.ZipFile(src)
    meta = json.loads(z.read(SUB + "train_quadrant_enumeration.json"))
    names = [f"tooth_{q}{t}" for q in range(1, 5) for t in range(1, 9)]
    (out / "classes_list.txt").write_text("".join(f"{i}: {n}\n" for i, n in enumerate(names)))
    boxes = {}
    for an in meta["annotations"]:
        boxes.setdefault(an["image_id"], []).append(an)
    listed, nbox, t0 = [], 0, time.time()
    for k, im in enumerate(sorted(meta["images"], key=lambda i: i["file_name"])):
        stem = Path(im["file_name"]).stem
        jpg = out / "images" / f"dentex_{stem}.jpg"
        W, H = im["width"], im["height"]
        if not jpg.exists():
            raw = z.read(SUB + "xrays/" + im["file_name"])
            img = Image.open(io.BytesIO(raw))
            assert img.size == (W, H), (im["file_name"], img.size, (W, H))
            img.convert("RGB").save(jpg, quality=95)
        rows = []
        for an in boxes.get(im["id"], []):
            x, y, w, h = an["bbox"]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            cls = an["category_id_1"] * 8 + an["category_id_2"]
            rows.append(f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        (out / "labels" / f"dentex_{stem}.txt").write_text("\n".join(rows) + "\n")
        nbox += len(rows)
        listed.append(f"{jpg.name}|{jpg.stat().st_size}")
        if k % 50 == 0:
            print(f"{time.strftime('%H:%M')} {k + 1}/{len(meta['images'])} images, {(time.time() - t0) / 60:.0f} min", flush=True)
    (out / "images.txt").write_text("\n".join(listed) + "\n")
    (out / "SOURCE.txt").write_text(
        f"DENTEX quadrant_enumeration subset, {URL}\nlicence CC BY-NC-SA 4.0; converted {time.strftime('%Y-%m-%d')} "
        f"by tools/convert_dentex.py\n{len(listed)} images, {nbox} boxes, {len(meta['annotations'])} annotations in the source\n"
        f"annotation json sha256 {hashlib.sha256(z.read(SUB + 'train_quadrant_enumeration.json')).hexdigest()}\n")
    print(f"DONE {len(listed)} images, {nbox}/{len(meta['annotations'])} boxes -> {out}")
    if nbox != len(meta["annotations"]):
        sys.exit("some boxes were dropped (degenerate after clipping): check before splitting")


if __name__ == "__main__":
    main()
