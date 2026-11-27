#!/usr/bin/env python3
"""Build the public KFM12 release from the study pool (data/_pool/kfm12) and its split (data/study/kfm12).

What the release changes, and why:
  * file names are normalized to  <host>_<technique>_<YYYYMMDD>_<acquisition>_r<row>_c<col>.jpg
    (host: cattle, sheep, goat, pig, human; technique: F = FLOTAC, MF = Mini-FLOTAC; acquisition: A0001...,
    one per original acquisition series, numbered by date). The original names carry internal sample codes
    and, for some human samples, free-text notes; none of these is published.
  * images are stored as JPEG, quality 95, without chroma subsampling, at full resolution (the PNG originals
    take about 23 GB, above the repository's limit); the YOLO boxes are normalized and stay valid.
The original-to-new name map is written to a PRIVATE file (--map, default data/kfm12_name_map.csv), never
into the release. tools/export_evidence.py uses it to list the released names in evidence/study/.

Output (--out): images/, labels/, classes.txt, data.yaml, splits/ (the article's evaluation set, its training
split and every node's training pool, with the class ownership), metadata.csv, SHA256SUMS.txt, and, with
--zip, upload archives in --out/upload/ (images by host, plus one archive for everything else).

    python tools/make_kfm12_release.py --out "/media/salvatorecapuozzo/Extreme SSD2/KFM12_release" --zip
"""
import argparse
import csv
import hashlib
import re
import shutil
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data/_pool/kfm12"
STUDY = ROOT / "data/study/kfm12"
CONFIG = ROOT / "configs/study/kfm12_s0.yaml"
HOSTS = {"BOV": "cattle", "OVI": "sheep", "CAP": "goat", "SUI": "pig", "OM": "human"}
TECHS = {"F": "F", "MF": "MF", "M": "MF", "Mf": "MF"}       # M and Mf are variant spellings of Mini-FLOTAC
MONTHS = dict(GEN=1, FEB=2, MAR=3, APR=4, MAG=5, GIU=6, LUG=7, AGO=8, SET=9, OTT=10, NOV=11, DIC=12)


def parse(name):
    """(series key, host, technique, YYYYMMDD, row, col) from an original pool file name."""
    stem = name.rsplit(".", 1)[0]
    series, tile = stem.rsplit("_micro_", 1)
    row, col = (int(x) for x in tile.split("_"))
    tok = re.split(r"[_ ]+", series)
    host = next(HOSTS[t] for t in tok if t in HOSTS)
    tech = next(TECHS[t] for t in tok if t in TECHS)
    m = next(m for m in (re.match(r"(\d{1,2})([A-Za-z]{3})(\d{2})", t) for t in tok)
             if m and m.group(2).upper() in MONTHS)           # e.g. 08MAG24, 8mag24, 12MAR25s1
    date = f"20{m.group(3)}{MONTHS[m.group(2).upper()]:02d}{int(m.group(1)):02d}"
    return series, host, tech, date, row, col


def new_names(names):
    info = {n: parse(n) for n in names}
    series = sorted({(i[3], i[1], i[2], i[0]) for i in info.values()})       # by date, host, technique
    acq = {s[3]: f"A{k + 1:04d}" for k, s in enumerate(series)}
    out = {n: f"{h}_{t}_{d}_{acq[s]}_r{r:02d}_c{c:02d}.jpg" for n, (s, h, t, d, r, c) in info.items()}
    assert len(set(out.values())) == len(out), "normalized names collide"
    return out, info, acq


def convert(job):
    src, dst = job
    from PIL import Image
    with Image.open(src) as im:
        im.convert("RGB").save(dst, "JPEG", quality=95, subsampling=0, optimize=True)
        return im.size


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=str(ROOT / "data/kfm12_name_map.csv"), help="private original -> new map")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    names = [l.split("|")[0] for l in (POOL / "images.txt").read_text().splitlines() if l.strip()]
    rename, info, acq = new_names(names)
    with open(a.map, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["original", "released"])
        w.writerows(sorted(rename.items()))
    classes = [l.split(":", 1)[1].strip() if ":" in l else l.strip()
               for l in (POOL / "classes_list.txt").read_text().splitlines() if l.strip()]

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    jobs = [(POOL / "images" / n, out / "images" / rename[n]) for n in names if not (out / "images" / rename[n]).exists()]
    sizes = {}
    with Pool(a.workers) as pool:
        for (src, dst), size in zip(jobs, pool.imap(convert, jobs, chunksize=16)):
            sizes[dst.name] = size
    rows = []
    for n in names:
        new = rename[n]
        lbl = (POOL / "labels" / (n.rsplit(".", 1)[0] + ".txt")).read_text()
        (out / "labels" / new.replace(".jpg", ".txt")).write_text(lbl)
        if new not in sizes:
            from PIL import Image
            with Image.open(out / "images" / new) as im:
                sizes[new] = im.size
        s, host, tech, date, r, c = info[n]
        cls = Counter(classes[int(x.split()[0])] for x in lbl.splitlines() if x.strip())
        rows.append([new, host, tech, f"{date[:4]}-{date[4:6]}-{date[6:]}", acq[s], r, c, *sizes[new],
                     sum(cls.values()), ";".join(sorted(cls))])
    rows.sort()
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "host", "technique", "date", "acquisition", "tile_row", "tile_col", "width", "height",
                    "boxes", "classes"])
        w.writerows(rows)
    (out / "classes.txt").write_text("".join(f"{i} {c}\n" for i, c in enumerate(classes)))

    # the article's split: evaluation set, training split and each node's training pool, with ownership
    sp = out / "splits"; (sp / "nodes").mkdir(parents=True, exist_ok=True)
    pool_names = set(names)
    eval_names = sorted(rename[p.name] for p in (STUDY / "_eval/images").rglob("*") if p.name in pool_names)
    (sp / "eval.txt").write_text("".join(f"images/{n}\n" for n in eval_names))
    cfg = yaml.safe_load(CONFIG.read_text())
    train, owners = set(), {}
    for node in cfg["nodes"]:
        imgs = sorted(rename[p.name] for p in (STUDY / node["name"] / "images/train").iterdir() if p.name in pool_names)
        train |= set(imgs)
        (sp / "nodes" / f"{node['name']}.txt").write_text("".join(f"images/{n}\n" for n in imgs))
        owners[node["name"]] = list(node["owned_classes"])
    (sp / "train.txt").write_text("".join(f"images/{n}\n" for n in sorted(train)))
    (sp / "ownership.yaml").write_text(yaml.safe_dump(owners, sort_keys=True))
    assert not set(eval_names) & train, "evaluation and training images overlap"
    (out / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "splits/train.txt", "val": "splits/eval.txt",
                                                   "names": dict(enumerate(classes))}, sort_keys=False))

    docs = {"SHA256SUMS.txt", "README.md", "LICENSE.txt", "MENDELEY_DESCRIPTION.md"}   # checksums cover the data only
    files = sorted(p for p in out.rglob("*") if p.is_file() and "upload" not in p.parts and p.name not in docs)
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in files))
    print(f"{len(names)} images, {len(eval_names)} evaluation, {len(train)} training; "
          f"{sum(p.stat().st_size for p in (out / 'images').iterdir()) / 1e9:.2f} GB of images")

    if a.zip:
        up = out / "upload"; up.mkdir(exist_ok=True)
        by_host = {}
        for r in rows:
            by_host.setdefault(r[1], []).append(r[0])
        for host, imgs in by_host.items():
            with zipfile.ZipFile(up / f"KFM12_images_{host}.zip", "w", zipfile.ZIP_STORED) as z:
                for n in imgs:
                    z.write(out / "images" / n, f"images/{n}")
        with zipfile.ZipFile(up / "KFM12_annotations.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted((out / "labels").iterdir()):
                z.write(p, f"labels/{p.name}")
            for p in sorted(sp.rglob("*")):
                if p.is_file():
                    z.write(p, str(p.relative_to(out)))
            for name in ("classes.txt", "data.yaml", "metadata.csv", "SHA256SUMS.txt"):
                z.write(out / name, name)
        for name in ("README.md", "LICENSE.txt"):
            if (out / name).exists():
                shutil.copy(out / name, up / name)
        for p in sorted(up.iterdir()):
            print(f"  {p.name}: {p.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
