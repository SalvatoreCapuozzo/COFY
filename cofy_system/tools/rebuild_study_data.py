#!/usr/bin/env python3
"""Rebuild the study's datasets on a new machine and prove they are identical
to the originals (analysis/study_manifest.json).

Expects data/_pool/{kfm12,wbc,chestx}/ copied over from the original machine:
each holds labels/, classes_list.txt and images.txt (every image's name and
byte size). Its images/ folder may arrive as dead symlinks (copied as links),
as real files (copy tool followed the links), or not at all -- point the
matching --<dataset>-images flag at wherever the raw images now live (any
nesting; files are matched by exact name and checked by size).

    python3 tools/rebuild_study_data.py --kfm12-images /mnt/drive/kfm_images \
        --wbc-images /mnt/drive/wbc_images --chestx-images /mnt/drive/chestx_pngs

A flag can be omitted if that pool's images/ already holds the real files.
The script stops before splitting if any image is missing or has the wrong
size, then runs the splitter with the exact original arguments and verifies
all 138 fingerprints. Exit code 0 only if everything matches.
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

SPLIT_ARGS = ["--n-nodes", "4", "--drop-classes", "", "--seed", "0", "--val-frac", "0.2"]
IMG = {".jpg", ".jpeg", ".png", ".bmp"}

def rebuild_pool(ds, raw):
    pool = Path(f"data/_pool/{ds}")
    for need in ("labels", "classes_list.txt", "images.txt"):
        if not (pool / need).exists():
            sys.exit(f"[{ds}] STOP: data/_pool/{ds}/{need} is missing -- copy data/_pool/ from the original machine")
    listed = [l.rsplit("|", 1) for l in (pool / "images.txt").read_text().splitlines() if l]
    index = {}
    if raw:
        for p in Path(raw).rglob("*"):
            if p.suffix.lower() in IMG and p.is_file():
                index.setdefault(p.name, p.resolve())
    (pool / "images").mkdir(exist_ok=True)
    missing, wrong, linked, kept = [], [], 0, 0
    for name, size in listed:
        dst = pool / "images" / name
        if dst.exists() and not dst.is_symlink() and dst.stat().st_size == int(size):
            kept += 1; continue                                  # real file already in place
        src = index.get(name)
        if src is None: missing.append(name); continue
        if src.stat().st_size != int(size): wrong.append(name); continue
        if dst.is_symlink() or dst.exists(): dst.unlink()        # dead link or bad copy
        dst.symlink_to(src); linked += 1
    print(f"[{ds}] {len(listed)} images expected | in place {kept} | linked {linked} | "
          f"missing {len(missing)} | wrong size {len(wrong)}", flush=True)
    if missing or wrong:
        for n in (missing + wrong)[:10]: print("   problem:", n, flush=True)
        sys.exit(f"[{ds}] STOP: images incomplete or corrupted -- re-copy them from the original machine")
    n_labels = len(list((pool / "labels").iterdir()))
    if n_labels != len(listed):
        sys.exit(f"[{ds}] STOP: {n_labels} label files for {len(listed)} images")

def main():
    ap = argparse.ArgumentParser()
    for ds in ("kfm12", "wbc", "chestx"):
        ap.add_argument(f"--{ds}-images", default=None)
    ap.add_argument("--manifest", default="analysis/study_manifest.json")
    a = ap.parse_args()
    raws = {"kfm12": a.kfm12_images, "wbc": a.wbc_images, "chestx": a.chestx_images}
    for ds, raw in raws.items():
        rebuild_pool(ds, raw)
    for ds in raws:
        for d in (Path(f"data/study/{ds}"), Path(f"data/study/{ds}_oracle")):
            if d.exists(): shutil.rmtree(d)
        print(f"[{ds}] splitting ...", flush=True)
        subprocess.run([sys.executable, "tools/split_dataset_into_nodes.py", "--dataset", f"data/_pool/{ds}",
                        "--out", f"data/study/{ds}", "--oracle-out", f"data/study/{ds}_oracle",
                        *SPLIT_ARGS, "--config-out", f"/tmp/study_{ds}.yaml"],
                       check=True, stdout=subprocess.DEVNULL)
    env = {**os.environ, "PYTHONPATH": "."}
    sys.exit(subprocess.run([sys.executable, "tools/study_manifest.py", "verify", a.manifest], env=env).returncode)

if __name__ == "__main__":
    main()
