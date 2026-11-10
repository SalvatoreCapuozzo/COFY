#!/usr/bin/env python3
"""Export node datasets of the study for another machine (MULTI_MACHINE_EXPERIMENTS.md at the repository root).

A node folder of data/study/<use case>/<node>/ is mostly symbolic links into the pooled images of the
use case; this copies it with real files, so that it can be moved to the machine that hosts the node.
The whole folder is copied: the training subset (a class-stratified fraction of the node's images, chosen
from the seed) is drawn on that machine at training time, from the full folder, exactly as in the study.
data.yaml is rewritten with `path` = --dest (where the folder will live on that machine); if the folder
ends up elsewhere, the client repairs `path` when it joins a room.

    python tools/export_node_data.py --use-case wbc --nodes node_B node_C --out ~/cofy_export/m2 \\
        --dest /home/<user>/cofy_data
    python tools/export_node_data.py --use-case wbc --eval --out ~/cofy_export/m1   # evaluation set (developer mode)
"""
import argparse, os, shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    files = size = 0
    for root, _, names in os.walk(src, followlinks=True):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for n in names:
            if n.endswith(".cache"):
                continue
            s, d = Path(root) / n, dst / rel / n
            shutil.copy2(s.resolve(), d)
            files += 1
            size += d.stat().st_size
    return files, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-case", required=True, choices=["kfm12", "wbc", "dentex"])
    ap.add_argument("--nodes", nargs="*", default=[])
    ap.add_argument("--eval", action="store_true", help="also export the evaluation set (_eval)")
    ap.add_argument("--out", required=True, help="export folder on this machine")
    ap.add_argument("--dest", help="folder that will hold the export on the target machine (for data.yaml)")
    a = ap.parse_args()
    out = Path(a.out).expanduser() / a.use_case
    for name in a.nodes + (["_eval"] if a.eval else []):
        src = ROOT / f"data/study/{a.use_case}/{name}"
        if not src.exists():
            raise SystemExit(f"{src} does not exist")
        dst = out / name
        if dst.exists():
            shutil.rmtree(dst)
        files, size = copy_tree(src, dst)
        meta = yaml.safe_load((src / "data.yaml").read_text())
        meta["path"] = str(Path(a.dest) / a.use_case / name) if a.dest else str(dst)
        (dst / "data.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        print(f"{a.use_case}/{name}: {files} files, {size / 1e9:.2f} GB -> {dst}")
    print(f"\ncopy to the target machine, e.g.:\n  rsync -a --progress {out}/ <user>@<machine>:{a.dest or '<dest>'}/{a.use_case}/")


if __name__ == "__main__":
    main()
