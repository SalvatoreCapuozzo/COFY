#!/usr/bin/env python3
"""Fingerprint the study's data so a second machine can prove it rebuilt the
exact same splits and training subsets.

    write:   python3 tools/study_manifest.py write  <manifest.json>
    verify:  python3 tools/study_manifest.py verify <manifest.json>

For each dataset it hashes: every pool image name+size, every label file, each
node's train/val file list, the eval set's file list and label contents, the
oracle's file lists, and -- for all five training seeds -- the balanced-fraction
subset each node and the oracle actually train on (cofy.sampling, exactly as
cofy.data calls it). A single mismatch fails verification.
"""
import hashlib, json, sys, tempfile, os
from pathlib import Path
from cofy.sampling import cached_balanced_image_subset

DATASETS = {"kfm12": 0.1, "wbc": 0.1, "chestx": 0.5}    # pre-registered fractions
SEEDS = range(5)
IMG = {".jpg", ".jpeg", ".png", ".bmp"}

def h(lines): return hashlib.sha256("\n".join(lines).encode()).hexdigest()
def names(d): return sorted(p.name for p in Path(d).iterdir() if p.suffix.lower() in IMG) if Path(d).exists() else []

def fingerprint(ds, frac):
    pool, st, orc = Path(f"data/_pool/{ds}"), Path(f"data/study/{ds}"), Path(f"data/study/{ds}_oracle/node_A")
    fp = {"pool_images": h(f"{n}|{os.path.getsize(os.path.realpath(pool/'images'/n))}" for n in names(pool/"images")),
          "pool_image_count": len(names(pool/"images")),
          "pool_labels": h(f"{p.name}|{p.read_text()}" for p in sorted((pool/"labels").iterdir())),
          "classes_list": h([(pool/"classes_list.txt").read_text()])}
    for n in "ABCD":
        for sp in ("train", "val"):
            fp[f"node_{n}_{sp}"] = h(names(st/f"node_{n}"/"images"/sp))
        lbl = st/f"node_{n}"/"labels"/"train"
        fp[f"node_{n}_train_labels"] = h(f"{p.name}|{p.read_text()}" for p in sorted(lbl.iterdir()))
    fp["eval_images"] = h(names(st/"_eval"/"images"/"val"))
    fp["eval_labels"] = h(f"{p.name}|{p.read_text()}" for p in sorted((st/"_eval"/"labels"/"val").iterdir()))
    fp["eval_count"] = len(names(st/"_eval"/"images"/"val"))
    for sp in ("train", "val"):
        fp[f"oracle_{sp}"] = h(names(orc/"images"/sp))
    for s in SEEDS:
        for n, d in [(f"node_{x}", st/f"node_{x}"/"images"/"train") for x in "ABCD"] + [("oracle", orc/"images"/"train")]:
            with tempfile.TemporaryDirectory() as t:
                sub = cached_balanced_image_subset(d, frac, s, cache_file=Path(t)/"c")
            fp[f"subset_seed{s}_{n}"] = h(sorted(p.name for p in sub))
    return fp

def main():
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "write":
        out = {ds: fingerprint(ds, f) for ds, f in DATASETS.items()}
        path.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(f"manifest written: {path} ({sum(len(v) for v in out.values())} fingerprints)")
        return
    ref, bad = json.loads(path.read_text()), 0
    for ds, f in DATASETS.items():
        got = fingerprint(ds, f)
        diff = [k for k in ref[ds] if ref[ds][k] != got.get(k)]
        bad += len(diff)
        print(f"{ds:7s} {'PASS' if not diff else 'FAIL'}  {len(ref[ds]) - len(diff)}/{len(ref[ds])} fingerprints match"
              + (f"  mismatched: {diff[:6]}" if diff else ""))
    print("ALL DATA VERIFIED -- identical to the Mac" if not bad else "VERIFICATION FAILED -- do not start the study")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
