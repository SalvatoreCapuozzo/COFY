#!/usr/bin/env python3
"""Freeze record of the confirmatory study (analysis/CONFIRMATORY_PREREG.md): SHA-256 of every
file that decides its result -- the frozen configs, the training / calibration / evaluation
code, the data fingerprints and the analysis script.

    python tools/confirm_manifest.py           # write analysis/confirm_manifest.json (registration only)
    python tools/confirm_manifest.py --check   # list files changed since registration
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/confirm_manifest.json"
FILES = ["configs/variants/confirm_v2sync/*.yaml", "configs/variants/confirm_v2async/*.yaml",
         "configs/variants/v2_central/*_s0.yaml", "configs/arch/yolo26_classic_l.yaml",
         "fedyolo/*.py", "tools/run_variant.py", "tools/run_baseline.py", "tools/calibrate_checkpoints.py",
         "tools/confirm_analysis.py", "analysis/study_manifest.json", "analysis/CONFIRMATORY_PREREG.md"]


def digest():
    out = {}
    for pat in FILES:
        for p in sorted(ROOT.glob(pat)):
            out[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main():
    if "--check" in sys.argv:
        old = json.loads(OUT.read_text())["sha256"]
        new = digest()
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        print("\n".join(f"CHANGED {k}" for k in changed) or "all frozen files unchanged")
        sys.exit(1 if changed else 0)
    if OUT.exists():
        sys.exit("analysis/confirm_manifest.json already exists: the study is registered (use --check)")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rec = {"registered": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "git_head": git("rev-parse", "HEAD"),
           "uncommitted_changes": bool(git("status", "--porcelain")), "sha256": digest()}
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"registered {len(rec['sha256'])} files in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
