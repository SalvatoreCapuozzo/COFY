#!/usr/bin/env python3
"""Freeze record of the confirmatory study (analysis/CONFIRMATORY_PREREG.md): SHA-256 of every
file that decides its result -- the frozen configs, the training / calibration / evaluation
code, the data fingerprints and the analysis script.

    python tools/confirm_manifest.py           # write analysis/confirm_manifest.json (registration only)
    python tools/confirm_manifest.py --check   # list files changed since registration

The check reads code from the frozen copy the queue runs (_frozen/confirm/), configs and analysis
files from the working tree, and lists files recorded by a later amendment manifest
(analysis/confirm_manifest_amend*.json) or changed by a dated amendment separately.
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis/confirm_manifest.json"
FROZEN = ROOT / "_frozen/confirm"
AMENDED = {"analysis/CONFIRMATORY_PREREG.md": "amendments in its section 8",
           "tools/confirm_analysis.py": "amendment 2 (new SHA-256 in CONFIRMATORY_PREREG.md)"}
FILES = ["configs/variants/confirm_v2sync/*.yaml", "configs/variants/confirm_v2async/*.yaml",
         "configs/variants/v2_central/*_s0.yaml", "configs/arch/yolo26_classic_l.yaml",
         "fedyolo/*.py", "tools/run_variant.py", "tools/run_baseline.py", "tools/calibrate_checkpoints.py",
         "tools/confirm_analysis.py", "analysis/study_manifest.json", "analysis/CONFIRMATORY_PREREG.md"]


def digest(code_root=ROOT):
    out = {}
    for pat in FILES:
        base = code_root if pat.startswith(("fedyolo/", "tools/")) else ROOT
        for p in sorted(base.glob(pat)):
            out[str(p.relative_to(base))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def amendment_records():
    rec = {}
    for f in sorted(ROOT.glob("analysis/confirm_manifest_amend*.json")):
        for k, h in json.loads(f.read_text())["sha256"].items():
            rec[k] = (h, f.name)
    return rec


def main():
    if "--check" in sys.argv:
        old = json.loads(OUT.read_text())["sha256"]
        new = digest(FROZEN)
        amend = amendment_records()
        changed, lines = [], []
        for k in sorted(old.keys() | new.keys()):
            if old.get(k) == new.get(k):
                continue
            if k in amend and amend[k][0] == new.get(k):
                lines.append(f"amended  {k}  (recorded in {amend[k][1]})")
            elif k in AMENDED:
                lines.append(f"amended  {k}  ({AMENDED[k]})")
            else:
                changed.append(k)
                lines.append(f"CHANGED  {k}")
        print("\n".join(lines))
        print(f"code read from {FROZEN.relative_to(ROOT)}/: " +
              ("no unrecorded change" if not changed else f"{len(changed)} unrecorded change(s)"))
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
