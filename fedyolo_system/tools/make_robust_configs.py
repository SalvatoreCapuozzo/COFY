#!/usr/bin/env python3
"""Configs and plan lines of the robustness study (analysis/ROBUSTNESS_PLAN.md).

Variant rb_<scheme>_<attack>[_ora][_noenf] = the scheme's protocol-v2 config (seq: v2_async,
conc: v2_vclock, par: v2_sync) with one attacking node (node seed mod 4, so it rotates over
seeds) and the defence on (_ora) or off; _noenf turns off ownership enforcement. The clean
undefended runs are the v2 runs themselves (linked, not retrained). Data attacks need
tools/make_attack_data.py first (done here for the nodes the seeds pick).

    python tools/make_robust_configs.py --dataset wbc --seeds 0 1 2 [--plan]   # --plan: print plan lines
"""
import argparse, os, subprocess, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = {"seq": ("v2_async", "async"), "conc": ("v2_vclock", "async"), "par": ("v2_sync", "sync")}
FACTOR = {"noise": 3.0, "signflip": 1.0, "scale": 10.0, "bnpoison": 3.0, "rowhijack": 5.0, "weightinflate": 100.0}
DATA_ATTACKS = ("labels", "backdoor")
MATRIX = {   # scheme: attacks (each with and without the defence, plus clean with the defence)
    "seq": ["noise", "signflip", "scale", "bnpoison", "labels", "backdoor"],
    "conc": ["noise", "signflip", "scale", "bnpoison", "labels", "backdoor"],
    "par": ["signflip", "scale", "weightinflate", "backdoor"],
}


def variants():
    out = []
    for sch, attacks in MATRIX.items():
        out.append((sch, "clean", True, True))
        for att in attacks:
            out += [(sch, att, False, True), (sch, att, True, True)]
        if sch != "par":
            out += [(sch, "rowhijack", False, True), (sch, "rowhijack", False, False)]
    return out


def name(sch, att, ora, enf):
    return f"rb_{sch}_{att}" + ("_ora" if ora else "") + ("" if enf else "_noenf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wbc")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    for s in a.seeds:
        nodes = yaml.safe_load((ROOT / f"configs/variants/v2_async/{ds}_s{s}.yaml").read_text())["nodes"]
        node = nodes[s % len(nodes)]["name"]
        for att in DATA_ATTACKS:
            if not (ROOT / f"data/attack/{ds}/{node}_{att}/data.yaml").exists():
                subprocess.run([sys.executable, str(ROOT / "tools/make_attack_data.py"), "--dataset", ds,
                                "--node", node, "--attack", att], check=True, cwd=ROOT)
    for sch in SRC:   # clean, undefended: the v2 runs themselves
        src, _ = SRC[sch]
        for s in a.seeds:
            link = ROOT / f"runs/variants/rb_{sch}_clean/{ds}/s{s}"
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                os.symlink(ROOT / f"runs/variants/{src}/{ds}/s{s}", link)
            cfg = ROOT / f"configs/variants/rb_{sch}_clean/{ds}_s{s}.yaml"
            cfg.parent.mkdir(parents=True, exist_ok=True)
            if not cfg.exists():
                os.symlink(ROOT / f"configs/variants/{src}/{ds}_s{s}.yaml", cfg)
    lines = []
    for sch, att, ora, enf in variants():
        src, mode = SRC[sch]
        nm = name(sch, att, ora, enf)
        for s in a.seeds:
            c = yaml.safe_load((ROOT / f"configs/variants/{src}/{ds}_s{s}.yaml").read_text())
            node = c["nodes"][s % len(c["nodes"])]["name"]
            rb = {"attacker": node, "attack": "none" if att in DATA_ATTACKS + ("clean",) else att,
                  "attack_factor": FACTOR.get(att, 1.0), "defense": "ora" if ora else "none",
                  "enforce_ownership": enf}
            if att in DATA_ATTACKS:
                for n in c["nodes"]:
                    if n["name"] == node:
                        n["data_yaml"] = f"data/attack/{ds}/{node}_{att}/data.yaml"
            c["federation"]["robust"] = rb
            c["output_dir"] = f"runs/variants/{nm}/{ds}"
            out = ROOT / f"configs/variants/{nm}/{ds}_s{s}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(c, sort_keys=False))
        lines.append(f"{nm:26s} {mode}@{ds:24s} {' '.join(map(str, a.seeds))}")
    if a.plan:
        print("\n".join(lines))
    else:
        print(f"{len(lines)} variants x {len(a.seeds)} seeds written")


if __name__ == "__main__":
    main()
