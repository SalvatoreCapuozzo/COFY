#!/usr/bin/env python3
"""Why the concurrent scheme loses on ChestX (analysis/CHESTX_CONCURRENT.md). Reads finished
runs only (no GPU).

1. Confirmation runs (seeds 100-109): Fed:Cent per calibration stage for the per-node final
   average (confirm_v2async, as registered) and for the last global version (diag_cx_last).
2. Fix screens on the development seeds (gate: tools/cx_gate.py): cx_chain4 (last 4 cycles as a
   server-ordered chain), cx_towers (owned class towers), cx_proj (conflict projection) against the concurrent (v2_vclock) and sequential (v2_async) schemes of
   the same seeds, overall and per node (mean retention of the node's classes).

    python tools/cx_diagnosis.py
"""
import json, statistics as st
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "runs/variants"
DS = "chestx"
STAGES = ["before", "bn+head", "acal_refresh"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def cent(s):
    return (load(VAR / f"calib/v2_centralized/{DS}/s{s}/centralized.json") or {}).get("bn+head")


def owners(src, s):
    cfg = yaml.safe_load((ROOT / f"configs/variants/{src}/{DS}_s{s}.yaml").read_text())
    return {n["data_yaml"].split("/")[-2][-1]: n["owned_classes"] for n in cfg["nodes"]}


def row(src, seeds, stage):
    """{seed: (Fed:Cent, {node: retention})} for the finished seeds."""
    out = {}
    for s in seeds:
        f = (load(VAR / f"calib/{src}/{DS}/s{s}/async.json") or {}).get(stage)
        c = cent(s)
        if not f or not c:
            continue
        nodes = {nm: st.mean(f["per_class_map50"][k] / c["per_class_map50"][k] * 100
                             for k in ks if c["per_class_map50"][k] > 0) for nm, ks in owners(src, s).items()}
        out[s] = (f["map50"] / c["map50"] * 100, nodes)
    return out


def fmt(r):
    if not r:
        return "not run yet"
    v = [x for x, _ in r.values()]
    nodes = {nm: st.mean(n[nm] for _, n in r.values()) for nm in next(iter(r.values()))[1]}
    return (f"mean {st.mean(v):5.1f} (n={len(v)}: " + " ".join(f"{x:.0f}" for x in v) + ") | nodes "
            + " ".join(f"{nm} {x:.0f}" for nm, x in sorted(nodes.items())))


def main():
    print("1. Confirmation runs (seeds 100-109), Fed:Cent % -- per-node final average vs last global version")
    for stage in STAGES:
        a, b = row("confirm_v2async", range(100, 110), stage), row("diag_cx_last", range(100, 110), stage)
        print(f"  {stage:13s} average: {fmt(a)}")
        print(f"  {'':13s} last:    {fmt(b)}")
        both = [s for s in a if s in b]
        if len(both) >= 2:
            d = [b[s][0] - a[s][0] for s in both]
            print(f"  {'':13s} last - average: {st.mean(d):+.1f} points (sd {st.stdev(d):.1f}, n={len(d)})")
    print("\n2. Fix screens, development seeds 3 (gate), 1, 2; Fed:Cent % (stage acal_refresh; sequential: bn+head)")
    for src, stage in [("v2_async", "bn+head"), ("v2_vclock", "acal_refresh"), ("cx_chain4", "acal_refresh"),
                       ("cx_towers", "acal_refresh"), ("cx_proj", "acal_refresh")]:
        print(f"  {src:10s} {stage:13s} {fmt(row(src, (3, 1, 2), stage))}")


if __name__ == "__main__":
    main()
