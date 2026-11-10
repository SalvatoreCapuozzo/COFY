#!/usr/bin/env python3
"""Gate of the ChestX fix screens (analysis/CHESTX_CONCURRENT.md §6), fixed before any result.

Each candidate first runs ChestX seed 3, the development seed where the concurrent scheme shows
the problem (v2_vclock: Fed:Cent 83.6 %, worst node 70 %). Once its calibration is done:
  pass  Fed:Cent >= 90 % (stage acal_refresh) and every node's classes >= 85 % on average
        -> its plan lines get seeds 1 and 2 (they run next, before the rest of the queue);
  fail  -> nothing more is run for it.
Each decision is taken once and kept in runs/variants/_cx_gate.json. Called by
tools/search_rules.py after every queue entry; safe to run by hand.

    python tools/cx_gate.py
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cx_diagnosis as cx

ROOT = Path(__file__).resolve().parent.parent
PLAN, LOG = ROOT / "runs/variants/_plan.txt", ROOT / "runs/variants/_cx_gate.json"
CANDIDATES = ["cx_chain4", "cx_towers", "cx_proj"]
SCREEN_SEED, MORE_SEEDS = 3, [1, 2]
MIN_RATIO, MIN_NODE = 90.0, 85.0


def say(msg):
    print(f"{time.strftime('%m-%d %H:%M')} [cx_gate] {msg}", flush=True)


def main():
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    lines = PLAN.read_text().splitlines()
    changed = False
    for name in CANDIDATES:
        if name in log:
            continue
        r = cx.row(name, [SCREEN_SEED], "acal_refresh")
        if SCREEN_SEED not in r:
            continue
        ratio, nodes = r[SCREEN_SEED]
        ok = ratio >= MIN_RATIO and min(nodes.values()) >= MIN_NODE
        log[name] = {"decided": time.strftime("%Y-%m-%d %H:%M"), "seed": SCREEN_SEED, "fed_cent": round(ratio, 1),
                     "nodes": {k: round(v, 1) for k, v in sorted(nodes.items())}, "pass": ok}
        say(f"{name} seed {SCREEN_SEED}: Fed:Cent {ratio:.1f} %, nodes "
            + " ".join(f"{k} {v:.0f}" for k, v in sorted(nodes.items()))
            + (" -> PASS, seeds 1-2 queued" if ok else " -> fail, stopped"))
        if ok:
            for i, l in enumerate(lines):
                p = l.split("#", 1)[0].split()
                if p and p[0] in (name, f"calib_{name}") and p[2:] == [str(SCREEN_SEED)]:
                    lines[i] = l.rstrip() + " " + " ".join(map(str, MORE_SEEDS))
            changed = True
    if changed:
        PLAN.write_text("\n".join(lines) + "\n")
    LOG.write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
