#!/usr/bin/env python3
"""Held-out dataset of the confirmatory study (analysis/CONFIRMATORY_PREREG.md, amendment 1):
once the budget pilot has finished, write the frozen configs, hash them, and queue the runs.

- budget = the pilot's budget_epochs (runs/variants/_budget/<ds>/s0/curve.json), or --budget when a
  dated amendment fixes it otherwise (amendment 3: DENTEX 20), rounded up to whole rounds of
  local_epochs; rounds = async_node_cycles = budget / local_epochs;
- configs/variants/v2_central/<ds>_s0.yaml gets those rounds (the centralized reference);
- confirm_v2sync / confirm_v2async/<ds>_s<seed>.yaml: the frozen schemes' settings (taken from
  their chestx configs, so nothing but the dataset, the training fraction and the budget changes);
- analysis/confirm_manifest_amend1.json: SHA-256 of the new configs and of the dataset's labels;
- plan lines (seed by seed, same order as the confirmation) after the '# ---- HELD-OUT' marker.

    python tools/make_heldout_configs.py --dataset dentex --budget 20 --amendment 3
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SEEDS = range(100, 110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--budget", type=int, help="epochs, overriding the pilot's (needs --amendment)")
    ap.add_argument("--amendment", type=int, help="the amendment of CONFIRMATORY_PREREG.md that fixes --budget")
    a = ap.parse_args()
    ds = a.dataset
    pilot = json.loads((ROOT / f"runs/variants/_budget/{ds}/s0/curve.json").read_text())
    if "budget_epochs" not in pilot:
        sys.exit("the pilot has not finished")
    if a.budget and not a.amendment:
        sys.exit("--budget needs --amendment")
    budget = a.budget or pilot["budget_epochs"]
    tpl_path = ROOT / f"configs/variants/v2_central/{ds}_s0.yaml"
    tpl = yaml.safe_load(tpl_path.read_text())
    le = tpl["federation"]["local_epochs"]
    rounds = math.ceil(budget / le)
    tpl["federation"]["rounds"] = tpl["federation"]["async_node_cycles"] = rounds
    tpl_path.write_text(yaml.safe_dump(tpl, sort_keys=False))
    written = [tpl_path]
    for scheme in ("confirm_v2sync", "confirm_v2async"):
        fed = yaml.safe_load((ROOT / f"configs/variants/{scheme}/chestx_s100.yaml").read_text())["federation"]
        fed.update(balanced_fraction=tpl["federation"]["balanced_fraction"], rounds=rounds, async_node_cycles=rounds)
        for s in SEEDS:
            c = {"global_classes": tpl["global_classes"], "model": tpl["model"], "federation": fed,
                 "nodes": tpl["nodes"], "eval_data_yaml": tpl["eval_data_yaml"],
                 "output_dir": f"runs/variants/{scheme}/{ds}", "seed": s}
            out = ROOT / f"configs/variants/{scheme}/{ds}_s{s}.yaml"
            out.write_text(yaml.safe_dump(c, sort_keys=False))
            written.append(out)
    labels = sorted((ROOT / f"data/_pool/{ds}/labels").glob("*.txt"))
    h = hashlib.sha256()
    for p in labels:
        h.update(p.name.encode() + p.read_bytes())
    rec = {"amendment": 1, "dataset": ds, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "pilot": {"budget_epochs": pilot["budget_epochs"], "best_map50": pilot.get("best_map50")},
           "budget": {"epochs": budget, "set_by": f"amendment {a.amendment}" if a.budget else "pilot rule",
                      "rounds": rounds, "epochs_trained": rounds * le},
           "labels_sha256": h.hexdigest(), "images_list_sha256":
               hashlib.sha256((ROOT / f"data/_pool/{ds}/images.txt").read_bytes()).hexdigest(),
           "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in written}}
    out = ROOT / "analysis/confirm_manifest_amend1.json"
    if out.exists():
        sys.exit(f"{out} exists: configs already frozen")
    out.write_text(json.dumps(rec, indent=1) + "\n")
    plan = ROOT / "runs/variants/_plan.txt"
    lines = plan.read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("# ---- HELD-OUT"))
    new = []
    for s in SEEDS:
        new += [f"{'confirm_central':22s} {'baseline:v2@' + ds:32s} {s}",
                f"{'calib_confirm_central':22s} {'posthoc:v2_centralized:-@' + ds:32s} {s}",
                f"{'confirm_v2sync':22s} {'async@' + ds:32s} {s}",
                f"{'calib_confirm_v2sync':22s} {'posthoc:confirm_v2sync:async@' + ds:32s} {s}",
                f"{'confirm_v2async':22s} {'async@' + ds:32s} {s}",
                f"{'calib_confirm_v2async':22s} {'posthoc:confirm_v2async:async@' + ds:32s} {s}"]
    lines[i + 1:i + 1] = new
    plan.write_text("\n".join(lines) + "\n")
    print(f"{ds}: budget {budget} epochs (pilot {pilot['budget_epochs']}) -> {rounds} rounds; {len(written)} configs; "
          f"{len(new)} plan lines; manifest {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
