#!/usr/bin/env python3
"""Score a multi-machine run and compare it with its simulated counterpart (MULTI_MACHINE_EXPERIMENTS.md).

Run on the server machine, which holds the evaluation set (developer mode): it evaluates the room's
latest 'final' and 'calibrated' checkpoints (<output_dir>/net_room_<room>/) on the use case's evaluation
set, and prints them next to the simulated run of the same seed (at the calibration stage that run was
registered with) and the calibrated centralized reference of that seed (Fed:Cent = federated /
centralized mAP50 x 100). Results go to runs/net/<experiment>/comparison.json.

    python tools/compare_net_run.py --experiment wbc_cwt_s100 --room wbc-cwt-s100
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def stored(path, stage):
    try:
        return json.loads((ROOT / path).read_text())[stage]["map50"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="a key of configs/net/index.json")
    ap.add_argument("--room", required=True, help="room id used on the server")
    a = ap.parse_args()
    from cofy.config import load_config
    from cofy.evaluate import evaluate_state_dict
    import torch
    exp = json.loads((ROOT / "configs/net/index.json").read_text())[a.experiment]
    cfg = load_config(ROOT / exp["config"])
    room_dir = ROOT / cfg.output_dir / f"net_room_{a.room}"
    res = {"experiment": a.experiment, "room": a.room}
    cent = stored(exp["centralized"]["result"], exp["centralized"]["stage"])
    for label in ("final", "calibrated"):
        ck = sorted(room_dir.glob(f"*_{a.room}_{label}.pt"))
        if not ck:
            print(f"{label}: no checkpoint in {room_dir}")
            continue
        sd = torch.load(ck[-1], map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        m = float(evaluate_state_dict(cfg, sd, name=f"net_{label}").box.map50)
        res[label] = {"checkpoint": str(ck[-1].relative_to(ROOT)), "map50": m,
                      "fed_cent": 100 * m / cent if cent else None}
    sim = exp.get("simulated")
    if sim:
        s = stored(sim["result"], sim["stage"])
        res["simulated"] = {"variant": sim["variant"], "stage": sim["stage"], "map50": s,
                            "fed_cent": 100 * s / cent if s and cent else None}
    res["centralized_map50"] = cent
    out = ROOT / cfg.output_dir / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    f = lambda d: "-" if not d or d.get("map50") is None else f"mAP50 {d['map50']:.3f}" + (
        f" (Fed:Cent {d['fed_cent']:.1f})" if d.get("fed_cent") else "")
    print(f"{a.experiment}: network final {f(res.get('final'))} | network calibrated {f(res.get('calibrated'))} | "
          f"simulated {f(res.get('simulated'))} | centralized {cent if cent is None else round(cent, 3)}\n-> {out}")


if __name__ == "__main__":
    main()
