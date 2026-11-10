#!/usr/bin/env python3
"""Score a finished federated run the low-noise way, for comparing merge strategies.

Why not just read summary.json
------------------------------
The in-training evals use `federation.balanced_fraction` (0.1 here), i.e. a
267-image class-balanced subsample, and summary.json reports whichever round
happened to be last. On this config that combination is far too noisy to
compare two merge rules against each other: the baseline WBC run scores
0.6527 / 0.6765 / 0.5723 at rounds 6 / 8 / 10 -- a 0.104 swing in the last two
rounds, larger than any merge-rule effect worth chasing.

So this scores on the FULL pooled validation set (2,667 images for WBC, no
subsample) and averages over whichever late-round checkpoints the run retained
(`federation.keep_checkpoints`, now 3). Runs made before that setting existed
kept only their final checkpoint; those are scored on the full set too, and the
last-3 average is reported from map_history.json / the run log instead, on the
small eval set, clearly labeled as such.

    python3 tools/score_run.py --run runs/.../wbc_m1ties_sync_l640 \
                               --config configs/..._m1ties.yaml --mode sync
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import statistics
from pathlib import Path

import torch

from cofy.config import load_config
from cofy.evaluate import evaluate_state_dict
from cofy.pooling import materialize_pooled_dataset


def full_pool(cfg, out_dir: Path) -> Path:
    if (out_dir / "pooled.yaml").exists():
        return out_dir / "pooled.yaml"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # fraction=None -> every val image, not the balanced_fraction subsample
    return materialize_pooled_dataset(cfg, split="val", out_dir=out_dir,
                                      fraction=None, seed=cfg.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="sync")
    ap.add_argument("--label", default=None, help="name for this variant in the output")
    args = ap.parse_args()

    run, cfg = Path(args.run), load_config(args.config)
    work = run / "score_full"
    cfg = dataclasses.replace(cfg, output_dir=str(work))
    work.mkdir(parents=True, exist_ok=True)
    full_pool(cfg, work / "pooled_val")

    ckpts = sorted((run / args.mode).glob("global_round*.pt"))
    final = run / args.mode / "global_final.pt"
    if final.exists() and final not in ckpts:
        ckpts.append(final)
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run/args.mode}")

    scores = {}
    for c in ckpts:
        blob = torch.load(c, map_location="cpu", weights_only=False)
        m = evaluate_state_dict(cfg, blob.get("state_dict", blob), name=f"full_{c.stem}")
        scores[c.stem] = float(m.box.map50)
        print(f"  {c.stem:20s} mAP50(full pool) = {scores[c.stem]:.4f}")

    # last-3 on the small in-training eval set, from whatever the run recorded
    hist = []
    hp = run / args.mode / "map_history.json"
    if hp.exists():
        hist = [h["map50"] for h in json.loads(hp.read_text())]
    else:
        log = run.parent.parent / f"_matrix_log_{run.name}.log"
        alt = list(Path("runs").glob(f"_*{run.name}*.log"))
        for cand in ([log] if log.exists() else []) + alt:
            txt = cand.read_text(errors="ignore").replace("\r", "\n")
            seen, vals = set(), []
            for lbl, v in re.findall(r"(round \d{3})[^0-9]+(0\.\d{4})", txt):
                if lbl not in seen:
                    seen.add(lbl); vals.append(float(v))
            if vals:
                hist = vals
                break

    out = {
        "variant": args.label or run.name,
        "full_pool_scores": scores,
        "full_pool_mean_of_retained": statistics.mean(scores.values()),
        "small_eval_history": hist,
        "small_eval_last3_mean": statistics.mean(hist[-3:]) if len(hist) >= 3 else None,
    }
    (run / "score_full.json").write_text(json.dumps(out, indent=2))
    print(f"\n  full-pool mean over {len(scores)} retained ckpt(s): "
          f"{out['full_pool_mean_of_retained']:.4f}")
    if out["small_eval_last3_mean"] is not None:
        print(f"  small-eval last-3 mean ({hist[-3:]}): {out['small_eval_last3_mean']:.4f}")
    print(f"written: {run/'score_full.json'}")


if __name__ == "__main__":
    main()
