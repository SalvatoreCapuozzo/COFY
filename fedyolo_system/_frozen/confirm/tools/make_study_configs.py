#!/usr/bin/env python3
"""Write the per-seed configs for the pre-registered study
(analysis/PREREGISTRATION.md). One config per (dataset, seed); each drives a
`--mode sync-async` run, so sync and async share one centralized and one
oracle baseline per seed.

    python3 tools/make_study_configs.py <dataset> <splitter_config.yaml> <balanced_fraction>
"""
import sys, yaml
from pathlib import Path

ds, src, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
base = yaml.safe_load(Path(src).read_text())
assert base.get("eval_data_yaml"), "splitter config lacks eval_data_yaml -- was --split-mode global used?"
out = Path("configs/study"); out.mkdir(parents=True, exist_ok=True)
for seed in range(5):
    cfg = {
        "global_classes": base["global_classes"],
        "model": {"arch": "configs/arch/yolo26_classic_l.yaml", "imgsz": 640, "pretrained": True},
        "federation": {
            # identical to the existing yolo26-classic@l640 runs
            "local_epochs": 2, "batch_size": 4, "lr0": 0.0008, "warmup_steps": 20,
            "grad_clip": 10.0, "workers": 0, "device": "auto",
            "balanced_fraction": frac,
            "rounds": 10,                      # sync
            "async_node_cycles": 10,           # async
            "staleness_alpha": 0.5, "max_concurrent_nodes": 1,
            "pseudo_label": {"enabled": False, "start_round": 5, "conf_thresh": 0.6, "weight": 0.5},
            # primary test is plain FedAvg -- improvements are evaluated separately
            "merge_rule": "mean", "normalize_steps": False,
            # in-training evals are for monitoring only (the endpoint is the
            # final checkpoint), so keep them sparse on the full eval set
            "eval_interval": 20,
            "keep_checkpoints": 1,
        },
        "nodes": base["nodes"],
        "eval_data_yaml": base["eval_data_yaml"],
        "output_dir": f"runs/study/{ds}",
        "seed": seed,
    }
    p = out / f"{ds}_s{seed}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("wrote", p)
