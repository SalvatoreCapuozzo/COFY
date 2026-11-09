"""Find the maximum batch_size that fits in GPU memory, for each
(model arch, input size) combination, using REAL training steps (forward +
backward + optimizer.step(), same as an actual run) on real data from one
of your configured nodes.

Each (arch, imgsz, batch) trial runs in its own subprocess -- an OOM in one
trial can leave the CUDA allocator fragmented, which would silently bias
the NEXT trial's result if run in the same process. Isolating trials keeps
every measurement clean and reproducible, at the cost of a few seconds of
subprocess/model-build overhead per trial.

    python tools/calibrate_gpu_limits.py --config configs/kfm_250423_filtered.yaml

    # narrower/faster sweep, and include pseudo_label's extra teacher-model cost
    python tools/calibrate_gpu_limits.py --config configs/kfm_250423_filtered.yaml \\
        --archs yolov8m.yaml,yolov8l.yaml --imgsz-list 640,960 --pseudo-label
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_trial(config: str, arch: str, imgsz: int, batch: int, node: str,
                pseudo_label: bool, steps: int, timeout: int) -> dict:
    cmd = [
        sys.executable, __file__, "--worker",
        "--config", config, "--arch", arch, "--imgsz", str(imgsz),
        "--batch", str(batch), "--node", node, "--steps", str(steps),
    ]
    if pseudo_label:
        cmd.append("--pseudo-label")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "peak_gb": None, "error": f"trial exceeded {timeout}s timeout"}
    for line in reversed(proc.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"ok": False, "peak_gb": None, "error": f"worker produced no result; stderr: {proc.stderr[-500:]}"}


def _worker(args: argparse.Namespace) -> None:
    """Runs INSIDE the isolated subprocess. Prints exactly one JSON line at
    the end (parent parses the LAST line, so noisy logging above is fine)."""
    import torch
    from fedyolo.config import load_config
    from fedyolo.model import build_model
    from fedyolo.data import build_node_dataloader
    from fedyolo.client import _class_mask
    from ultralytics.utils.loss import v8DetectionLoss

    try:
        cfg = load_config(args.config)
        cfg.model.arch = args.arch
        cfg.model.imgsz = args.imgsz
        cfg.federation.batch_size = args.batch
        device = torch.device("cuda")
        node = next(n for n in cfg.nodes if n.name == args.node)

        model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained).to(device)
        model.class_weights = _class_mask(cfg.nc, node.owned_global_ids(cfg.global_classes), device)
        model.train()
        criterion = v8DetectionLoss(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0008, momentum=0.9, weight_decay=5e-4)

        teacher = None
        if args.pseudo_label:
            teacher = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=cfg.model.pretrained).to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        loader = build_node_dataloader(node, cfg, split="train")
        it = iter(loader)

        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.steps):
            batch = next(it)
            batch["img"] = batch["img"].to(device).float() / 255.0
            optimizer.zero_grad()
            preds = model.forward(batch["img"])
            parsed = criterion.parse_output(preds)
            loss, _ = criterion.loss(parsed, batch)
            loss = loss.sum()
            if teacher is not None:
                with torch.no_grad():
                    t_preds = teacher.forward(batch["img"])
                    criterion.parse_output(t_preds)  # touch it to allocate its activations
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(json.dumps({"ok": True, "peak_gb": round(peak_gb, 2), "error": None}))
    except torch.OutOfMemoryError:
        print(json.dumps({"ok": False, "peak_gb": None, "error": "OOM"}))
    except Exception as exc:
        print(json.dumps({"ok": False, "peak_gb": None, "error": str(exc)[:300]}))


def _pick_biggest_node(cfg) -> str:
    import yaml
    counts = {}
    for n in cfg.nodes:
        dy = yaml.safe_load(Path(n.data_yaml).read_text())
        base = Path(dy.get("path", Path(n.data_yaml).parent))
        counts[n.name] = sum(1 for _ in (base / dy["train"]).iterdir())
    return max(counts, key=counts.get)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--archs", default="yolov8n.yaml,yolov8s.yaml,yolov8m.yaml,yolov8l.yaml,yolov8x.yaml")
    p.add_argument("--imgsz-list", default="640,960,1280")
    p.add_argument("--batch-candidates", default="1,2,4,6,8,12,16,24,32")
    p.add_argument("--node", default=None, help="node to source real batches from (default: node with the most train images)")
    p.add_argument("--pseudo-label", action="store_true", help="also build a teacher model, matching pseudo_label.enabled=true's extra memory cost")
    p.add_argument("--steps", type=int, default=3, help="training steps per trial (memory usually peaks within the first 1-2)")
    p.add_argument("--timeout", type=int, default=300, help="per-trial subprocess timeout in seconds")
    # internal worker-mode flags (not for direct use)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arch")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)
    args = p.parse_args()

    if args.worker:
        _worker(args)
        return

    from fedyolo.config import load_config
    cfg = load_config(args.config)
    node_name = args.node or _pick_biggest_node(cfg)
    print(f"[calibrate] using node '{node_name}' for realistic batches")
    print(f"[calibrate] pseudo_label memory cost {'INCLUDED' if args.pseudo_label else 'NOT included'} in this sweep\n")

    archs = args.archs.split(",")
    imgsz_list = [int(x) for x in args.imgsz_list.split(",")]
    batch_candidates = sorted(int(x) for x in args.batch_candidates.split(","))

    results = {}
    for arch in archs:
        for imgsz in imgsz_list:
            print(f"=== {arch} @ imgsz={imgsz} ===")
            best_batch, best_peak = None, None
            for batch in batch_candidates:
                r = _run_trial(args.config, arch, imgsz, batch, node_name, args.pseudo_label, args.steps, args.timeout)
                if r["ok"]:
                    print(f"  batch={batch:3d}  OK    peak={r['peak_gb']:.2f} GB")
                    best_batch, best_peak = batch, r["peak_gb"]
                else:
                    print(f"  batch={batch:3d}  FAIL  ({r['error']})")
                    break  # larger batches only fail harder -- stop here, move to next combo
            results[f"{arch}@{imgsz}"] = {"max_batch": best_batch, "peak_gb": best_peak}
            print()

    print("=== summary ===")
    print(f"{'arch':<16}{'imgsz':<8}{'max_batch':<24}{'peak_gb':<10}")
    for key, r in results.items():
        arch, imgsz = key.split("@")
        mb = str(r["max_batch"]) if r["max_batch"] is not None else "0 (OOM even at batch=1)"
        pg = f"{r['peak_gb']:.2f}" if r["peak_gb"] is not None else "-"
        print(f"{arch:<16}{imgsz:<8}{mb:<24}{pg:<10}")

    out_path = Path("gpu_calibration_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    print("Note: pick a batch size a step BELOW the reported max for headroom -- "
          "this measures peak allocation over a few steps, not worst-case over a full run "
          "(augmentation, distinct batches, and memory fragmentation over many steps can push higher).")


if __name__ == "__main__":
    main()
