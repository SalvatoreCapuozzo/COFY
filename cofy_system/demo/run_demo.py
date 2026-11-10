#!/usr/bin/env python3
"""One-command demonstration of a COFY federation over the network, on one machine.

It starts the real server (cofy.net.server_app) and one real client process per node
(cofy.net.client_app), creates a room for the chosen use case and scheme, joins every node with
its own small dataset from demo/data/, runs the federation and its calibration, and prints what
happens: turns, uploads, the calibration contributions, and the accuracy of the released model.
The server and client dashboards can be watched in a browser while it runs.

Every node only sees its own images, labelled only for the classes it owns; it learns about the
other classes only through the model it downloads, and only the owner of a class changes that
class's output rows. The server holds no data: here it runs in developer mode, so that it can
score the model on demo/data/<use case>/_eval for the demonstration (a real deployment does not).

    cd cofy_system
    python demo/run_demo.py --use-case wbc --scheme sequential
    python demo/run_demo.py --use-case kfm12 --scheme concurrent --keep-open
    python demo/run_demo.py --use-case dentex --scheme sequential --scale l --imgsz 640   # the study's model

Options: --use-case kfm12|wbc|dentex, --scheme sequential|concurrent, --cycles N (default 3),
--local-epochs N (default 10), --lr0 (default 0.01, the study used 0.0008 over full runs),
--scale n|l (model size; n is fast on a CPU, l is the study's), --imgsz, --cpu, --port,
--keep-open (leave the dashboards running at the end; Ctrl-C to stop).
Output: runs/demo/<use case>_<scheme>/ (config, logs, server state, models).
"""
import argparse
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "demo/data"
PROCS: list[subprocess.Popen] = []


def stop_all():
    for p in PROCS:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in PROCS:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def wait_up(url: str, what: str, timeout: float = 120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    sys.exit(f"{what} did not start within {timeout:.0f} s -- see its log in the run folder")


def write_config(args, out: Path) -> Path:
    """A room config: the study's protocol for the chosen scheme, made small for a demonstration."""
    meta = yaml.safe_load((DATA / args.use_case / "classes.yaml").read_text())
    nodes = []
    for n in meta["nodes"]:
        folder = (DATA / args.use_case / n["name"]).resolve()
        y = out / f"data_{n['name']}.yaml"
        y.write_text(yaml.safe_dump({"path": str(folder), "train": "images/train", "val": "images/val",
                                     "names": dict(enumerate(n["owned_classes"]))}, sort_keys=False))
        nodes.append({"name": n["name"], "data_yaml": str(y), "owned_classes": n["owned_classes"]})
    ev = out / "data_eval.yaml"
    ev.write_text(yaml.safe_dump({"path": str((DATA / args.use_case / "_eval").resolve()), "train": "images/val",
                                  "val": "images/val", "names": dict(enumerate(meta["global_classes"]))},
                                 sort_keys=False))
    fed = {"mode": "async", "local_epochs": args.local_epochs, "batch_size": 4, "lr0": args.lr0, "warmup_steps": 5,
           "grad_clip": 10.0, "workers": 0, "device": "cpu" if args.cpu else "auto", "balanced_fraction": 1.0,
           "rounds": args.cycles, "async_node_cycles": args.cycles, "staleness_alpha": 0.5,
           "merge_rule": "mean", "eval_interval": 1000, "keep_checkpoints": 2, "lr_schedule": "linear",
           "lrf": 0.01, "final_calibration": True, "pseudo_label": {"enabled": False}}
    if args.scheme == "sequential":     # server-scheduled turns, random order every cycle, mean of the last 4 versions
        fed.update(max_concurrent_nodes=1, async_order="shuffle", swa_last=4)
    else:                               # free-running nodes, merged on arrival; mean of each node's last version
        fed.update(async_order="fixed", swa_last=0, swa_per_node=True, progressive_refresh=True)
    cfg = {"global_classes": meta["global_classes"],
           "model": {"arch": f"configs/arch/yolo26_classic_{args.scale}.yaml", "imgsz": args.imgsz, "pretrained": True},
           "federation": fed, "nodes": nodes, "eval_data_yaml": str(ev),
           "output_dir": str(out / "server_runs"), "seed": 0}
    path = out / f"{args.use_case}_{args.scheme}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--use-case", choices=["kfm12", "wbc", "dentex"], default="wbc")
    ap.add_argument("--scheme", choices=["sequential", "concurrent"], default="sequential")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--local-epochs", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=0.01,
                    help="higher than the study's 0.0008: a demo has only a few images and steps")
    ap.add_argument("--scale", choices=["n", "s", "m", "l"], default="n")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--cpu", action="store_true", help="train on the CPU even if a GPU is available")
    ap.add_argument("--port", type=int, default=8900, help="server port; clients use the next four")
    ap.add_argument("--keep-open", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    out = ROOT / "runs/demo" / f"{args.use_case}_{args.scheme}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "server_state", ignore_errors=True)   # a fresh server for every run
    cfg_path = write_config(args, out)
    meta = yaml.safe_load(cfg_path.read_text())
    env = dict(os.environ, PYTHONNOUSERSITE="1", **({"CUDA_VISIBLE_DEVICES": ""} if args.cpu else {}))
    atexit.register(stop_all)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    server = f"http://127.0.0.1:{args.port}"
    log = open(out / "server.log", "w")
    PROCS.append(subprocess.Popen([sys.executable, "-m", "cofy.net.server_app", "--host", "127.0.0.1",
                                   "--port", str(args.port), "--config-dir", str(out),
                                   "--state-dir", str(out / "server_state"), "--developer-mode",
                                   "--heartbeat-timeout", "600"],
                                  stdout=log, stderr=subprocess.STDOUT, env=env))
    wait_up(f"{server}/api/info", "the server")
    room_id = f"demo-{args.use_case}-{args.scheme}-{int(time.time())}"
    r = requests.post(f"{server}/api/rooms", json={"room_id": room_id, "config_path": str(cfg_path)}, timeout=60)
    r.raise_for_status()
    password = r.json()["password"]

    clients = []
    for i, node in enumerate(meta["nodes"]):
        port = args.port + 1 + i
        clog = open(out / f"client_{node['name']}.log", "w")
        PROCS.append(subprocess.Popen([sys.executable, "-m", "cofy.net.client_app", "--port", str(port)],
                                      stdout=clog, stderr=subprocess.STDOUT, env=env))
        clients.append((node, f"http://127.0.0.1:{port}"))
    for node, url in clients:
        wait_up(f"{url}/api/status", f"client {node['name']}")
        j = requests.post(f"{url}/api/join", json={"server_url": server, "room_id": room_id, "password": password,
                                                   "data_yaml": node["data_yaml"]}, timeout=60)
        if j.status_code != 200:
            sys.exit(f"{node['name']} could not join: {j.text}")
        got = j.json()["node_name"]
        if got != node["name"]:
            sys.exit(f"the server gave slot {got} to the client holding {node['name']}'s data -- restart the demo")
        print(f"  {got} joined, owns {', '.join(node['owned_classes'])}")

    print(f"\nCOFY demo: {args.use_case}, {args.scheme} scheme, {len(clients)} nodes, {args.cycles} cycles each, "
          f"YOLO26-{args.scale.upper()} at {args.imgsz} px")
    print(f"  server dashboard: {server}   (room {room_id})")
    print("  client pages:     " + "  ".join(u for _, u in clients))
    requests.post(f"{server}/api/rooms/{room_id}/start", timeout=30).raise_for_status()
    for _, url in clients:
        requests.post(f"{url}/api/start", timeout=30).raise_for_status()

    t0, last, seen_evals, done_at = time.time(), "", 0, None
    while True:
        time.sleep(3)
        s = requests.get(f"{server}/api/rooms/{room_id}", timeout=30).json()
        cal = s.get("calibration")
        line = (f"  [{(time.time() - t0) / 60:4.1f} min] version {s['round']}/{s['total']}"
                + (f" | turn: {s['turn']}" if s.get("turn") else "")
                + " | " + ", ".join(f"{n}: {v['status']}" for n, v in sorted(s["nodes"].items()))
                + (f" | calibration: {cal['contributed']} statistics, {cal['rows']} row sets"
                   + (f", {cal['refreshed']} refreshes" if cal.get("refresh") else "")
                   + (" -- complete" if cal["complete"] else "") if cal else ""))
        if line.split("] ", 1)[1] != last:
            print(line, flush=True)
            last = line.split("] ", 1)[1]
        for e in s.get("eval_history", [])[seen_evals:]:
            print(f"    server-side evaluation ({e.get('label', '')}): mAP50 {e['map50']:.3f}, "
                  f"mAP50-95 {e['map5095']:.3f}", flush=True)
        seen_evals = len(s.get("eval_history", []))
        done = s["finished"] and (cal is None or cal["complete"])
        done_at = (done_at or time.time()) if done else None
        labels = [e.get("label", "") for e in s.get("eval_history", [])]
        scored = any("calibrated" in x for x in labels) if cal is not None else bool(labels)
        if done and (scored or time.time() - done_at > 180):   # the last evaluation runs in the background
            break
        if all(p.poll() is not None for p in PROCS[1:]) and not s["finished"]:
            sys.exit("every client stopped before the end -- see the client logs in " + str(out))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Models, logs and the server's audit trail: {out}")
    if args.keep_open:
        print("Dashboards stay open; press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
