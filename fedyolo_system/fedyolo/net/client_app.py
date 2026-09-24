"""FastAPI client app: the "personal room" -- one node's local control
panel. Binds 127.0.0.1 by default (it's this machine's own panel); the
actual federation traffic (pull current global weights, push this node's
trained update) goes out from here to the server's IP:port over plain
HTTP, so this machine only ever needs filesystem access to whichever
node's data IT is training on.

Run: python -m fedyolo.net.client_app --port 8901
Then open http://127.0.0.1:8901 and join a room.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from pathlib import Path

import requests
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..client import run_node_round
from ..config import load_config_from_dict

log = logging.getLogger("fedyolo.net.client_app")

app = FastAPI(title="fedyolo client node")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

REQUIRED_YOLO_SUBDIRS = ["images/train", "images/val", "labels/train", "labels/val"]


class Session:
    """One client_app process handles exactly one node -- matching one
    person, one machine, one local dataset. Run a separate client_app
    instance (different port) per simulated node on the same Mac."""

    def __init__(self):
        self.server_url: str | None = None
        self.room_id: str | None = None
        self.password: str | None = None
        self.node_name: str | None = None
        self.data_yaml: str | None = None
        self.owned_classes: list[str] = []
        self.global_classes: list[str] = []
        self.dynamic: bool = False
        self.mode: str | None = None
        self.total: int | None = None
        self.phase = "idle"  # idle | joined | training | uploading | waiting | done | error
        self.detail = ""
        self.cycle = 0
        self.log_lines: list[str] = []
        self.error: str | None = None
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_requested = False

    def log(self, msg: str) -> None:
        with self.lock:
            self.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            self.log_lines = self.log_lines[-300:]
        log.info(msg)

    def set(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "server_url": self.server_url, "room_id": self.room_id, "node_name": self.node_name,
                "data_yaml": self.data_yaml, "owned_classes": self.owned_classes, "mode": self.mode,
                "total": self.total, "phase": self.phase, "detail": self.detail, "cycle": self.cycle,
                "log_lines": list(self.log_lines[-150:]), "error": self.error,
                "running": self.thread is not None and self.thread.is_alive(),
            }


session = Session()


def _validate_local_yaml(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise HTTPException(400, f"file not found on this machine: {path}")
    try:
        d = yaml.safe_load(p.read_text())
    except Exception as e:
        raise HTTPException(400, f"not valid YAML: {e}")
    if not isinstance(d, dict) or "train" not in d or "val" not in d:
        raise HTTPException(400, "data yaml must have 'train' and 'val' keys")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "client_dashboard.html").read_text()


@app.get("/api/status")
def status():
    return session.snapshot()


# ── local filesystem browsing (safe here specifically: 127.0.0.1-bound,
# one user, one machine -- never exposed beyond this client's own port) ──
@app.get("/api/browse")
def browse(path: str | None = None):
    root = Path(path) if path else Path.home()
    if not root.is_dir():
        raise HTTPException(400, f"not a directory: {root}")
    try:
        entries = sorted(
            ({"name": e.name, "is_dir": e.is_dir()} for e in root.iterdir() if not e.name.startswith(".")),
            key=lambda e: (not e["is_dir"], e["name"].lower()),
        )
    except PermissionError:
        raise HTTPException(403, f"permission denied: {root}")
    return {"cwd": str(root), "parent": str(root.parent) if root != root.parent else None, "entries": entries}


@app.post("/api/build_dataset")
def build_dataset(body: dict):
    """From-scratch mode: given class names (in label-index order) and a
    folder already laid out YOLO-style (images/{train,val},
    labels/{train,val} -- the convention every dataset in this project
    uses, and the one ultralytics itself expects), write a data.yaml into
    that folder and return its path."""
    class_names = body["class_names"]
    folder = Path(body["data_folder"])
    if not folder.is_dir():
        raise HTTPException(400, f"not a directory: {folder}")
    missing = [d for d in REQUIRED_YOLO_SUBDIRS if not (folder / d).is_dir()]
    if missing:
        raise HTTPException(
            400,
            f"missing required subfolder(s): {', '.join(missing)} -- see the "
            f"folder-layout guide on this page before building a dataset here",
        )
    data_yaml = {
        "path": str(folder),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": list(class_names),
    }
    out_path = folder / "data.yaml"
    out_path.write_text(yaml.dump(data_yaml))
    return {"data_yaml": str(out_path)}


@app.post("/api/fetch_room")
def fetch_room(body: dict):
    server_url = body["server_url"].rstrip("/")
    room_id = body["room_id"]
    try:
        r = requests.get(f"{server_url}/api/rooms/{room_id}", timeout=10)
    except requests.RequestException as e:
        raise HTTPException(502, f"could not reach server: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return {"room": r.json()}


@app.post("/api/join")
def join(body: dict):
    server_url = body["server_url"].rstrip("/")
    room_id = body["room_id"]
    password = body.get("password", "")
    data_yaml = body["data_yaml"]
    class_names = body.get("class_names")  # only meaningful for dynamic rooms

    _validate_local_yaml(data_yaml)

    try:
        r = requests.post(
            f"{server_url}/api/rooms/{room_id}/join",
            json={"password": password, "class_names": class_names}, timeout=10,
        )
    except requests.RequestException as e:
        raise HTTPException(502, f"could not reach server: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    info = r.json()

    session.set(
        server_url=server_url, room_id=room_id, password=password,
        node_name=info["node_name"], data_yaml=data_yaml,
        owned_classes=info["owned_classes"], global_classes=info["global_classes"],
        dynamic=info.get("dynamic", False),
        mode=info["mode"], total=info["rounds_or_cycles"], phase="joined", error=None, cycle=0,
    )
    session.log(f"joined room '{room_id}' as node '{info['node_name']}' ({info['mode']} mode, "
                f"owns {info['owned_classes']}) -- auto-assigned by the server")
    return session.snapshot()


@app.post("/api/add_classes")
def add_classes(body: dict):
    """Async, dynamic rooms only: declare additional classes this
    already-joined node wants to contribute from now on."""
    s = session
    if s.node_name is None:
        raise HTTPException(400, "join a room first")
    try:
        r = requests.post(
            f"{s.server_url}/api/rooms/{s.room_id}/add_classes",
            headers={"X-Room-Password": s.password or ""},
            json={"node_name": s.node_name, "class_names": body["class_names"]}, timeout=10,
        )
    except requests.RequestException as e:
        raise HTTPException(502, f"could not reach server: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    info = r.json()
    session.set(owned_classes=info["owned_classes"])
    session.log(f"declared additional classes: {body['class_names']} -> owned now: {info['owned_classes']}")
    return session.snapshot()


@app.post("/api/start")
def start_training():
    with session.lock:
        if session.node_name is None:
            raise HTTPException(400, "join a room first")
        if session.thread is not None and session.thread.is_alive():
            raise HTTPException(409, "already running")
        session.stop_requested = False
        t = threading.Thread(target=_run_loop, daemon=True)
        session.thread = t
    t.start()
    return {"started": True}


@app.post("/api/stop")
def stop_training():
    session.set(stop_requested=True)
    session.log("stop requested -- will halt after the current cycle")
    return {"stopping": True}


@app.post("/api/shutdown")
def shutdown():
    def _die():
        time.sleep(0.4)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return {"shutting_down": True}


def _pull(server_url: str, room_id: str, node_name: str) -> tuple[dict, int]:
    r = requests.get(f"{server_url}/api/rooms/{room_id}/pull", params={"node": node_name}, timeout=120)
    r.raise_for_status()
    payload = torch.load(io.BytesIO(r.content), map_location="cpu", weights_only=False)
    return payload["state_dict"], payload["round_or_version"]


def _push(server_url: str, room_id: str, node_name: str, password: str, result, pulled_round_or_version: int) -> dict:
    buf = io.BytesIO()
    torch.save(result, buf)
    buf.seek(0)
    r = requests.post(
        f"{server_url}/api/rooms/{room_id}/push",
        params={"node": node_name, "pulled_round_or_version": pulled_round_or_version},
        headers={"X-Room-Password": password or ""},
        files={"file": ("result.pt", buf, "application/octet-stream")},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def _run_loop() -> None:
    s = session
    try:
        raw_cfg = requests.get(f"{s.server_url}/api/rooms/{s.room_id}/config", timeout=10).json()
        for n in raw_cfg["nodes"]:
            if n["name"] == s.node_name:
                n["data_yaml"] = s.data_yaml  # the only thing this client overrides
        cfg = load_config_from_dict(raw_cfg, partial=s.dynamic)
        node_cfg = next(n for n in cfg.nodes if n.name == s.node_name)
    except Exception as e:
        session.set(phase="error", error=f"could not load room config: {e}")
        session.log(f"ERROR loading config: {e}")
        return

    optimizer_state = None
    cycle = 0
    total = s.total or 10**9
    while not session.stop_requested and cycle < total:
        try:
            session.set(phase="training", detail="downloading global weights")
            global_sd, pulled_marker = _pull(s.server_url, s.room_id, s.node_name)

            session.set(detail=f"training locally (cycle {cycle + 1}/{s.total})")
            session.log(f"cycle {cycle + 1}/{s.total}: pulled round/version {pulled_marker}, training...")
            # round_idx must match what the in-process path would seed with:
            # sync shares ONE round counter across every node (pulled_marker
            # IS that counter); async seeds off each node's OWN local cycle
            # count, never the shared global version pulled_marker is there.
            seed_round_idx = pulled_marker if s.mode == "sync" else cycle
            result = run_node_round(
                node=node_cfg, cfg=cfg, global_state_dict=global_sd,
                round_idx=seed_round_idx, teacher_state_dict=None,
                optimizer_state=optimizer_state,
            )
            optimizer_state = result.optimizer_state

            session.set(phase="uploading", detail="uploading trained update")
            outcome = _push(s.server_url, s.room_id, s.node_name, s.password, result, pulled_marker)
            cycle += 1
            session.set(cycle=cycle, phase="waiting", detail=str(outcome))
            session.log(f"cycle {cycle}/{s.total} pushed -> {outcome}")

            if outcome.get("finished"):
                session.set(phase="done", detail="federation finished")
                session.log("federation finished")
                break

            if s.mode == "sync" and not outcome.get("round_advanced"):
                session.set(detail="waiting for other nodes to finish this round")
                while not session.stop_requested:
                    try:
                        snap = requests.get(f"{s.server_url}/api/rooms/{s.room_id}", timeout=30).json()
                    except requests.RequestException as e:
                        session.log(f"status poll failed ({e}), retrying...")
                        time.sleep(3)
                        continue
                    if snap["round"] > pulled_marker or snap["finished"]:
                        break
                    time.sleep(3)
        except Exception as e:
            session.set(phase="error", error=str(e))
            session.log(f"ERROR: {e}")
            break

    if not session.stop_requested and session.phase != "error":
        session.set(phase="done")
        session.log("done (reached configured rounds/cycles)")


def main():
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="fedyolo client node -- personal room")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8901)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
