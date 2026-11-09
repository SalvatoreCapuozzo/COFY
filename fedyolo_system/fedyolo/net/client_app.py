"""FastAPI client app: the "personal room" -- one node's local control
panel. Binds 127.0.0.1 by default (it's this machine's own panel); the
federation traffic (pull the global model, push this node's update) goes to
the server over HTTP or HTTPS, so this machine only ever needs filesystem
access to its own node's data.

Robustness: every server call is retried with back-off (a network blip or
a server restart doesn't end the node), a heartbeat thread keeps the node
marked alive while it trains, and each push carries a push_id so a retried
upload is never aggregated twice. A restarted client can reclaim its node
slot by name once the old session has gone silent.

HTTPS with a self-signed/private CA: start with --ca-cert /path/ca.pem.

Run: python -m fedyolo.net.client_app --port 8901
Then open http://127.0.0.1:8901 and join a room.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path

import requests
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..client import run_node_round
from ..config import load_config_from_dict
from . import protocol

log = logging.getLogger("fedyolo.net.client_app")

app = FastAPI(title="fedyolo client node")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

REQUIRED_YOLO_SUBDIRS = ["images/train", "images/val", "labels/train", "labels/val"]
FIXED_YAML_DIR = Path.home() / ".fedyolo_client" / "data"
VERIFY: bool | str = True          # requests' TLS verification (--ca-cert sets a CA bundle path)
HEARTBEAT_EVERY = 20.0


class StopRequested(Exception):
    pass


class Session:
    """One client_app process handles exactly one node."""

    def __init__(self):
        self.server_url: str | None = None
        self.room_id: str | None = None
        self.password: str | None = None
        self.node_name: str | None = None
        self.node_token: str | None = None
        self.data_yaml: str | None = None
        self.owned_classes: list[str] = []
        self.global_classes: list[str] = []
        self.dynamic: bool = False
        self.mode: str | None = None
        self.total: int | None = None
        self.phase = "idle"  # idle | joined | training | uploading | waiting | reconnecting | done | error
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

    def headers(self) -> dict:
        return {"X-Room-Password": self.password or "", "X-Node-Token": self.node_token or ""}

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


# ── HTTP with retries ────────────────────────────────────────────────────────
def _call(method: str, url: str, *, retry: bool = True, what: str = "request", **kw) -> requests.Response:
    """Server call. Connection errors, timeouts and 5xx are retried with
    exponential back-off (2 s .. 60 s) until they succeed or the user stops
    the node; 4xx answers are raised immediately (they won't fix themselves),
    except 409 'room not started' which is waited out, and 409 'not your turn'
    (sequential scheme: another node is training) which is polled every 10 s."""
    attempt = 0
    while True:
        if session.stop_requested and retry:
            raise StopRequested()
        try:
            r = requests.request(method, url, verify=VERIFY, **kw)
            if r.status_code == 409 and "not your turn" in r.text:
                session.set(phase="waiting", detail=r.json().get("detail", r.text)[:200])
                for _ in range(10):
                    if session.stop_requested:
                        raise StopRequested()
                    time.sleep(1)
                continue
            if r.status_code < 500 and not (r.status_code == 409 and "not started" in r.text):
                if r.status_code >= 400:
                    raise RuntimeError(f"{what} rejected by server ({r.status_code}): {r.text[:300]}")
                if attempt:
                    session.log(f"{what}: server reachable again")
                return r
            problem = f"HTTP {r.status_code}: {r.text[:120]}"
        except requests.RequestException as e:
            problem = f"{type(e).__name__}"
        if not retry:
            raise RuntimeError(f"{what} failed: {problem}")
        attempt += 1
        wait = min(60, 2 ** attempt)
        session.set(phase="reconnecting", detail=f"{what}: {problem} -- retry {attempt} in {wait}s")
        session.log(f"{what} failed ({problem}); retrying in {wait}s")
        for _ in range(wait):
            if session.stop_requested:
                raise StopRequested()
            time.sleep(1)


def _room_url(path: str = "") -> str:
    return f"{session.server_url}/api/rooms/{session.room_id}{path}"


# ── local data.yaml checks ───────────────────────────────────────────────────
def _resolve_data_yaml(path: str, owned_classes: list[str] | None = None) -> tuple[str, str | None]:
    """Check a local data.yaml before training and return (path to use, note).

    If the yaml's `path:` doesn't exist on this machine (e.g. a dataset
    copied from another computer) but its train/val folders sit next to the
    yaml, a corrected copy is written under ~/.fedyolo_client/data/ and used
    instead -- the original file is never modified. With owned_classes,
    also checks that `names` matches them exactly (fedyolo requires it)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise HTTPException(400, f"file not found on this machine: {path}")
    p = p.resolve()
    try:
        d = yaml.safe_load(p.read_text())
    except Exception as e:
        raise HTTPException(400, f"not valid YAML: {e}")
    if not isinstance(d, dict) or not all(k in d for k in ("train", "val", "names")):
        raise HTTPException(400, "data yaml must have 'train', 'val' and 'names' keys")

    def ok(root: Path) -> bool:
        return (root / d["train"]).exists() and (root / d["val"]).exists()

    root = Path(d["path"]) if d.get("path") else p.parent
    if not root.is_absolute():
        root = p.parent / root
    note = None
    if not ok(root):
        if ok(p.parent):
            note = f"'path: {d.get('path')}' does not exist on this machine -- using the yaml's own folder {p.parent}"
            root = p.parent
        else:
            raise HTTPException(400, f"training/validation images not found: {root / d['train']} / {root / d['val']}")
    if owned_classes is not None:
        names = d["names"]
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if names != list(owned_classes):
            raise HTTPException(400, f"the dataset's class names {names} must exactly match this node's classes "
                                     f"{list(owned_classes)} (same order)")
    if str(root) == str(d.get("path")):
        return str(p), note
    fixed = dict(d, path=str(root))
    FIXED_YAML_DIR.mkdir(parents=True, exist_ok=True)
    out = FIXED_YAML_DIR / f"{hashlib.sha1(str(p).encode()).hexdigest()[:10]}_{p.name}"
    out.write_text(yaml.safe_dump(fixed, sort_keys=False))
    return str(out), note or f"absolute dataset path written to {out}"


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "client_dashboard.html").read_text()


@app.get("/api/status")
def status():
    return session.snapshot()


# ── local filesystem browsing (safe: 127.0.0.1-bound, one user, one machine) ─
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
    """Write a data.yaml into a folder laid out YOLO-style
    (images/{train,val}, labels/{train,val}); class names in label-index order."""
    class_names = body["class_names"]
    folder = Path(body["data_folder"]).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400, f"not a directory: {folder}")
    missing = [d for d in REQUIRED_YOLO_SUBDIRS if not (folder / d).is_dir()]
    if missing:
        raise HTTPException(400, f"missing required subfolder(s): {', '.join(missing)} -- see the "
                                 f"folder-layout guide on this page before building a dataset here")
    out_path = folder / "data.yaml"
    out_path.write_text(yaml.dump({"path": str(folder), "train": "images/train", "val": "images/val",
                                   "nc": len(class_names), "names": list(class_names)}))
    return {"data_yaml": str(out_path)}


@app.post("/api/fetch_room")
def fetch_room(body: dict):
    server_url = body.get("server_url", session.server_url or "").rstrip("/")
    room_id = body.get("room_id", session.room_id)
    password = body.get("password") or session.password or ""
    try:
        r = requests.get(f"{server_url}/api/rooms/{room_id}", headers={"X-Room-Password": password},
                         timeout=10, verify=VERIFY)
    except requests.RequestException as e:
        raise HTTPException(502, f"could not reach server: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return {"room": r.json()}


def _membership_valid() -> bool:
    """True if the server still recognizes this session's node and token
    (false after the room was deleted/recreated or the password changed)."""
    try:
        r = requests.post(_room_url("/heartbeat"), params={"node": session.node_name},
                          headers=session.headers(), timeout=10, verify=VERIFY)
        return r.status_code == 200 and not r.json().get("finished")
    except requests.RequestException:
        return False


@app.post("/api/join")
def join(body: dict):
    server_url = body["server_url"].rstrip("/")
    room_id = body["room_id"]
    password = body.get("password", "")
    data_yaml = body["data_yaml"]
    class_names = body.get("class_names")      # dynamic rooms only
    reclaim = (body.get("node_name") or "").strip() or None
    _resolve_data_yaml(data_yaml)              # basic checks before touching the server

    if (session.node_name and session.server_url == server_url and session.room_id == room_id
            and session.password == password and not reclaim
            and not (session.thread and session.thread.is_alive()) and _membership_valid()):
        # already joined this room: just switch the local dataset (e.g. after fixing it)
        resolved, note = _resolve_data_yaml(data_yaml, session.owned_classes)
        session.set(data_yaml=resolved, error=None, phase="joined")
        session.log(f"dataset updated for node '{session.node_name}'" + (f" ({note})" if note else ""))
        return session.snapshot()

    try:
        r = requests.post(f"{server_url}/api/rooms/{room_id}/join", timeout=15, verify=VERIFY,
                          json={"password": password, "class_names": class_names, "node_name": reclaim})
    except requests.RequestException as e:
        raise HTTPException(502, f"could not reach server: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    info = r.json()
    session.set(server_url=server_url, room_id=room_id, password=password, node_name=info["node_name"],
                node_token=info["node_token"], owned_classes=info["owned_classes"],
                global_classes=info["global_classes"], dynamic=info.get("dynamic", False), mode=info["mode"],
                total=info["rounds_or_cycles"], cycle=info.get("round_or_cycle", 0), phase="joined", error=None)
    session.log(f"{'reclaimed' if reclaim else 'joined'} room '{room_id}' as node '{info['node_name']}' "
                f"({info['mode']} mode, owns {info['owned_classes']})")
    try:
        resolved, note = _resolve_data_yaml(data_yaml, info["owned_classes"])
    except HTTPException as e:
        session.set(phase="error", error=f"dataset does not fit this node: {e.detail}")
        session.log(f"ERROR: {e.detail} -- fix the dataset and press Join again (you keep node '{info['node_name']}')")
        raise
    session.set(data_yaml=resolved)
    if note:
        session.log(note)
    return session.snapshot()


@app.post("/api/add_classes")
def add_classes(body: dict):
    """Async dynamic rooms only: declare more classes for this node."""
    if session.node_name is None:
        raise HTTPException(400, "join a room first")
    try:
        r = requests.post(_room_url("/add_classes"), headers=session.headers(), timeout=10, verify=VERIFY,
                          json={"node_name": session.node_name, "class_names": body["class_names"]})
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
        if session.error:
            raise HTTPException(400, f"fix this first: {session.error}")
        if session.thread is not None and session.thread.is_alive():
            raise HTTPException(409, "already running")
        session.stop_requested = False
        t = threading.Thread(target=_run_loop, daemon=True)
        session.thread = t
    t.start()
    threading.Thread(target=_heartbeat_loop, args=(t,), daemon=True).start()
    return {"started": True}


@app.post("/api/stop")
def stop_training():
    session.set(stop_requested=True)
    session.log("stop requested -- will halt after the current local round")
    return {"stopping": True}


@app.post("/api/shutdown")
def shutdown():
    def _die():
        time.sleep(0.4)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return {"shutting_down": True}


# ── the training loop ────────────────────────────────────────────────────────
def _heartbeat_loop(worker: threading.Thread) -> None:
    while worker.is_alive():
        try:
            requests.post(_room_url("/heartbeat"), params={"node": session.node_name},
                          headers=session.headers(), timeout=10, verify=VERIFY)
        except requests.RequestException:
            pass
        time.sleep(HEARTBEAT_EVERY)


def _pull() -> tuple[dict, int]:
    r = _call("GET", _room_url("/pull"), params={"node": session.node_name}, headers=session.headers(),
              timeout=300, what="download global model")
    return protocol.decode_global(r.content)


def _push(result, pulled: int, push_id: str) -> dict:
    payload = protocol.encode_update(result)
    r = _call("POST", _room_url("/push"),
              params={"node": session.node_name, "pulled_round_or_version": pulled, "push_id": push_id},
              headers=session.headers(), files={"file": ("update.pt", payload, "application/octet-stream")},
              timeout=600, what="upload update")
    return r.json()


def _room_snapshot() -> dict:
    return _call("GET", _room_url(), headers=session.headers(), timeout=30, what="room status").json()


def _calibration_phase(cfg, node_cfg) -> None:
    """After training: wait for the room's final model, then take part in its calibration phase
    (fedyolo/net/rooms.py): BatchNorm statistics of the final model on this node's images, then
    this node's own class rows re-fitted on the pooled model (and once more, if the room refreshes)."""
    import torch
    from ..calibrate import node_bn_stats, node_head_align
    s = session
    snap = {}
    while not s.stop_requested:
        snap = _room_snapshot()
        if snap.get("finished"):
            break
        s.set(phase="waiting", detail="waiting for the other nodes to finish training")
        time.sleep(10)
    # the server marks the room finished a moment before it opens the calibration phase (it saves the
    # last checkpoint in between): a node that asks in that gap waits for the phase instead of leaving
    t0 = time.time()
    while (not s.stop_requested and cfg.federation.final_calibration and not snap.get("calibration")
           and time.time() - t0 < 300):
        time.sleep(2)
        snap = _room_snapshot()
    if s.stop_requested or not snap.get("calibration"):
        return
    device = torch.device(cfg.federation.device)
    while not s.stop_requested:
        st = _call("GET", _room_url("/calib"), params={"node": s.node_name}, headers=s.headers(),
                   timeout=30, what="calibration status").json()
        stage = st["stage"]
        if stage in ("done", "none"):
            s.log(f"calibration done for this node ({st.get('contributed', 0)}/{st.get('of', 0)} nodes contributed"
                  + (", complete" if st.get("complete") else ", the others continue") + ")")
            return
        if stage == "wait":
            s.set(phase="calibrating", detail=f"waiting for the other nodes ({st['contributed']}/{st['of']} "
                                              f"statistics, {st['rows']} rows)")
            time.sleep(5)
            continue
        r = _call("GET", _room_url("/calib/model"), params={"node": s.node_name}, headers=s.headers(),
                  timeout=300, what="calibration model")
        sd, _ = protocol.decode_global(r.content)
        if stage == "bn":
            s.set(phase="calibrating", detail="BatchNorm statistics on the local images")
            stats, n = node_bn_stats(node_cfg, cfg, sd, device)
            payload, path = protocol.encode_bn_stats(stats, n), "/calib/bn"
        else:
            s.set(phase="calibrating", detail=f"re-fitting this node's class rows ({stage})")
            rows, _ = node_head_align(node_cfg, cfg, sd, device, cfg.federation.calibration_epochs)
            payload, path = protocol.encode_rows(rows), "/calib/rows"
        try:
            out = _call("POST", _room_url(path), params={"node": s.node_name}, headers=s.headers(),
                        files={"file": ("calib.pt", payload, "application/octet-stream")}, timeout=600,
                        what=f"calibration upload ({stage})").json()
            s.log(f"calibration: {stage} sent -> next {out.get('stage')}")
        except RuntimeError as e:   # a retried upload the server already has: ask for the stage again
            if "not expected" not in str(e):
                raise
            s.log(f"calibration: {stage} already received by the server")


def _run_loop() -> None:
    s = session
    try:
        raw_cfg = _call("GET", _room_url("/config"), headers=s.headers(), timeout=30, what="room config").json()
        for n in raw_cfg["nodes"]:
            if n["name"] == s.node_name:
                n["data_yaml"] = s.data_yaml  # the only thing this client overrides
        cfg = load_config_from_dict(raw_cfg, partial=s.dynamic)
        node_cfg = next(n for n in cfg.nodes if n.name == s.node_name)
    except StopRequested:
        s.set(phase="joined", detail="stopped")
        return
    except Exception as e:
        s.set(phase="error", error=f"could not load room config: {e}")
        s.log(f"ERROR loading config: {e}")
        return

    pl = cfg.federation.pseudo_label
    optimizer_state = None         # momentum stays on this machine, across rounds/cycles
    prev_global = None             # sync teacher = the previous round's global model
    cycle = s.cycle
    total = s.total or 10**9
    while not s.stop_requested and cycle < total:
        try:
            s.set(phase="training", detail="downloading global weights")
            global_sd, pulled = _pull()
            s.set(phase="training", detail=f"training locally (cycle {cycle + 1}/{s.total})")
            s.log(f"cycle {cycle + 1}/{s.total}: pulled round/version {pulled}, training...")
            # seeding matches the in-process path: sync uses the shared round
            # counter, async each node's own cycle count
            round_idx = pulled if s.mode == "sync" else cycle
            teacher = (prev_global if s.mode == "sync" else global_sd) if pl.enabled else None
            result = run_node_round(node=node_cfg, cfg=cfg, global_state_dict=global_sd, round_idx=round_idx,
                                    teacher_state_dict=teacher, optimizer_state=optimizer_state)
            optimizer_state = result.optimizer_state
            prev_global = global_sd if s.mode == "sync" else None

            s.set(phase="uploading", detail="uploading trained update")
            outcome = _push(result, pulled, push_id=f"{s.node_name}-{pulled}-{cycle}")
            cycle += 1
            s.set(cycle=cycle, phase="waiting", detail=str(outcome))
            s.log(f"cycle {cycle}/{s.total} pushed -> {outcome}")
            if outcome.get("finished"):
                s.set(phase="done", detail="federation finished")
                s.log("federation finished")
                break
            if s.mode == "sync" and not outcome.get("round_advanced"):
                s.set(phase="waiting", detail="waiting for the other nodes to finish this round")
                snap = {}
                while not s.stop_requested:
                    snap = _room_snapshot()
                    if snap["round"] > pulled or snap["finished"]:
                        break
                    time.sleep(3)
                if snap.get("finished"):
                    s.set(phase="done", detail="federation finished")
                    s.log("federation finished")
                    break
        except StopRequested:
            break
        except Exception as e:
            s.set(phase="error", error=str(e))
            s.log(f"ERROR: {e}")
            break

    if not s.stop_requested and s.phase != "error":
        try:
            _calibration_phase(cfg, node_cfg)
        except StopRequested:
            pass
        except Exception as e:
            s.set(phase="error", error=f"calibration: {e}")
            s.log(f"ERROR in calibration: {e}")
    if s.stop_requested:
        s.set(phase="joined", detail="stopped by user")
        s.log("stopped")
    elif s.phase not in ("error", "done"):
        s.set(phase="done")
        s.log("done (reached configured rounds/cycles)")


def main():
    import argparse

    import uvicorn

    global VERIFY
    p = argparse.ArgumentParser(description="fedyolo client node -- personal room")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--ca-cert", help="CA certificate (PEM) to trust for an HTTPS server with a private certificate")
    args = p.parse_args()
    if args.ca_cert:
        VERIFY = args.ca_cert
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
