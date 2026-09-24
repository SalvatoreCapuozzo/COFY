"""FastAPI server app: room registry, sync/async round orchestration over
HTTP, and the server dashboard.

Binds 0.0.0.0 by default -- reachable from 127.0.0.1 (simulating several
nodes on one Mac) and from any other machine on the same LAN (a real
second Mac) with zero code differences; see rooms.py for the room design
this wraps.

Run: python -m fedyolo.net.server_app --port 8900 [--config-dir configs]
"""

from __future__ import annotations

import io
import logging
import os
import signal
import threading
import time
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..client import NodeRoundResult
from .rooms import RoomRegistry

log = logging.getLogger("fedyolo.net.server_app")

app = FastAPI(title="fedyolo federation server")
registry = RoomRegistry()
CONFIG_DIR = Path("configs")  # overridden by main()'s --config-dir

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _get_room(room_id: str):
    try:
        return registry.get(room_id)
    except KeyError:
        raise HTTPException(404, f"no such room '{room_id}'")


def _require_password(room, x_room_password: str | None) -> None:
    if not room.check_password(x_room_password or ""):
        raise HTTPException(403, "wrong or missing room password")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "server_dashboard.html").read_text()


@app.get("/room/{room_id}", response_class=HTMLResponse)
def room_page(room_id: str):
    return (STATIC_DIR / "server_dashboard.html").read_text()


# ── config browsing (for the "create static room" dropdown + preview) ──────
@app.get("/api/configs")
def list_configs():
    if not CONFIG_DIR.is_dir():
        return {"configs": []}
    return {"configs": sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))}


@app.get("/api/configs/{name}")
def preview_config(name: str):
    path = CONFIG_DIR / name
    if not path.is_file() or path.suffix != ".yaml":
        raise HTTPException(404, "no such config")
    return {"name": name, "text": path.read_text()}


# ── room CRUD ────────────────────────────────────────────────────────────────
@app.post("/api/rooms")
def create_room(body: dict):
    room_id = body["room_id"]
    try:
        if body.get("dynamic"):
            room = registry.create(room_id, dynamic_spec=body["dynamic_spec"])
        else:
            config_path = body["config_path"]
            if not Path(config_path).exists():
                raise HTTPException(404, f"config not found on server: {config_path}")
            room = registry.create(room_id, config_path)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return room.snapshot() | {"password": room.password}


@app.get("/api/rooms")
def list_rooms():
    return {"rooms": registry.list()}


@app.delete("/api/rooms/{room_id}")
def delete_room(room_id: str):
    try:
        registry.delete(room_id)
    except KeyError:
        raise HTTPException(404, "no such room")
    except PermissionError as e:
        raise HTTPException(409, str(e))
    return {"deleted": room_id}


@app.get("/api/rooms/{room_id}")
def room_status(room_id: str):
    return _get_room(room_id).snapshot()


@app.get("/api/rooms/{room_id}/password")
def get_password(room_id: str):
    return {"password": _get_room(room_id).password}


@app.post("/api/rooms/{room_id}/password/refresh")
def refresh_password(room_id: str):
    import secrets
    room = _get_room(room_id)
    room.password = secrets.token_urlsafe(9)
    return {"password": room.password}


@app.get("/api/rooms/{room_id}/config")
def room_config(room_id: str):
    """The room's full raw config dict, verbatim -- clients build an
    identical FedYoloConfig from this via config.load_config_from_dict(),
    overriding only their own node's data_yaml. For dynamic rooms (no
    on-disk YAML), built fresh from the live cfg each call so it always
    reflects whatever classes/nodes have registered so far."""
    room = _get_room(room_id)
    if room.raw_config is not None:
        return room.raw_config
    import dataclasses
    cfg = room.cfg
    return {
        "global_classes": list(cfg.global_classes),
        "model": dataclasses.asdict(cfg.model),
        "federation": dataclasses.asdict(cfg.federation),
        "nodes": [dataclasses.asdict(n) for n in cfg.nodes],
        "output_dir": cfg.output_dir,
        "seed": cfg.seed,
    }


# ── node lifecycle ───────────────────────────────────────────────────────────
@app.post("/api/rooms/{room_id}/join")
def join_room(room_id: str, body: dict, request: Request):
    room = _get_room(room_id)
    _require_password(room, body.get("password"))
    try:
        info = room.join(_client_ip(request), class_names=body.get("class_names"))
    except (PermissionError, ValueError) as e:
        raise HTTPException(409, str(e))
    return {
        **info,
        "global_classes": room.cfg.global_classes,
        "mode": room.mode,
        "max_classes": room.max_classes,
        "dynamic": room.dynamic,
        "rounds_or_cycles": room.total_rounds if room.mode == "sync" else room.cfg.federation.async_node_cycles,
    }


@app.post("/api/rooms/{room_id}/add_classes")
def add_classes(room_id: str, body: dict, x_room_password: str | None = Header(default=None)):
    room = _get_room(room_id)
    _require_password(room, x_room_password)
    try:
        return room.add_classes(body["node_name"], body["class_names"])
    except (PermissionError, KeyError, ValueError) as e:
        raise HTTPException(409, str(e))


@app.post("/api/rooms/{room_id}/start")
def start_room(room_id: str):
    room = _get_room(room_id)
    room.start()
    return room.snapshot()


@app.post("/api/rooms/{room_id}/finish")
def finish_room(room_id: str):
    """Explicit operator action to close a dynamic async room, which
    never auto-finishes on its own (see rooms.py's async_push) since a
    new node can join and start contributing at any point."""
    room = _get_room(room_id)
    room.finish()
    return room.snapshot()


# ── checkpoint history / rollback ───────────────────────────────────────────
@app.get("/api/rooms/{room_id}/checkpoints")
def list_checkpoints(room_id: str):
    return {"checkpoints": _get_room(room_id).list_checkpoints()}


@app.post("/api/rooms/{room_id}/checkpoints/serve")
def set_serving_checkpoint(room_id: str, body: dict):
    room = _get_room(room_id)
    try:
        room.set_serving_checkpoint(body.get("path"))
    except KeyError:
        raise HTTPException(404, "no such checkpoint")
    return room.snapshot()


@app.post("/api/rooms/{room_id}/checkpoints/rollback")
def rollback(room_id: str, body: dict):
    room = _get_room(room_id)
    try:
        room.rollback(body["path"])
    except KeyError:
        raise HTTPException(404, "no such checkpoint")
    return room.snapshot()


# ── pull / push ──────────────────────────────────────────────────────────────
@app.get("/api/rooms/{room_id}/pull")
def pull(room_id: str, node: str):
    room = _get_room(room_id)
    if not room.started:
        raise HTTPException(409, "room not started yet")
    room.set_status(node, "downloading")
    sd, round_or_version = room.sync_pull(node) if room.mode == "sync" else room.async_pull(node)

    buf = io.BytesIO()
    torch.save({"state_dict": sd, "round_or_version": round_or_version}, buf)
    size = buf.tell()
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/octet-stream",
        headers={"X-Round-Or-Version": str(round_or_version), "Content-Length": str(size)},
    )


@app.post("/api/rooms/{room_id}/push")
async def push(
    room_id: str, node: str, pulled_round_or_version: int, file: UploadFile,
    x_room_password: str | None = Header(default=None),
):
    room = _get_room(room_id)
    _require_password(room, x_room_password)
    room.set_status(node, "uploading")
    raw = await file.read()
    # weights_only=False: this file is a whole NodeRoundResult dataclass
    # (state_dict + plain-Python metadata), not just tensors -- our own
    # client's own output, trusted the same way a locally-produced
    # checkpoint already is everywhere else in this project.
    result: NodeRoundResult = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)

    outcome = (
        room.sync_push(node, result, pulled_round_or_version)
        if room.mode == "sync"
        else room.async_push(node, result, pulled_round_or_version)
    )

    room.set_status(node, "done" if outcome.get("finished") or outcome.get("node_done") else "waiting")
    return outcome


@app.websocket("/ws/{room_id}")
async def ws_room(websocket: WebSocket, room_id: str):
    try:
        room = registry.get(room_id)
    except KeyError:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    q = room.subscribe()
    try:
        await websocket.send_json({"kind": "snapshot", **room.snapshot()})
        while True:
            ev = await q.get()
            await websocket.send_json(ev.to_json())
    except WebSocketDisconnect:
        pass
    finally:
        room.unsubscribe(q)


# ── shutdown via the UI ──────────────────────────────────────────────────────
@app.post("/api/shutdown")
def shutdown():
    def _die():
        time.sleep(0.4)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return {"shutting_down": True}


def main():
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="fedyolo federation server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--config-dir", default="configs", help="directory the config dropdown/preview lists from")
    p.add_argument("--config", help="create+start a static room immediately from this config")
    p.add_argument("--room", default="default", help="room id to create if --config is given")
    args = p.parse_args()

    global CONFIG_DIR
    CONFIG_DIR = Path(args.config_dir)

    if args.config:
        room = registry.create(args.room, args.config)
        log.info("pre-created room '%s' from %s (password: %s)", args.room, args.config, room.password)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
