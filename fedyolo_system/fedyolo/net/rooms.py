"""Room: one FedYoloConfig, live and network-orchestrated.

Two room types:

- **Static** -- created from an existing fedyolo config YAML, exactly like
  fedyolo.simulate's in-process runs: nodes/global_classes fixed and
  server-authoritative. federation.mode decides sync vs async, same as
  everywhere else in this project.
- **Dynamic** -- created from scratch with just a `max_classes` ceiling,
  model arch/imgsz, and federation hyperparameters. No predefined nodes
  or classes: the model is built once with nc=max_classes (a model's
  output-channel count can't grow after construction), and classes get
  "activated" as clients declare them on join -- global class ids are
  assigned in order of first appearance across node registrations.
  Unclaimed slots are handled by the existing per-class aggregation
  machinery already (fedyolo.model.aggregate_per_class's own docstring:
  "classes with zero contributing nodes keep their randomly initialized
  value... handled defensively") -- no core aggregation changes needed.

Sync rooms implement their own small round barrier (buffer submissions
until every node has pushed for the current round, then call the SAME
aggregate() FedServer.run() uses). Async rooms wrap one real
AsyncFedServer instance and forward pull()/push() onto its existing,
already-thread-safe methods.

Known simplification vs. the in-process path: pseudo-labeling (the
"teacher" distillation pass) is not wired through the network protocol --
every config in this project has federation.pseudo_label.enabled: false,
so run_node_round() is always called here with teacher_state_dict=None.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ..client import NodeRoundResult
from ..config import FederationConfig, FedYoloConfig, ModelConfig, NodeConfig, _validate, load_config
from ..model import aggregate, build_model
from ..server import AsyncFedServer, _run_eval

log = logging.getLogger("fedyolo.net.rooms")

# How many recent global-state snapshots an async room keeps in memory so a
# late push's staleness delta can be computed without the client having to
# re-upload the (possibly 100s of MB) state_dict it pulled.
_PULL_CACHE_SIZE = 30

# Checkpoint history depth. Confirmed with the project owner: bounded, not
# unlimited -- this project has hit real "No space left on device" crashes
# from unbounded checkpoint retention before (see the field report).
_CHECKPOINT_HISTORY_SIZE = 10

_UNCLAIMED_PREFIX = "__unclaimed_"


def _placeholder(i: int) -> str:
    return f"{_UNCLAIMED_PREFIX}{i}__"


def _is_placeholder(name: str) -> bool:
    return name.startswith(_UNCLAIMED_PREFIX)


@dataclasses.dataclass
class NodeStatus:
    name: str
    ip: str | None = None
    status: str = "unclaimed"  # unclaimed | idle | training | uploading | waiting | done
    last_seen: float = 0.0
    last_metrics: dict | None = None
    round_or_cycle: int = 0
    detail: str = ""
    owned_classes: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RoomEvent:
    ts: float
    kind: str
    data: dict = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return {"ts": self.ts, "kind": self.kind, **self.data}


@dataclasses.dataclass
class CheckpointEntry:
    path: Path
    marker: int  # round (sync) or version (async)
    label: str
    created: float


class Room:
    def __init__(
        self,
        room_id: str,
        config_path: str | None = None,
        *,
        dynamic_spec: dict | None = None,
    ):
        """Exactly one of config_path (static) / dynamic_spec (dynamic)
        should be given. dynamic_spec keys: max_classes, model (dict for
        ModelConfig), federation (dict for FederationConfig),
        eval_data_yaml (optional str), output_dir, seed."""
        self.room_id = room_id
        self.password = secrets.token_urlsafe(9)
        self.dynamic = dynamic_spec is not None
        self.eval_data_yaml: str | None = None

        if self.dynamic:
            self.config_path = None
            self.max_classes = int(dynamic_spec["max_classes"])
            model = ModelConfig(**dynamic_spec.get("model", {}))
            fed_raw = dict(dynamic_spec.get("federation", {}))
            fed_raw.setdefault("mode", "sync")
            federation = FederationConfig(**fed_raw)
            from ..config import _resolve_device
            federation.device = _resolve_device(federation.device)
            self.eval_data_yaml = dynamic_spec.get("eval_data_yaml")
            self.cfg = FedYoloConfig(
                global_classes=[_placeholder(i) for i in range(self.max_classes)],
                model=model, federation=federation, nodes=[],
                output_dir=dynamic_spec.get("output_dir", "runs/net_dynamic"),
                seed=dynamic_spec.get("seed", 0),
            )
            self.raw_config = None
            self._next_node_idx = 0
        else:
            self.config_path = config_path
            # Kept verbatim so it can be served byte-identical to clients
            # via /api/rooms/{id}/config -- they build their own
            # FedYoloConfig from this same dict via
            # config.load_config_from_dict(), overriding only their own
            # node's data_yaml.
            self.raw_config = yaml.safe_load(Path(config_path).read_text())
            self.cfg = load_config(config_path)
            self.max_classes = self.cfg.nc

        self.mode = self.cfg.federation.mode
        self.nodes: dict[str, NodeStatus] = {n.name: NodeStatus(name=n.name, owned_classes=list(n.owned_classes)) for n in self.cfg.nodes}
        self.started = False
        self.finished = False
        self.deletable = True  # False once started, until finished

        self._lock = threading.Lock()
        self.events: list[RoomEvent] = []
        self._subscribers: list[asyncio.Queue] = []

        self.model = build_model(
            self.cfg.model.arch, self.cfg.nc, self.cfg.model.imgsz,
            pretrained=self.cfg.model.pretrained, class_names=self.cfg.global_classes,
        )
        self.out_dir = Path(self.cfg.output_dir) / f"net_room_{room_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.eval_history: list[dict] = []
        self.checkpoint_history: list[CheckpointEntry] = []
        self.serving_checkpoint: Path | None = None  # None = serve live state

        self._async_server: AsyncFedServer | None = None
        self.global_state_dict: dict | None = None
        self.round_idx = 0
        self.total_rounds: int | None = None

        if self.mode == "async":
            self._async_server = AsyncFedServer(self.cfg)
            self._pull_cache: dict[int, dict] = {}
            self._pull_cache_order: list[int] = []
        else:
            self.global_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.total_rounds = self.cfg.federation.rounds
            self._pending: dict[str, NodeRoundResult] = {}

    # ── events / pub-sub for the dashboard's WebSocket ──────────────────────
    def _emit(self, kind: str, **data) -> None:
        ev = RoomEvent(ts=time.time(), kind=kind, data=data)
        with self._lock:
            self.events.append(ev)
            if len(self.events) > 500:
                self.events = self.events[-500:]
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(ev)
            except Exception:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def check_password(self, password: str) -> bool:
        return secrets.compare_digest(password or "", self.password)

    # ── node lifecycle ───────────────────────────────────────────────────────
    def _first_free_static_node(self) -> NodeConfig:
        for n in self.cfg.nodes:
            if self.nodes[n.name].status == "unclaimed":
                return n
        raise RuntimeError("every node in this room is already claimed")

    def join(self, ip: str, class_names: list[str] | None = None) -> dict:
        """Auto-assigns the first free node slot. For dynamic rooms,
        class_names is this node's own declared classes -- new names get
        the next free global id (order of first appearance across all
        node registrations); names already claimed by an earlier node
        reuse that id. Returns {node_name, owned_classes, owned_global_ids}."""
        with self._lock:
            if self.finished:
                raise PermissionError("room has finished -- registration is closed")
            if self.dynamic:
                if self.started and self.mode == "sync":
                    raise PermissionError("sync room already started -- registration is closed")
                class_names = class_names or []
                owned_ids = []
                for name in class_names:
                    if name in self.cfg.global_classes and not _is_placeholder(name):
                        owned_ids.append(self.cfg.global_classes.index(name))
                        continue
                    try:
                        slot = next(i for i, n in enumerate(self.cfg.global_classes) if _is_placeholder(n))
                    except StopIteration:
                        raise ValueError(
                            f"class ceiling reached (max_classes={self.max_classes}) -- "
                            f"cannot add new class '{name}'"
                        )
                    self.cfg.global_classes[slot] = name
                    owned_ids.append(slot)

                node_name = f"node_{self._next_node_idx}"
                self._next_node_idx += 1
                node_cfg = NodeConfig(name=node_name, data_yaml="", owned_classes=list(class_names))
                self.cfg.nodes.append(node_cfg)
                _validate(self.cfg, partial=True)
                self.nodes[node_name] = NodeStatus(name=node_name, owned_classes=list(class_names))
                if self.mode == "async" and self._async_server is not None:
                    # AsyncFedServer computed _total_submissions =
                    # len(cfg.nodes) * async_node_cycles at construction
                    # time, when a dynamic room legitimately has ZERO
                    # nodes yet -- left at 0, that made the room report
                    # itself "finished" after a single real submission
                    # (found via real end-to-end testing). Kept current
                    # here as an informational "as of the last join"
                    # figure; Room.async_push below never actually uses
                    # it to auto-finish a dynamic room (see the comment
                    # there for why an always-open room can't have a
                    # fixed target in the first place).
                    self._async_server._total_submissions = len(self.cfg.nodes) * self.cfg.federation.async_node_cycles
            else:
                node_cfg = self._first_free_static_node()
                node_name = node_cfg.name
                owned_ids = node_cfg.owned_global_ids(self.cfg.global_classes)

            st = self.nodes[node_name]
            st.ip = ip
            st.status = "idle"
            st.last_seen = time.time()

        self._emit("node_joined", node=node_name, ip=ip, owned_classes=class_names or node_cfg.owned_classes)
        return {
            "node_name": node_name,
            "owned_classes": node_cfg.owned_classes,
            "owned_global_ids": owned_ids,
            "server_data_yaml_hint": node_cfg.data_yaml,
        }

    def add_classes(self, node_name: str, new_class_names: list[str]) -> dict:
        """Async, dynamic rooms only: an already-joined node declares
        additional classes it wants to start contributing to, going
        forward. New classes start from their (until now untouched)
        random init -- there's no way around that; a class only ever
        learns from the rounds/cycles it's actually been trained in."""
        if not self.dynamic or self.mode != "async":
            raise PermissionError("adding classes mid-run is only supported for dynamic async rooms")
        with self._lock:
            if node_name not in self.nodes:
                raise KeyError(f"no such node '{node_name}'")
            node_cfg = next(n for n in self.cfg.nodes if n.name == node_name)
            added_ids = []
            for name in new_class_names:
                if name in node_cfg.owned_classes:
                    continue
                if name in self.cfg.global_classes and not _is_placeholder(name):
                    gid = self.cfg.global_classes.index(name)
                else:
                    try:
                        gid = next(i for i, n in enumerate(self.cfg.global_classes) if _is_placeholder(n))
                    except StopIteration:
                        raise ValueError(f"class ceiling reached (max_classes={self.max_classes})")
                    self.cfg.global_classes[gid] = name
                node_cfg.owned_classes.append(name)
                added_ids.append(gid)
            self.nodes[node_name].owned_classes = list(node_cfg.owned_classes)
        self._emit("classes_added", node=node_name, added=new_class_names)
        return {"node_name": node_name, "owned_classes": node_cfg.owned_classes, "added_global_ids": added_ids}

    def set_status(self, node_name: str, status: str, **extra) -> None:
        st = self.nodes.get(node_name)
        if st is None:
            return
        st.status = status
        st.last_seen = time.time()
        for k, v in extra.items():
            setattr(st, k, v)
        self._emit("status_changed", node=node_name, status=status, **extra)

    def start(self) -> None:
        with self._lock:
            self.started = True
            self.deletable = False
        self._emit("room_started")

    def finish(self) -> None:
        """Operator action to explicitly close an always-open dynamic
        async room (see async_push's comment on why those never
        auto-finish) -- makes it deletable and stops new joins the same
        way a naturally-finished room already does."""
        with self._lock:
            self.finished = True
            self.deletable = True
        self._emit("room_finished")

    # ── checkpoint history ───────────────────────────────────────────────────
    def _save_checkpoint(self, state_dict: dict, marker: int, label: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{ts}_{self.room_id}_{label.replace(' ', '_')}.pt"
        torch.save({"state_dict": state_dict, "global_classes": list(self.cfg.global_classes)}, path)
        entry = CheckpointEntry(path=path, marker=marker, label=label, created=time.time())
        with self._lock:
            self.checkpoint_history.append(entry)
            while len(self.checkpoint_history) > _CHECKPOINT_HISTORY_SIZE:
                old = self.checkpoint_history.pop(0)
                old.path.unlink(missing_ok=True)

    def list_checkpoints(self) -> list[dict]:
        with self._lock:
            return [
                {"path": str(e.path), "marker": e.marker, "label": e.label, "created": e.created}
                for e in self.checkpoint_history
            ]

    def set_serving_checkpoint(self, path: str | None) -> None:
        """Non-destructive: pins what new /pull calls serve, without
        touching the live training state (rollback() below is the
        destructive one). path=None returns to serving the live state."""
        with self._lock:
            if path is None:
                self.serving_checkpoint = None
                return
            match = next((e for e in self.checkpoint_history if str(e.path) == path), None)
            if match is None:
                raise KeyError(f"no such checkpoint: {path}")
            self.serving_checkpoint = match.path
        self._emit("serving_checkpoint_changed", path=path)

    def rollback(self, path: str) -> None:
        """Destructive: resets the room's LIVE state to a past checkpoint
        and discards everything after it -- newer checkpoints deleted,
        round/version counters reset, eval history truncated to entries
        at or before that point."""
        with self._lock:
            match = next((e for e in self.checkpoint_history if str(e.path) == path), None)
            if match is None:
                raise KeyError(f"no such checkpoint: {path}")
            payload = torch.load(match.path, map_location="cpu", weights_only=False)
            sd = payload["state_dict"]

            newer = [e for e in self.checkpoint_history if e.marker > match.marker]
            for e in newer:
                e.path.unlink(missing_ok=True)
            self.checkpoint_history = [e for e in self.checkpoint_history if e.marker <= match.marker]
            self.eval_history = [e for e in self.eval_history if e.get("round_or_version", 0) <= match.marker]
            self.serving_checkpoint = None
            self.finished = False
            for st in self.nodes.values():
                st.round_or_cycle = min(st.round_or_cycle, match.marker)
                if st.status == "done":
                    st.status = "waiting"

            if self.mode == "sync":
                self.global_state_dict = sd
                self.round_idx = match.marker
                self._pending.clear()
            else:
                self._async_server._global_sd = {k: v.clone() for k, v in sd.items()}
                self._async_server._version = match.marker
                self._async_server._submission_count = match.marker
                self._pull_cache.clear()
                self._pull_cache_order.clear()
        self._emit("room_rolled_back", to_marker=match.marker, label=match.label)

    # ── sync protocol ────────────────────────────────────────────────────────
    def sync_pull(self, node_name: str) -> tuple[dict, int]:
        with self._lock:
            if self.serving_checkpoint is not None:
                payload = torch.load(self.serving_checkpoint, map_location="cpu", weights_only=False)
                return payload["state_dict"], self.round_idx
            sd = {k: v.clone() for k, v in self.global_state_dict.items()}
            r = self.round_idx
        return sd, r

    def sync_push(self, node_name: str, result: NodeRoundResult, pulled_round: int) -> dict:
        """Buffer this node's result; aggregate once every configured node
        has reported in for the current round."""
        with self._lock:
            if pulled_round != self.round_idx:
                return {"round_advanced": False, "round": self.round_idx, "stale": True}

            self._pending[node_name] = result
            self.nodes[node_name].round_or_cycle = self.round_idx + 1
            if set(self._pending) < set(self.nodes):
                return {"round_advanced": False, "round": self.round_idx}

            results = [self._pending[n.name] for n in self.cfg.nodes]
            self.global_state_dict = aggregate(
                node_state_dicts=[r.state_dict for r in results],
                node_num_images=[r.num_images for r in results],
                node_owned_global_ids=[r.owned_global_ids for r in results],
                node_class_counts=[r.class_counts for r in results],
                model=self.model,
                nc=self.cfg.nc,
            )
            self._pending.clear()
            self.round_idx += 1
            round_now = self.round_idx
            state_snapshot = {k: v.clone() for k, v in self.global_state_dict.items()}

            finished = round_now >= self.total_rounds
            due = finished or (round_now % self.cfg.federation.eval_interval == 0)

        self._save_checkpoint(state_snapshot, round_now, f"round{round_now:03d}")
        self._emit("round_advanced", round=round_now, total=self.total_rounds)

        if due:
            res = self._maybe_eval(state_snapshot, f"round {round_now:03d}", round_now)
            if res:
                self._emit("eval_result", **{k: v for k, v in res.items() if k != "per_class"})

        if finished:
            with self._lock:
                self.finished = True
                self.deletable = True
                for st in self.nodes.values():
                    st.status = "done"
            self._emit("room_finished")

        return {"round_advanced": True, "round": round_now, "finished": finished}

    # ── async protocol (thin wrapper over AsyncFedServer's existing pull/push) ─
    def async_pull(self, node_name: str) -> tuple[dict, int]:
        with self._lock:
            if self.serving_checkpoint is not None:
                payload = torch.load(self.serving_checkpoint, map_location="cpu", weights_only=False)
                return payload["state_dict"], self._async_server._version
        sd, version = self._async_server.pull()
        with self._lock:
            self._pull_cache[version] = {k: v.clone() for k, v in sd.items()}
            self._pull_cache_order.append(version)
            while len(self._pull_cache_order) > _PULL_CACHE_SIZE:
                old = self._pull_cache_order.pop(0)
                self._pull_cache.pop(old, None)
        return sd, version

    def async_push(self, node_name: str, result: NodeRoundResult, pulled_version: int) -> dict:
        with self._lock:
            pulled_sd = self._pull_cache.get(pulled_version)
        if pulled_sd is None:
            pulled_sd, _ = self._async_server.pull()
            log.warning(
                "node=%s pulled_version=%s not in cache -- staleness "
                "approximated against the current global state", node_name, pulled_version,
            )

        self._async_server.push(result, pulled_sd, pulled_version)

        with self._lock:
            st = self.nodes[node_name]
            st.round_or_cycle += 1
            node_done = st.round_or_cycle >= self.cfg.federation.async_node_cycles
            version = self._async_server._version
            # Dynamic rooms never auto-finish: a new node can join and
            # start contributing at any point, so there's no fixed
            # "total" the way a static room's node list guarantees --
            # _total_submissions here is only ever an as-of-last-join
            # estimate (see join() above), not a real target. Finishing
            # a dynamic room is an explicit operator action (Room.finish()
            # / POST /api/rooms/{id}/finish). Static rooms keep the
            # original auto-finish behavior.
            finished = (not self.dynamic) and (
                self._async_server._submission_count >= self._async_server._total_submissions
            )
            state_snapshot = {k: v.clone() for k, v in self._async_server._global_sd.items()}
            new_eval_raw = None
            if self._async_server._map_history:
                latest = self._async_server._map_history[-1]
                if not self.eval_history or self.eval_history[-1].get("label") != latest.get("label"):
                    new_eval_raw = latest
            if finished:
                self.finished = True
                self.deletable = True

        if new_eval_raw is not None:
            new_eval = {**new_eval_raw, "round_or_version": version}
            with self._lock:
                self.eval_history.append(new_eval)
            self._emit("eval_result", **{k: v for k, v in new_eval.items() if k != "per_class"})
        self._save_checkpoint(state_snapshot, version, f"v{version:04d}_{node_name}")
        self._emit("async_submitted", node=node_name, version=version, cycle=self.nodes[node_name].round_or_cycle)
        if finished:
            self._emit("room_finished")

        return {"version": version, "node_done": node_done, "finished": finished}

    def _maybe_eval(self, state_dict: dict, label: str, marker: int) -> dict | None:
        """Server-side eval is optional -- dynamic rooms usually have no
        server-local access to any node's data (that's the whole point:
        data stays on each client's own machine), so this is skipped
        cleanly rather than assumed available. Static rooms keep today's
        behavior (pools each node's own val split, which the server DOES
        have local access to in every real run in this project so far)."""
        if self.dynamic and not self.eval_data_yaml:
            return None
        if self.dynamic:
            res = self._eval_against_single_yaml(state_dict, label)
        else:
            res = _run_eval(self.cfg, state_dict, label)
        if res:
            with self._lock:
                res = {**res, "round_or_version": marker}
                self.eval_history.append(res)
        return res

    def _eval_against_single_yaml(self, state_dict: dict, label: str) -> dict | None:
        """Dynamic-room eval: no per-node local data to pool (see
        _maybe_eval), so this validates directly against one operator-
        supplied merged/pooled data yaml instead of
        evaluate.py's materialize_pooled_dataset(cfg, ...) path."""
        try:
            from ultralytics import YOLO
            from .. import evaluate as _eval_mod
            model = build_model(self.cfg.model.arch, self.cfg.nc, self.cfg.model.imgsz,
                                 pretrained=self.cfg.model.pretrained, class_names=self.cfg.global_classes)
            model.load_state_dict(state_dict)
            model.names = {i: n for i, n in enumerate(self.cfg.global_classes)}
            yolo = YOLO("yolov8n.yaml")
            yolo.model = model
            metrics = yolo.val(data=self.eval_data_yaml, imgsz=self.cfg.model.imgsz,
                                device=self.cfg.federation.device, verbose=False, plots=False)
            return {
                "label": label,
                "map50": float(metrics.box.map50),
                "map5095": float(metrics.box.map),
                "per_class": _eval_mod.per_class_map50(metrics),
            }
        except Exception as exc:
            log.warning(f"dynamic-room evaluation failed ({label}): {exc}")
            return None

    # ── snapshot for GET /api/rooms/{id} and the dashboard's initial render ──
    def snapshot(self) -> dict:
        with self._lock:
            if self.mode == "sync":
                round_now, total = self.round_idx, self.total_rounds
            else:
                round_now = self._async_server._version if self._async_server else 0
                total = self._async_server._total_submissions if self._async_server else None
            return {
                "room_id": self.room_id,
                "dynamic": self.dynamic,
                "mode": self.mode,
                "max_classes": self.max_classes,
                "started": self.started,
                "finished": self.finished,
                "deletable": self.deletable,
                "round": round_now,
                "total": total,
                "serving_checkpoint": str(self.serving_checkpoint) if self.serving_checkpoint else None,
                "nodes": {
                    name: {
                        "name": s.name, "ip": s.ip, "status": s.status,
                        "last_seen": s.last_seen, "last_metrics": s.last_metrics,
                        "round_or_cycle": s.round_or_cycle, "detail": s.detail,
                        "owned_classes": s.owned_classes,
                    }
                    for name, s in self.nodes.items()
                },
                "claimed_classes": [c for c in self.cfg.global_classes if not _is_placeholder(c)],
                "eval_history": [
                    {k: v for k, v in e.items() if k != "per_class"} for e in self.eval_history
                ],
            }


class RoomRegistry:
    def __init__(self):
        self._rooms: dict[str, Room] = {}
        self._lock = threading.Lock()

    def create(self, room_id: str, config_path: str | None = None, *, dynamic_spec: dict | None = None) -> Room:
        with self._lock:
            if room_id in self._rooms:
                raise ValueError(f"room '{room_id}' already exists")
            room = Room(room_id, config_path, dynamic_spec=dynamic_spec)
            self._rooms[room_id] = room
        return room

    def get(self, room_id: str) -> Room:
        with self._lock:
            room = self._rooms.get(room_id)
        if room is None:
            raise KeyError(room_id)
        return room

    def delete(self, room_id: str) -> None:
        with self._lock:
            room = self._rooms.get(room_id)
            if room is None:
                raise KeyError(room_id)
            if not room.deletable:
                raise PermissionError("room is actively running -- stop it before deleting")
            del self._rooms[room_id]

    def list(self) -> list[dict]:
        with self._lock:
            rooms = list(self._rooms.values())
        return [
            {
                "room_id": r.room_id, "mode": r.mode, "dynamic": r.dynamic,
                "node_count": len(r.nodes), "started": r.started, "finished": r.finished,
            }
            for r in rooms
        ]
