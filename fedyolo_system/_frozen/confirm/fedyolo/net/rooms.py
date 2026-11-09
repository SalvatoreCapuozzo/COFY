"""Room: one FedYoloConfig, live and network-orchestrated.

Two room types:

- **Static** -- created from an existing fedyolo config YAML, exactly like
  fedyolo.simulate's in-process runs: nodes/global_classes fixed and
  server-authoritative. federation.mode decides sync vs async.
- **Dynamic** -- created from scratch with just a `max_classes` ceiling,
  model arch/imgsz, and federation hyperparameters. Clients declare their
  own classes on join; global class ids are assigned in order of first
  appearance. Unclaimed slots keep their initial value (per-class
  aggregation only averages over a class's owners).

Sync rooms run their own round barrier, then call the SAME aggregate()
FedServer.run() uses, with every merge option of the config passed through
(merge_rule, normalize_steps, weight_basis, fisher, ...). Async rooms wrap
one real AsyncFedServer and forward pull()/push() to it.

Robustness:
- Every node has a secret token (issued at join, stored hashed) required
  with the room password for pull/push/heartbeat.
- Nodes send heartbeats. A sync round whose missing nodes have all gone
  silent for `heartbeat_timeout` seconds (or that has waited longer than
  `sync_round_timeout`, if set) is aggregated with the nodes that reported;
  classes owned only by absent nodes keep their previous weights. A static
  async room finishes once every node is done or gone.
- Pushes carry a push_id: a retried push returns the recorded outcome
  instead of being aggregated twice.
- Room state, node tokens and pending sync results are persisted under the
  server's state dir; RoomRegistry.load_all() restores them after a server
  restart (the async CA2FL cache and pull cache are not persisted).

Evaluation happens on the server only in developer mode (the server never
holds labeled data in real use): against a path given at room creation,
else the config's eval_data_yaml, else the nodes' own validation splits
when the server can reach them (testing on one machine). It runs in a
background thread so pushes never wait on it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import secrets
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ..client import NodeRoundResult
from ..config import FederationConfig, FedYoloConfig, ModelConfig, NodeConfig, _validate, load_config
from ..model import aggregate, build_model, per_class_param_names
from ..server import AsyncFedServer, _run_eval
from . import protocol

log = logging.getLogger("fedyolo.net.rooms")

_PULL_CACHE_SIZE = 30
# Bounded: this project has hit real "No space left on device" crashes from
# unbounded checkpoint retention before.
_CHECKPOINT_HISTORY_SIZE = 10
_UNCLAIMED_PREFIX = "__unclaimed_"


def _placeholder(i: int) -> str:
    return f"{_UNCLAIMED_PREFIX}{i}__"


def _is_placeholder(name: str) -> bool:
    return name.startswith(_UNCLAIMED_PREFIX)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclasses.dataclass
class RoomOptions:
    state_dir: Path = Path("runs/net_server_state")
    developer_mode: bool = False
    heartbeat_timeout: float = 180.0     # a node silent this long counts as gone
    sync_round_timeout: float = 0.0      # >0: hard cap on waiting for a sync round


@dataclasses.dataclass
class NodeStatus:
    name: str
    ip: str | None = None
    status: str = "unclaimed"  # unclaimed | idle | downloading | training | uploading | waiting | done | lost
    last_seen: float = 0.0
    last_metrics: dict | None = None
    round_or_cycle: int = 0
    detail: str = ""
    owned_classes: list = dataclasses.field(default_factory=list)
    token_hash: str | None = None
    last_push_id: str | None = None
    last_push_outcome: dict | None = None


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
    def __init__(self, room_id: str, config_path: str | None = None, *,
                 dynamic_spec: dict | None = None, eval_data_yaml: str | None = None,
                 options: RoomOptions | None = None):
        """Exactly one of config_path (static) / dynamic_spec (dynamic).
        eval_data_yaml is only honoured in developer mode."""
        self.room_id = room_id
        self.options = options or RoomOptions()
        self.password = secrets.token_urlsafe(9)
        self.dynamic = dynamic_spec is not None
        self.dynamic_spec = dynamic_spec
        self.eval_data_yaml: str | None = eval_data_yaml if self.options.developer_mode else None

        if self.dynamic:
            self.config_path = None
            self.max_classes = int(dynamic_spec["max_classes"])
            model = ModelConfig(**dynamic_spec.get("model", {}))
            fed_raw = dict(dynamic_spec.get("federation", {}))
            fed_raw.setdefault("mode", "sync")
            federation = FederationConfig(**fed_raw)
            from ..config import _resolve_device
            federation.device = _resolve_device(federation.device)
            if self.options.developer_mode and dynamic_spec.get("eval_data_yaml"):
                self.eval_data_yaml = dynamic_spec["eval_data_yaml"]
            self.cfg = FedYoloConfig(
                global_classes=[_placeholder(i) for i in range(self.max_classes)],
                model=model, federation=federation, nodes=[],
                output_dir=dynamic_spec.get("output_dir", "runs/net_dynamic"),
                seed=dynamic_spec.get("seed", 0),
            )
            self.raw_config = None
            self._next_node_idx = 0
        else:
            self.config_path = str(config_path)
            # served verbatim to clients (they override only their own data_yaml)
            self.raw_config = yaml.safe_load(Path(config_path).read_text())
            self.cfg = load_config(config_path)
            self.max_classes = self.cfg.nc

        self.mode = self.cfg.federation.mode
        self.nodes: dict[str, NodeStatus] = {
            n.name: NodeStatus(name=n.name, owned_classes=list(n.owned_classes)) for n in self.cfg.nodes}
        self.started = False
        self.finished = False
        self.deletable = True

        self._lock = threading.RLock()
        self.events: list[RoomEvent] = []
        self._subscribers: list[asyncio.Queue] = []
        self._eval_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"eval-{room_id}")

        self.model = build_model(self.cfg.model.arch, self.cfg.nc, self.cfg.model.imgsz,
                                 pretrained=self.cfg.model.pretrained, class_names=self.cfg.global_classes)
        if self.cfg.federation.etf_head:
            from ..class_geometry import apply_etf_head
            apply_etf_head(self.model, seed=self.cfg.seed)
        self._reference_sd = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.out_dir = Path(self.cfg.output_dir) / f"net_room_{room_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path(self.options.state_dir) / "rooms" / room_id
        (self.state_dir / "pending").mkdir(parents=True, exist_ok=True)
        self.eval_history: list[dict] = []
        self.checkpoint_history: list[CheckpointEntry] = []
        self.serving_checkpoint: Path | None = None

        self._async_server: AsyncFedServer | None = None
        self.global_state_dict: dict | None = None
        self.round_idx = 0
        self.total_rounds: int | None = None
        self._pending: dict[str, NodeRoundResult] = {}
        self._pending_since: float | None = None

        if self.mode == "async":
            self._async_server = AsyncFedServer(self.cfg)
            self._async_server.eval_enabled = False
            self._pull_cache: dict[int, dict] = {}
            self._pull_cache_order: list[int] = []
        else:
            self.global_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.total_rounds = self.cfg.federation.rounds

    # ── events / pub-sub for the dashboard's WebSocket ──────────────────────
    def _emit(self, kind: str, **data) -> None:
        ev = RoomEvent(ts=time.time(), kind=kind, data=data)
        with self._lock:
            self.events.append(ev)
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

    # ── auth ─────────────────────────────────────────────────────────────────
    def check_password(self, password: str) -> bool:
        return secrets.compare_digest(password or "", self.password)

    def check_node(self, node_name: str, token: str | None) -> bool:
        st = self.nodes.get(node_name)
        return bool(st and st.token_hash and token and secrets.compare_digest(_hash(token), st.token_hash))

    def refresh_password(self) -> str:
        with self._lock:
            self.password = secrets.token_urlsafe(9)
        self.save_state()
        return self.password

    def _alive(self, st: NodeStatus, now: float) -> bool:
        return now - st.last_seen <= self.options.heartbeat_timeout

    # ── node lifecycle ───────────────────────────────────────────────────────
    def join(self, ip: str, class_names: list[str] | None = None, reclaim_node: str | None = None) -> dict:
        """Assigns the first free node slot, or reclaims `reclaim_node` if
        that node has gone silent (a client that restarted). Returns
        {node_name, node_token, owned_classes, owned_global_ids, ...}."""
        now = time.time()
        with self._lock:
            if self.finished:
                raise PermissionError("room has finished -- registration is closed")
            if reclaim_node:
                st = self.nodes.get(reclaim_node)
                if st is None:
                    raise ValueError(f"no such node '{reclaim_node}' in this room")
                if st.status != "unclaimed" and self._alive(st, now) and st.status not in ("lost",):
                    raise PermissionError(
                        f"node '{reclaim_node}' is still active (last seen {now - st.last_seen:.0f}s ago); "
                        f"it can be reclaimed after {self.options.heartbeat_timeout:.0f}s of silence")
                node_cfg = next(n for n in self.cfg.nodes if n.name == reclaim_node)
                node_name = reclaim_node
                owned_ids = node_cfg.owned_global_ids(self.cfg.global_classes)
            elif self.dynamic:
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
                        raise ValueError(f"class ceiling reached (max_classes={self.max_classes}) -- "
                                         f"cannot add new class '{name}'")
                    self.cfg.global_classes[slot] = name
                    owned_ids.append(slot)
                node_name = f"node_{self._next_node_idx}"
                self._next_node_idx += 1
                node_cfg = NodeConfig(name=node_name, data_yaml="", owned_classes=list(class_names))
                self.cfg.nodes.append(node_cfg)
                _validate(self.cfg, partial=True)
                self.nodes[node_name] = NodeStatus(name=node_name, owned_classes=list(class_names))
                if self.mode == "async" and self._async_server is not None:
                    # informational only: dynamic async rooms never auto-finish
                    self._async_server._total_submissions = len(self.cfg.nodes) * self.cfg.federation.async_node_cycles
            else:
                node_cfg = next((n for n in self.cfg.nodes if self.nodes[n.name].status == "unclaimed"), None)
                if node_cfg is None:
                    raise PermissionError("every node in this room is already claimed "
                                          "(a restarted client can reclaim its node by name)")
                node_name = node_cfg.name
                owned_ids = node_cfg.owned_global_ids(self.cfg.global_classes)

            token = secrets.token_urlsafe(24)
            st = self.nodes[node_name]
            st.ip, st.last_seen, st.token_hash = ip, now, _hash(token)
            if st.status in ("unclaimed", "lost"):
                st.status = "idle"
        self.save_state()
        self._emit("node_joined", node=node_name, ip=ip, owned_classes=node_cfg.owned_classes,
                   reclaimed=bool(reclaim_node))
        return {"node_name": node_name, "node_token": token, "owned_classes": node_cfg.owned_classes,
                "owned_global_ids": owned_ids, "server_data_yaml_hint": node_cfg.data_yaml,
                "round_or_cycle": self.nodes[node_name].round_or_cycle}

    def add_classes(self, node_name: str, new_class_names: list[str]) -> dict:
        """Async dynamic rooms only: an already-joined node declares more
        classes (they start from their untouched initial weights)."""
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
        self.save_state()
        self._emit("classes_added", node=node_name, added=new_class_names)
        return {"node_name": node_name, "owned_classes": node_cfg.owned_classes, "added_global_ids": added_ids}

    def heartbeat(self, node_name: str) -> dict:
        with self._lock:
            st = self.nodes[node_name]
            st.last_seen = time.time()
            if st.status == "lost":
                st.status = "idle"
                self._emit("status_changed", node=node_name, status="idle", detail="back online")
            return {"round_or_version": self.round_idx if self.mode == "sync" else self._async_server._version,
                    "finished": self.finished}

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
        self.save_state()
        self._emit("room_started")

    def finish(self) -> None:
        """Operator action (dynamic async rooms never auto-finish)."""
        with self._lock:
            self.finished = True
            self.deletable = True
        self.save_state()
        self._emit("room_finished")

    def owned_ids_of(self, node_name: str) -> list[int]:
        node_cfg = next(n for n in self.cfg.nodes if n.name == node_name)
        return node_cfg.owned_global_ids(self.cfg.global_classes)

    # ── checkpoint history ───────────────────────────────────────────────────
    def _save_checkpoint(self, state_dict: dict, marker: int, label: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{ts}_{self.room_id}_{label.replace(' ', '_')}.pt"
        torch.save({"state_dict": state_dict, "global_classes": list(self.cfg.global_classes)}, path)
        entry = CheckpointEntry(path=path, marker=marker, label=label, created=time.time())
        with self._lock:
            self.checkpoint_history.append(entry)
            while len(self.checkpoint_history) > _CHECKPOINT_HISTORY_SIZE:
                self.checkpoint_history.pop(0).path.unlink(missing_ok=True)

    def list_checkpoints(self) -> list[dict]:
        with self._lock:
            return [{"path": str(e.path), "marker": e.marker, "label": e.label, "created": e.created}
                    for e in self.checkpoint_history]

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]

    def set_serving_checkpoint(self, path: str | None) -> None:
        """Non-destructive: pins what new /pull calls serve (None = live state)."""
        with self._lock:
            if path is None:
                self.serving_checkpoint = None
            else:
                match = next((e for e in self.checkpoint_history if str(e.path) == path), None)
                if match is None:
                    raise KeyError(f"no such checkpoint: {path}")
                self.serving_checkpoint = match.path
        self.save_state()
        self._emit("serving_checkpoint_changed", path=path)

    def rollback(self, path: str) -> None:
        """Destructive: resets the live state to a past checkpoint and
        discards everything newer."""
        with self._lock:
            match = next((e for e in self.checkpoint_history if str(e.path) == path), None)
            if match is None:
                raise KeyError(f"no such checkpoint: {path}")
            sd = self._load_checkpoint(match.path)
            for e in [e for e in self.checkpoint_history if e.marker > match.marker]:
                e.path.unlink(missing_ok=True)
            self.checkpoint_history = [e for e in self.checkpoint_history if e.marker <= match.marker]
            self.eval_history = [e for e in self.eval_history if e.get("round_or_version", 0) <= match.marker]
            self.serving_checkpoint = None
            self.finished = False
            for st in self.nodes.values():
                st.round_or_cycle = min(st.round_or_cycle, match.marker)
                st.last_push_id = st.last_push_outcome = None
                if st.status == "done":
                    st.status = "waiting"
            if self.mode == "sync":
                self.global_state_dict = sd
                self.round_idx = match.marker
                self._clear_pending()
            else:
                self._async_server._global_sd = {k: v.clone() for k, v in sd.items()}
                self._async_server._version = match.marker
                self._async_server._submission_count = match.marker
                self._pull_cache.clear()
                self._pull_cache_order.clear()
        self.save_state()
        self._emit("room_rolled_back", to_marker=match.marker, label=match.label)

    # ── sync protocol ────────────────────────────────────────────────────────
    def sync_pull(self, node_name: str) -> tuple[dict, int]:
        with self._lock:
            if self.serving_checkpoint is not None:
                return self._load_checkpoint(self.serving_checkpoint), self.round_idx
            return {k: v.clone() for k, v in self.global_state_dict.items()}, self.round_idx

    def _clear_pending(self) -> None:
        self._pending.clear()
        self._pending_since = None
        for f in (self.state_dir / "pending").glob("*.pt"):
            f.unlink(missing_ok=True)

    def sync_push(self, node_name: str, result: NodeRoundResult, pulled_round: int, raw: bytes | None = None) -> dict:
        """Buffer this node's result; aggregate once every configured node
        has reported (or the absent ones are gone -- see tick())."""
        with self._lock:
            if pulled_round != self.round_idx:
                return {"round_advanced": False, "round": self.round_idx, "stale": True}
            self._pending[node_name] = result
            if self._pending_since is None:
                self._pending_since = time.time()
            if raw is not None:   # persisted so a server restart doesn't lose this round
                (self.state_dir / "pending" / f"{node_name}.pt").write_bytes(raw)
            self.nodes[node_name].round_or_cycle = self.round_idx + 1
            if set(self._pending) < set(self.nodes):
                self.save_state()
                return {"round_advanced": False, "round": self.round_idx}
        return self._aggregate_round(reason="all nodes reported")

    def _aggregate_round(self, reason: str) -> dict:
        with self._lock:
            present = [n for n in self.cfg.nodes if n.name in self._pending]
            if not present:
                return {"round_advanced": False, "round": self.round_idx}
            absent = [n.name for n in self.cfg.nodes if n.name not in self._pending]
            results = [self._pending[n.name] for n in present]
            fed = self.cfg.federation
            prev = {k: v.clone() for k, v in self.global_state_dict.items()}
            new_sd = aggregate(
                node_state_dicts=[r.state_dict for r in results],
                node_num_images=[r.num_images for r in results],
                node_owned_global_ids=[r.owned_global_ids for r in results],
                node_class_counts=[r.class_counts for r in results],
                model=self.model, nc=self.cfg.nc, prev_global=prev,
                rule=fed.merge_rule, trim=fed.merge_trim,
                node_steps=([r.num_steps for r in results] if fed.normalize_steps else None),
                weight_basis=fed.weight_basis,
                node_instance_totals=[sum(r.class_counts.values()) for r in results],
                node_fishers=([r.fisher for r in results] if fed.merge_rule == "fisher" else None),
                fisher_damping=fed.fisher_damping,
            )
            if absent:
                # classes owned only by absent nodes keep last round's weights
                covered = {c for r in results for c in r.owned_global_ids}
                orphan = [c for c in range(self.cfg.nc) if c not in covered]
                for wk, bk in per_class_param_names(self.model):
                    for c in orphan:
                        new_sd[wk][c] = prev[wk][c]
                        new_sd[bk][c] = prev[bk][c]
                for name in absent:
                    if self.nodes[name].status != "unclaimed":
                        self.nodes[name].status = "lost"
            self.global_state_dict = new_sd
            self._clear_pending()
            self.round_idx += 1
            round_now = self.round_idx
            snapshot = {k: v.clone() for k, v in new_sd.items()}
            finished = round_now >= self.total_rounds
            due = finished or (round_now % fed.eval_interval == 0)
            if finished:
                self.finished = True
                self.deletable = True
                for st in self.nodes.values():
                    if st.status not in ("lost", "unclaimed"):
                        st.status = "done"
        self._save_checkpoint(snapshot, round_now, f"round{round_now:03d}")
        self.save_state()
        self._emit("round_advanced", round=round_now, total=self.total_rounds,
                   absent=absent, reason=reason)
        if absent:
            log.warning("room %s: round %d aggregated without %s (%s)", self.room_id, round_now, absent, reason)
        if due:
            self._schedule_eval(snapshot, f"round {round_now:03d}", round_now)
        if finished:
            self._emit("room_finished")
        return {"round_advanced": True, "round": round_now, "finished": finished}

    # ── async protocol (thin wrapper over AsyncFedServer's pull/push) ────────
    def async_pull(self, node_name: str) -> tuple[dict, int]:
        with self._lock:
            if self.serving_checkpoint is not None:
                return self._load_checkpoint(self.serving_checkpoint), self._async_server._version
        sd, version = self._async_server.pull()
        with self._lock:
            self._pull_cache[version] = {k: v.clone() for k, v in sd.items()}
            self._pull_cache_order.append(version)
            while len(self._pull_cache_order) > _PULL_CACHE_SIZE:
                self._pull_cache.pop(self._pull_cache_order.pop(0), None)
        return sd, version

    def async_push(self, node_name: str, result: NodeRoundResult, pulled_version: int, raw: bytes | None = None) -> dict:
        with self._lock:
            pulled_sd = self._pull_cache.get(pulled_version)
        if pulled_sd is None:
            pulled_sd, _ = self._async_server.pull()
            log.warning("node=%s pulled_version=%s not in cache -- staleness approximated "
                        "against the current global state", node_name, pulled_version)
        self._async_server.push(result, pulled_sd, pulled_version)
        with self._lock:
            st = self.nodes[node_name]
            st.round_or_cycle += 1
            node_done = st.round_or_cycle >= self.cfg.federation.async_node_cycles
            version = self._async_server._version
            sub = self._async_server._submission_count
            snapshot = {k: v.clone() for k, v in self._async_server._global_sd.items()}
            finished = (not self.dynamic) and sub >= self._async_server._total_submissions
            if finished:
                self.finished = True
                self.deletable = True
        self._save_checkpoint(snapshot, version, f"v{version:04d}_{node_name}")
        self.save_state()
        self._emit("async_submitted", node=node_name, version=version, cycle=st.round_or_cycle)
        if finished or sub % self.cfg.federation.eval_interval == 0:
            self._schedule_eval(snapshot, f"sub {sub:04d} v{version}", version)
        if finished:
            self._emit("room_finished")
        return {"version": version, "node_done": node_done, "finished": finished}

    # ── watchdog (called periodically by RoomRegistry) ───────────────────────
    def tick(self, now: float) -> None:
        if not self.started or self.finished:
            return
        with self._lock:
            for st in self.nodes.values():
                if st.status not in ("unclaimed", "done", "lost") and st.last_seen and not self._alive(st, now):
                    st.status = "lost"
                    self._emit("status_changed", node=st.name, status="lost",
                               detail=f"no heartbeat for {now - st.last_seen:.0f}s")
            if self.mode == "sync" and self._pending:
                missing = [st for name, st in self.nodes.items() if name not in self._pending]
                all_gone = all(not self._alive(st, now) for st in missing)
                timed_out = (self.options.sync_round_timeout > 0 and self._pending_since
                             and now - self._pending_since > self.options.sync_round_timeout)
                reason = ("absent nodes silent for > heartbeat timeout" if all_gone
                          else "sync round timeout" if timed_out else None)
            else:
                reason = None
                if self.mode == "async" and not self.dynamic:
                    cycles = self.cfg.federation.async_node_cycles
                    if all(st.round_or_cycle >= cycles or not self._alive(st, now) for st in self.nodes.values()) \
                            and any(st.round_or_cycle > 0 for st in self.nodes.values()):
                        self.finished = True
                        self.deletable = True
                        gone = [s.name for s in self.nodes.values() if s.round_or_cycle < cycles]
                        snapshot = {k: v.clone() for k, v in self._async_server._global_sd.items()}
                        version = self._async_server._version
                    else:
                        gone = None
        if reason:
            self._aggregate_round(reason)
        elif self.mode == "async" and not self.dynamic and self.finished:
            log.warning("room %s: async room finished without %s (no heartbeat)", self.room_id, gone)
            self.save_state()
            self._schedule_eval(snapshot, f"final v{version}", version)
            self._emit("room_finished", absent=gone)

    # ── evaluation: developer mode only, in a background thread ─────────────
    def _eval_source(self) -> str | None:
        if not self.options.developer_mode:
            return None
        if self.eval_data_yaml:
            return self.eval_data_yaml
        if not self.dynamic:
            return self.cfg.eval_data_yaml or "pooled-node-val"
        return None

    def _schedule_eval(self, state_dict: dict, label: str, marker: int) -> None:
        if self._eval_source() is None:
            return
        self._eval_pool.submit(self._run_eval_job, state_dict, label, marker)

    def _run_eval_job(self, state_dict: dict, label: str, marker: int) -> None:
        try:
            src = self._eval_source()
            if src == "pooled-node-val" or (not self.dynamic and src == self.cfg.eval_data_yaml):
                res = _run_eval(self.cfg, state_dict, label)
            else:
                res = self._eval_against_single_yaml(state_dict, label, src)
        except Exception as exc:
            log.warning("room %s: evaluation %s failed: %s", self.room_id, label, exc)
            return
        if res:
            res = {**res, "round_or_version": marker}
            with self._lock:
                self.eval_history.append(res)
            self.save_state()
            self._emit("eval_result", **{k: v for k, v in res.items() if k != "per_class"})

    def _eval_against_single_yaml(self, state_dict: dict, label: str, data_yaml: str) -> dict | None:
        from ultralytics import YOLO
        from .. import evaluate as _eval_mod
        model = build_model(self.cfg.model.arch, self.cfg.nc, self.cfg.model.imgsz,
                            pretrained=self.cfg.model.pretrained, class_names=self.cfg.global_classes)
        model.load_state_dict(state_dict)
        model.names = {i: n for i, n in enumerate(self.cfg.global_classes)}
        yolo = YOLO("yolov8n.yaml")
        yolo.model = model
        metrics = yolo.val(data=data_yaml, imgsz=self.cfg.model.imgsz, device=self.cfg.federation.device,
                           verbose=False, plots=False)
        return {"label": label, "map50": float(metrics.box.map50), "map5095": float(metrics.box.map),
                "per_class": _eval_mod.per_class_map50(metrics)}

    # ── persistence ──────────────────────────────────────────────────────────
    def save_state(self) -> None:
        with self._lock:
            state = {
                "version": 1, "room_id": self.room_id, "password": self.password,
                "dynamic": self.dynamic, "config_path": self.config_path, "dynamic_spec": self.dynamic_spec,
                "eval_data_yaml": self.eval_data_yaml,
                "global_classes": list(self.cfg.global_classes),
                "cfg_nodes": [dataclasses.asdict(n) for n in self.cfg.nodes],
                "next_node_idx": getattr(self, "_next_node_idx", 0),
                "started": self.started, "finished": self.finished, "deletable": self.deletable,
                "round_idx": self.round_idx, "pending_since": self._pending_since,
                "async_version": self._async_server._version if self._async_server else None,
                "async_submissions": self._async_server._submission_count if self._async_server else None,
                "serving_checkpoint": str(self.serving_checkpoint) if self.serving_checkpoint else None,
                "checkpoints": [{"path": str(e.path), "marker": e.marker, "label": e.label, "created": e.created}
                                for e in self.checkpoint_history],
                "eval_history": self.eval_history,
                "nodes": {n: {k: getattr(s, k) for k in ("name", "ip", "status", "round_or_cycle", "owned_classes",
                                                           "token_hash", "last_push_id", "last_push_outcome")}
                          for n, s in self.nodes.items()},
            }
            tmp = self.state_dir / "state.json.tmp"
            tmp.write_text(json.dumps(state, indent=1))
            tmp.chmod(0o600)
            tmp.replace(self.state_dir / "state.json")

    @classmethod
    def restore(cls, state_file: Path, options: RoomOptions) -> "Room":
        s = json.loads(state_file.read_text())
        room = cls(s["room_id"], s["config_path"], dynamic_spec=s["dynamic_spec"],
                   eval_data_yaml=s.get("eval_data_yaml"), options=options)
        with room._lock:
            room.password = s["password"]
            if room.dynamic:
                room.cfg.global_classes[:] = s["global_classes"]
                room.cfg.nodes[:] = [NodeConfig(**n) for n in s["cfg_nodes"]]
                room._next_node_idx = s["next_node_idx"]
                if room._async_server is not None:
                    room._async_server._total_submissions = len(room.cfg.nodes) * room.cfg.federation.async_node_cycles
            room.nodes = {}
            for name, ns in s["nodes"].items():
                st = NodeStatus(**ns)
                st.last_seen = time.time()   # grace period: give clients a heartbeat window to reconnect
                if st.status in ("downloading", "training", "uploading"):
                    st.status = "idle"
                room.nodes[name] = st
            room.started, room.finished, room.deletable = s["started"], s["finished"], s["deletable"]
            room.eval_history = s["eval_history"]
            room.checkpoint_history = [CheckpointEntry(Path(c["path"]), c["marker"], c["label"], c["created"])
                                       for c in s["checkpoints"] if Path(c["path"]).exists()]
            room.serving_checkpoint = Path(s["serving_checkpoint"]) if s.get("serving_checkpoint") else None
            latest = room.checkpoint_history[-1].path if room.checkpoint_history else None
            if room.mode == "sync":
                room.round_idx = s["round_idx"]
                if latest is not None:
                    room.global_state_dict = cls._load_checkpoint(latest)
                room._pending_since = s.get("pending_since")
                for f in (room.state_dir / "pending").glob("*.pt"):
                    name = f.stem
                    if name in room.nodes:
                        room._pending[name] = protocol.decode_update(
                            f.read_bytes(), node_name=name, reference_sd=room._reference_sd,
                            owned_global_ids=room.owned_ids_of(name), nc=room.cfg.nc)
            else:
                if latest is not None:
                    room._async_server._global_sd = cls._load_checkpoint(latest)
                room._async_server._version = s["async_version"] or 0
                room._async_server._submission_count = s["async_submissions"] or 0
        log.info("restored room %s (%s, %s, round/version %s)", room.room_id, room.mode,
                 "finished" if room.finished else "running" if room.started else "not started",
                 room.round_idx if room.mode == "sync" else room._async_server._version)
        return room

    def delete_state(self) -> None:
        shutil.rmtree(self.state_dir, ignore_errors=True)
        self._eval_pool.shutdown(wait=False)

    # ── snapshot for GET /api/rooms/{id} and the dashboard ───────────────────
    def snapshot(self) -> dict:
        with self._lock:
            if self.mode == "sync":
                round_now, total = self.round_idx, self.total_rounds
            else:
                round_now = self._async_server._version if self._async_server else 0
                total = self._async_server._total_submissions if self._async_server else None
            return {
                "room_id": self.room_id, "dynamic": self.dynamic, "mode": self.mode,
                "max_classes": self.max_classes, "started": self.started, "finished": self.finished,
                "deletable": self.deletable, "round": round_now, "total": total,
                "serving_checkpoint": str(self.serving_checkpoint) if self.serving_checkpoint else None,
                "server_eval": self._eval_source() is not None,
                "pending_nodes": sorted(self._pending) if self.mode == "sync" else [],
                "nodes": {
                    name: {"name": s.name, "ip": s.ip, "status": s.status, "last_seen": s.last_seen,
                           "last_metrics": s.last_metrics, "round_or_cycle": s.round_or_cycle,
                           "detail": s.detail, "owned_classes": s.owned_classes}
                    for name, s in self.nodes.items()
                },
                "claimed_classes": [c for c in self.cfg.global_classes if not _is_placeholder(c)],
                "eval_history": [{k: v for k, v in e.items() if k != "per_class"} for e in self.eval_history],
            }


class RoomRegistry:
    def __init__(self, options: RoomOptions | None = None):
        self.options = options or RoomOptions()
        self._rooms: dict[str, Room] = {}
        self._lock = threading.Lock()
        self._watchdog: threading.Thread | None = None

    def load_all(self) -> int:
        """Restore every persisted room (after a server restart)."""
        n = 0
        for f in sorted((Path(self.options.state_dir) / "rooms").glob("*/state.json")):
            try:
                room = Room.restore(f, self.options)
            except Exception as e:
                log.error("could not restore room from %s: %s", f, e)
                continue
            with self._lock:
                self._rooms[room.room_id] = room
            n += 1
        return n

    def start_watchdog(self, interval: float = 10.0) -> None:
        def loop():
            while True:
                time.sleep(interval)
                with self._lock:
                    rooms = list(self._rooms.values())
                for r in rooms:
                    try:
                        r.tick(time.time())
                    except Exception as e:
                        log.error("watchdog: room %s: %s", r.room_id, e)
        self._watchdog = threading.Thread(target=loop, daemon=True, name="room-watchdog")
        self._watchdog.start()

    def create(self, room_id: str, config_path: str | None = None, *, dynamic_spec: dict | None = None,
               eval_data_yaml: str | None = None) -> Room:
        if not room_id or not all(ch.isalnum() or ch in "-_" for ch in room_id):
            raise ValueError("room id must be non-empty and use only letters, digits, '-' and '_'")
        with self._lock:
            if room_id in self._rooms:
                raise ValueError(f"room '{room_id}' already exists")
        room = Room(room_id, config_path, dynamic_spec=dynamic_spec, eval_data_yaml=eval_data_yaml,
                    options=self.options)
        with self._lock:
            if room_id in self._rooms:
                raise ValueError(f"room '{room_id}' already exists")
            self._rooms[room_id] = room
        room.save_state()
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
        room.delete_state()

    def list(self) -> list[dict]:
        with self._lock:
            rooms = list(self._rooms.values())
        return [{"room_id": r.room_id, "mode": r.mode, "dynamic": r.dynamic, "node_count": len(r.nodes),
                 "started": r.started, "finished": r.finished} for r in rooms]
