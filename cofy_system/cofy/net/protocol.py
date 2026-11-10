"""Wire format between cofy.net clients and server.

Only plain tensors and plain Python values cross the network, and both sides
deserialize with torch.load(weights_only=True): unpickling arbitrary objects
from a peer would let anyone holding a room password execute code on the
receiver. The server also validates every upload against the room's own
model (same keys, same shapes) and never trusts the client for which classes
it owns -- that comes from the server's own node registry.

Uploads deliberately omit the node's optimizer state: momentum buffers stay
on the client (it passes them back into its own next round), which also
halves the upload size.
"""

from __future__ import annotations

import io

import torch

from ..client import NodeRoundResult

MAX_NODES_CLASSES = 100_000


class ProtocolError(ValueError):
    """Malformed or inconsistent payload (the server answers 400)."""


def _load(raw: bytes) -> dict:
    try:
        obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as e:
        raise ProtocolError(f"payload is not a valid tensor archive: {e}") from None
    if not isinstance(obj, dict):
        raise ProtocolError("payload must be a dict")
    return obj


def _dump(obj: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


# ── server -> client: the global model ─────────────────────────────────────
def encode_global(state_dict: dict, round_or_version: int) -> bytes:
    return _dump({"state_dict": state_dict, "round_or_version": int(round_or_version)})


def decode_global(raw: bytes) -> tuple[dict, int]:
    obj = _load(raw)
    sd, marker = obj.get("state_dict"), obj.get("round_or_version")
    if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise ProtocolError("global model payload has no tensor state_dict")
    if not isinstance(marker, int):
        raise ProtocolError("global model payload has no integer round/version")
    return sd, marker


# ── client -> server: one node's update ───────────────────────────────────
def encode_update(result: NodeRoundResult) -> bytes:
    return _dump({
        "state_dict": {k: v.detach().cpu() for k, v in result.state_dict.items()},
        "num_images": int(result.num_images),
        "class_counts": {int(k): int(v) for k, v in result.class_counts.items()},
        "num_steps": int(result.num_steps),
        "fisher": ({k: v.detach().cpu() for k, v in result.fisher.items()} if result.fisher else None),
    })


def decode_update(raw: bytes, *, node_name: str, reference_sd: dict, owned_global_ids: list[int],
                  nc: int) -> NodeRoundResult:
    """Rebuild a NodeRoundResult from an upload, rejecting anything that
    doesn't match the room's model exactly."""
    obj = _load(raw)
    sd = obj.get("state_dict")
    if not isinstance(sd, dict):
        raise ProtocolError("update has no state_dict")
    if set(sd) != set(reference_sd):
        missing, extra = set(reference_sd) - set(sd), set(sd) - set(reference_sd)
        raise ProtocolError(f"state_dict keys differ from the room's model "
                            f"(missing {sorted(missing)[:3]}, unexpected {sorted(extra)[:3]})")
    for k, v in sd.items():
        ref = reference_sd[k]
        if not isinstance(v, torch.Tensor) or v.shape != ref.shape or v.is_floating_point() != ref.is_floating_point():
            raise ProtocolError(f"tensor '{k}' has the wrong type or shape")
        if v.is_floating_point() and not torch.isfinite(v).all():
            raise ProtocolError(f"tensor '{k}' contains NaN/Inf")
    n_img, steps = obj.get("num_images"), obj.get("num_steps", 0)
    if not isinstance(n_img, int) or n_img <= 0 or not isinstance(steps, int) or steps < 0:
        raise ProtocolError("num_images must be a positive int and num_steps a non-negative int")
    counts = obj.get("class_counts") or {}
    if (not isinstance(counts, dict) or len(counts) > MAX_NODES_CLASSES
            or not all(isinstance(k, int) and 0 <= k < nc and isinstance(v, int) and v >= 0 for k, v in counts.items())):
        raise ProtocolError("class_counts must map class ids in [0, nc) to non-negative ints")
    fisher = obj.get("fisher")
    if fisher is not None:
        if not isinstance(fisher, dict) or not all(
                k in reference_sd and isinstance(v, torch.Tensor) and v.shape == reference_sd[k].shape
                for k, v in fisher.items()):
            raise ProtocolError("fisher must map parameter names to tensors of the parameter's shape")
    return NodeRoundResult(
        name=node_name, state_dict=sd, num_images=n_img, class_counts=counts,
        owned_global_ids=list(owned_global_ids), optimizer_state={}, num_steps=steps, fisher=fisher,
    )


# ── calibration phase (cofy/calibrate.py over the network) ──────────────
def encode_bn_stats(stats: dict, n_batches: int) -> bytes:
    """Client -> server: BatchNorm mean/var of the final model on this node's images."""
    flat = {}
    for name, (mean, var) in stats.items():
        flat[f"{name}.running_mean"], flat[f"{name}.running_var"] = mean.detach().cpu(), var.detach().cpu()
    return _dump({"stats": flat, "n_batches": int(n_batches)})


def decode_bn_stats(raw: bytes, *, reference_sd: dict) -> tuple[dict, int]:
    """Every BatchNorm layer of the room's model, right shapes, finite, variances >= 0."""
    obj = _load(raw)
    flat, n = obj.get("stats"), obj.get("n_batches")
    keys = {k for k in reference_sd if k.endswith((".running_mean", ".running_var"))}
    if not isinstance(flat, dict) or set(flat) != keys:
        raise ProtocolError("BatchNorm statistics must cover exactly the model's BatchNorm layers")
    for k, v in flat.items():
        if not isinstance(v, torch.Tensor) or v.shape != reference_sd[k].shape or not torch.isfinite(v).all():
            raise ProtocolError(f"BatchNorm statistic '{k}' has the wrong shape or is not finite")
        if k.endswith("running_var") and (v < 0).any():
            raise ProtocolError(f"negative variance in '{k}'")
    if not isinstance(n, int) or n <= 0:
        raise ProtocolError("n_batches must be a positive int")
    names = [k[:-len(".running_mean")] for k in flat if k.endswith(".running_mean")]
    return {nm: (flat[f"{nm}.running_mean"], flat[f"{nm}.running_var"]) for nm in names}, n


def encode_rows(rows: dict) -> bytes:
    """Client -> server: this node's re-fitted class rows ({param: rows of its owned classes})."""
    return _dump({"rows": {k: v.detach().cpu() for k, v in rows.items()}})


def decode_rows(raw: bytes, *, reference_sd: dict, per_class_keys: set, n_owned: int) -> dict:
    """Only per-class parameters, one row per class the server registered for the node."""
    obj = _load(raw)
    rows = obj.get("rows")
    if not isinstance(rows, dict) or not rows or not set(rows) <= per_class_keys:
        raise ProtocolError("rows must map per-class parameter names to tensors")
    for k, v in rows.items():
        ref = reference_sd[k]
        if (not isinstance(v, torch.Tensor) or tuple(v.shape) != (n_owned, *ref.shape[1:])
                or not torch.isfinite(v).all()):
            raise ProtocolError(f"rows of '{k}' must be {n_owned} finite rows of shape {tuple(ref.shape[1:])}")
    return rows
