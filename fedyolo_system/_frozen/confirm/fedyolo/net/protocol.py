"""Wire format between fedyolo.net clients and server.

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
