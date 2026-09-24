"""Server-side merge rules and merge diagnostics for the shared parameters.

Everything here operates on DELTAS (a node's trained weights minus the exact
global snapshot it started the round from), never on raw weights, because the
question this module exists to answer is about *disagreement between nodes*,
which only shows up in the deltas.

Two things live here:

1. `delta_stats()` -- the diagnostic. Under this project's extreme non-IID
   split (one class, exactly one node) each node optimizes a genuinely
   different objective, so the interesting quantity is how much of each
   node's learning survives being averaged with the others'. The headline
   metric is `retained`:

       retained = || Σ_k w_k·δ_k ||_1  /  Σ_k w_k·||δ_k||_1

   1.0 means every node pushed the same direction and the mean keeps all of
   it; 0.0 means the nodes exactly cancelled and averaging destroyed the
   round's learning outright. Reported per layer group, because the whole
   point of the split is that different groups are expected to behave
   differently -- the backbone should transfer across cell types, the
   classification convs feeding the per-class layer are where objectives
   genuinely conflict.

2. `merge_shared()` -- the merge rules. `"mean"` is the existing
   sample-weighted FedAvg, kept bit-identical as the default so nothing
   changes unless asked. `"ties"` adds TIES-Merging's conflict resolution:
   trim each node's smallest-magnitude changes, elect one sign per
   coordinate by summed weighted magnitude, then average only the nodes
   that agree with the elected sign -- so a minority pushing the other way
   is dropped instead of being allowed to cancel the majority out.
"""

from __future__ import annotations

import torch

# Layer groups, resolved by substring against a state_dict key. Order
# matters: the first match wins, and `cv3_class` (the per-class rows) is
# checked before `cv3_mix` because both live under the same `cv3.` prefix.
_HEAD_CLASS_SUFFIXES = (".2.weight", ".2.bias")


def layer_group(key: str, head_prefix: str, per_class_keys: set[str]) -> str:
    """Bucket one state_dict key. `head_prefix` is e.g. "model.23." -- the
    Detect module -- so backbone/neck keys that also contain ".cv3." (C2f and
    C3k2 blocks use that name internally) don't get misfiled into the head."""
    if key in per_class_keys:
        return "cv3_class"
    if key.startswith(head_prefix):
        tail = key[len(head_prefix):]
        if tail.startswith("cv2"):
            return "cv2_box"
        if tail.startswith("cv3") or tail.startswith("one2one_cv3"):
            return "cv3_mix"
        return "head_other"
    if "running_mean" in key or "running_var" in key:
        return "bn_stats"
    return "trunk"


def _groups_for(keys, head_prefix: str, per_class_keys: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k in keys:
        out.setdefault(layer_group(k, head_prefix, per_class_keys), []).append(k)
    return out


def delta_stats(node_state_dicts: list[dict], prev_global: dict, keys: list[str],
                weights: list[float], head_prefix: str,
                per_class_keys: set[str]) -> dict:
    """Per-layer-group disagreement statistics across nodes.

    Streams per key and accumulates scalars only -- never materializes the
    concatenated delta vectors, which would be ~400MB for 4 nodes at this
    model size.
    """
    K = len(node_state_dicts)
    tot_w = sum(weights) or 1.0
    groups = _groups_for(keys, head_prefix, per_class_keys)

    stats = {}
    for gname, gkeys in sorted(groups.items()):
        abs_of_sum = 0.0      # Σ_coords |Σ_k w_k δ_k|
        sum_of_abs = 0.0      # Σ_coords Σ_k w_k |δ_k|
        numel = 0
        all_agree = 0         # coords where every nonzero delta shares a sign
        dots = [[0.0] * K for _ in range(K)]
        sqn = [0.0] * K

        for k in gkeys:
            ref = prev_global[k]
            if not torch.is_floating_point(ref):
                continue
            deltas = [(sd[k].to(torch.float32) - ref.to(torch.float32)).flatten()
                      for sd in node_state_dicts]
            wsum = torch.zeros_like(deltas[0])
            wabs = torch.zeros_like(deltas[0])
            for i, d in enumerate(deltas):
                wsum += d * weights[i]
                wabs += d.abs() * weights[i]
                sqn[i] += float(d.dot(d))
                for j in range(i, K):
                    dots[i][j] += float(d.dot(deltas[j]))
            abs_of_sum += float(wsum.abs().sum())
            sum_of_abs += float(wabs.sum())
            numel += deltas[0].numel()

            signs = torch.stack([torch.sign(d) for d in deltas])
            nz = signs != 0
            # a coord "agrees" if all its nonzero signs are equal
            pos = ((signs > 0) | ~nz).all(dim=0)
            neg = ((signs < 0) | ~nz).all(dim=0)
            all_agree += int((pos | neg).sum())

        cos = []
        for i in range(K):
            for j in range(i + 1, K):
                den = (sqn[i] ** 0.5) * (sqn[j] ** 0.5)
                if den > 0:
                    cos.append(dots[i][j] / den)
        stats[gname] = {
            "n_keys": len(gkeys),
            "n_params": numel,
            "retained": (abs_of_sum / sum_of_abs) if sum_of_abs > 0 else None,
            "mean_pairwise_cosine": (sum(cos) / len(cos)) if cos else None,
            "sign_agreement": (all_agree / numel) if numel else None,
            "mean_abs_delta": (sum_of_abs / tot_w / numel) if numel else None,
        }
    return stats


def step_scales(weights: list[float], node_steps: list[float] | None) -> list[float] | None:
    """FedNova's per-node delta rescaling, or None if it isn't applicable.

    Nodes run a uniform `local_epochs` but hold different amounts of data, so
    they take different numbers of gradient steps per round and arrive with
    systematically different-magnitude deltas -- and sample weighting then
    COMPOUNDS that, since the node that took the most steps is usually also
    the one holding the most images. FedNova removes the step-count component
    by rescaling each delta to a common effective step count before weighting:

        δ'_k = δ_k · (τ_eff / τ_k),   τ_eff = Σ_k w_k·τ_k / Σ_k w_k

    so the merged direction reflects data proportions rather than step-count
    accidents, while the overall update magnitude is preserved.
    """
    if not node_steps or any(s <= 0 for s in node_steps):
        return None
    tw = sum(weights) or 1.0
    tau_eff = sum(w * s for w, s in zip(weights, node_steps)) / tw
    return [tau_eff / s for s in node_steps]


def merge_shared(node_state_dicts: list[dict], weights: list[float], keys: list[str],
                 prev_global: dict | None, rule: str = "mean",
                 trim: float = 0.2, node_steps: list[float] | None = None) -> dict:
    """Combine the shared keys. `rule="mean"` with no `node_steps` reproduces
    the original sample-weighted FedAvg exactly; `rule="ties"` and FedNova step
    normalization both need `prev_global` to form deltas, and are skipped if it
    isn't available."""
    if prev_global is None:
        rule, node_steps = ("mean" if rule == "ties" else rule), None

    scales = step_scales(weights, node_steps)
    total = sum(weights)
    out = {}
    for k in keys:
        ref = node_state_dicts[0][k]
        if not torch.is_floating_point(ref):
            # buffers like num_batches_tracked: keep the largest node's value
            out[k] = node_state_dicts[weights.index(max(weights))][k]
            continue

        if rule == "mean" and scales is None:
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for sd, w in zip(node_state_dicts, weights):
                acc += sd[k].to(torch.float32) * (w / total)
            out[k] = acc.to(ref.dtype)
            continue

        # --- delta-space path: FedNova rescaling and/or TIES ---
        base = prev_global[k].to(torch.float32)
        deltas = torch.stack([sd[k].to(torch.float32) - base for sd in node_state_dicts])
        w = torch.tensor(weights, dtype=torch.float32).view(-1, *([1] * (deltas.dim() - 1)))

        if scales is not None:
            s = torch.tensor(scales, dtype=torch.float32).view(-1, *([1] * (deltas.dim() - 1)))
            deltas = deltas * s

        if rule != "ties":
            # plain weighted mean, but in delta space so the rescaling applies
            out[k] = (base + (deltas * w).sum(dim=0) / total).to(ref.dtype)
            continue

        if trim > 0:
            # per node, zero the smallest-magnitude `trim` fraction of entries
            flat = deltas.reshape(deltas.shape[0], -1)
            n_drop = int(flat.shape[1] * trim)
            if n_drop > 0:
                thresh = flat.abs().kthvalue(n_drop, dim=1, keepdim=True).values
                flat = torch.where(flat.abs() > thresh, flat, torch.zeros_like(flat))
                deltas = flat.view_as(deltas)

        elected = torch.sign((deltas * w).sum(dim=0))
        agree = (torch.sign(deltas) == elected) & (elected != 0)
        wmask = w * agree
        denom = wmask.sum(dim=0)
        merged = (deltas * wmask).sum(dim=0) / denom.clamp(min=1e-12)
        # coords where nothing agreed (or the election tied at zero) don't move
        merged = torch.where(denom > 0, merged, torch.zeros_like(merged))
        out[k] = (base + merged).to(ref.dtype)
    return out
