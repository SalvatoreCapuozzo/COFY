"""Attacks on the federation and the server's defence against them (opt-in: federation.robust).

Threat model: one of the N nodes (one of 4 here) is malicious or faulty. It controls its own
data, its training and what it uploads, and it knows the protocol. The server is honest, holds
no data and trains nothing (the federation's rules), so a defence can only look at the uploads.

Attacks on the upload (simulated here at the server, as the attacker's client would send them;
data attacks -- corrupted labels, a backdoor trigger -- are a poisoned copy of the attacker's
data instead, tools/make_attack_data.py):
  noise          a faulty node: its update is replaced by Gaussian noise, `factor` x the honest
                 update's size, tensor by tensor
  signflip       pulled - factor x honest update (pushes the model away from its own objective)
  scale          pulled + factor x honest update (model replacement by boosting, Bagdasaryan et al. 2020)
  bnpoison       BatchNorm running statistics corrupted: mean + factor x std, variance x factor^2
  rowhijack      claims every class and uploads, for the classes it does NOT own, rows that fire
                 everywhere (bias + factor): blocked by ownership from the server's registry
  weightinflate  reports factor x its number of images (the FedAvg weights of parallel sync)

Defence "ora" -- ownership-aware robust aggregation -- server side:
  1. ownership comes from the server's registry, never from the upload (also without "ora",
     as in cofy/net; enforce_ownership: false only to show what it prevents);
  2. norm bounds per part of the update -- shared weights, BatchNorm statistics, each owned class
     row: a part larger than clip_kappa x the median size of that part in the other nodes' recent
     updates is scaled down to that bound. The reference needs recent updates from at least 3 other
     nodes (with one attacker among 4, their median is an honest node's); before that, a part is
     accepted as sent. A class row has a single owner, so its bound uses every node's row sizes;
  3. parallel sync: coordinate-wise trimmed mean of the shared deltas (the `trim` highest and lowest
     values dropped), unweighted: client-reported image counts are not trusted.
  4. (candidate defence P4, exploratory; history_bound "run", asynchronous schemes) a node's consecutive
     merges, with no other node's merge in between, are bounded jointly for the shared weights and the
     BatchNorm statistics: their cumulative change must stay within the bound of one update. Per-update
     bounds cannot stop a direction that persists when no honest update follows (the concurrent seed
     whose attacker sent its last three uploads in a row); a window over interleaved updates would not
     separate it either, because honest nodes also pull the statistics towards their own data every time.
     Class rows are left out: they affect only their owner's classes.
Every decision is appended to <output_dir>/robust_log.jsonl (node, part sizes, bounds, scaling).
"""
from __future__ import annotations

import json
import statistics
from collections import deque
from pathlib import Path

import torch

BN_SUFFIXES = ("running_mean", "running_var")


def attacker_name(cfg) -> str:
    """Name of the attacking node ('' if none). 'seed' picks node seed mod N, so seeds rotate it."""
    a = cfg.federation.robust.attacker
    if a == "seed":
        return cfg.nodes[cfg.seed % len(cfg.nodes)].name
    return a


def poison_bn_stats(stats: dict, factor: float) -> dict:
    """Calibration-time BatchNorm poisoning: the same corruption as the upload attack `bnpoison`,
    applied to the per-layer (mean, variance) a node sends in the calibration phase."""
    return {n: (m.float() + factor * v.float().clamp_min(0).sqrt(), v.float() * factor ** 2)
            for n, (m, v) in stats.items()}


def _parts(sd_keys, per_class_keys):
    shared = [k for k in sd_keys if k not in per_class_keys and not k.endswith(BN_SUFFIXES)
              and not k.endswith("num_batches_tracked")]
    bn = [k for k in sd_keys if k.endswith(BN_SUFFIXES)]
    return shared, bn


def _norm(sd, ref, keys):
    return float(torch.sqrt(sum(((sd[k].float() - ref[k].float()) ** 2).sum() for k in keys)))


def _row_norm(sd, ref, pairs, c):
    return float(torch.sqrt(sum(((sd[k][c].float() - ref[k][c].float()) ** 2).sum() for p in pairs for k in p)))


class Robust:
    """Attack (for the attacking node) and defence (for every upload), one instance per server."""

    def __init__(self, cfg, model, out_dir):
        from .model import per_class_param_names
        self.cfg, self.rc = cfg, cfg.federation.robust
        self.attacker = attacker_name(cfg)
        self.pairs = per_class_param_names(model)
        self.per_class_keys = {k for p in self.pairs for k in p}
        self.registry = {n.name: n.owned_global_ids(cfg.global_classes) for n in cfg.nodes}
        self.hist = {part: deque(maxlen=self.rc.history * len(cfg.nodes)) for part in ("shared", "bn", "row")}
        self.n_seen = 0
        self.run_node, self.run_delta = None, {}   # history_bound "run": the current run of consecutive merges
        self.log_path = Path(out_dir) / "robust_log.jsonl"

    @property
    def active(self) -> bool:
        return bool(self.attacker and self.rc.attack != "none") or self.rc.defense != "none"

    def _log(self, rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ── attack ────────────────────────────────────────────────────────────

    def attack(self, result, pulled_sd):
        """The upload the attacker sends instead of its honest one (other nodes: unchanged)."""
        rc = self.rc
        if result.name != self.attacker or rc.attack == "none":
            return result
        sd, f = result.state_dict, rc.attack_factor
        gen = torch.Generator().manual_seed(self.cfg.seed * 100_003 + self.n_seen)
        out = {}
        for k, v in sd.items():
            if not torch.is_floating_point(v):
                out[k] = v
                continue
            base, d = pulled_sd[k].float(), v.float() - pulled_sd[k].float()
            if rc.attack == "noise" and not k.endswith(BN_SUFFIXES):
                std = float(d.std()) if d.numel() > 1 else float(d.abs().mean())
                new = base + f * std * torch.randn(d.shape, generator=gen)
            elif rc.attack == "signflip" and not k.endswith(BN_SUFFIXES):
                new = base - f * d
            elif rc.attack == "scale" and not k.endswith(BN_SUFFIXES):
                new = base + f * d
            elif rc.attack == "bnpoison" and k.endswith("running_mean"):
                new = v.float() + f * torch.sqrt(sd[k[:-len("running_mean")] + "running_var"].float().clamp_min(0))
            elif rc.attack == "bnpoison" and k.endswith("running_var"):
                new = v.float() * f ** 2
            else:
                new = v.float()
            out[k] = new.to(v.dtype)
        if rc.attack == "rowhijack":
            own = set(self.registry[result.name])
            for wk, bk in self.pairs:
                for c in range(self.cfg.nc):
                    if c not in own:
                        out[bk][c] = out[bk][c] + f
            result.owned_global_ids = list(range(self.cfg.nc))   # what the upload claims
        if rc.attack == "weightinflate":
            result.num_images = int(result.num_images * f)
        result.state_dict = out
        return result

    # ── defence ───────────────────────────────────────────────────────────

    def _bound(self, part, node):
        """clip_kappa x median over the other nodes of their median recent size of `part`, or None."""
        per_node = {}
        for nd, x in self.hist[part]:
            if nd != node:
                per_node.setdefault(nd, []).append(x)
        if len(per_node) < 3:
            return None
        return self.rc.clip_kappa * statistics.median(statistics.median(v) for v in per_node.values())

    def _run_scale(self, part, sd, pulled_sd, keys, s, b):
        """history_bound 'run': the largest scale <= s keeping |R + scale * d| <= b, where R is the
        cumulative change of the node's current run of consecutive merges and d its new update."""
        R = self.run_delta.get(part)
        if R is None or b is None:
            return s
        rr = sum(float((R[k] ** 2).sum()) for k in keys)
        if rr >= b * b:
            return 0.0
        rd = sum(float((R[k] * (sd[k].float() - pulled_sd[k].float())).sum()) for k in keys)
        dd = sum(float(((sd[k].float() - pulled_sd[k].float()) ** 2).sum()) for k in keys)
        if dd == 0.0:
            return s
        root = (-rd + (rd * rd - dd * (rr - b * b)) ** 0.5) / dd   # positive root of |R + x d|^2 = b^2
        return max(0.0, min(s, root))

    def screen(self, result, pulled_sd, async_push=False):
        """Ownership from the registry, then the per-part norm bounds (defence 'ora'). Returns the
        upload to aggregate; logs every decision. Called for every upload, attacker or not.
        `async_push`: one merge per upload (asynchronous schemes), where history_bound applies."""
        rc = self.rc
        claimed = list(result.owned_global_ids)
        if rc.enforce_ownership:
            result.owned_global_ids = list(self.registry[result.name])
        rec = {"n": self.n_seen, "node": result.name, "attacker": result.name == self.attacker,
               "claimed_extra": sorted(set(claimed) - set(self.registry[result.name]))}
        self.n_seen += 1
        if rc.defense == "none":
            self._log(rec)
            return result
        sd = dict(result.state_dict)
        shared, bn = _parts(sd.keys(), self.per_class_keys)
        run = async_push and rc.history_bound == "run"
        if run and result.name != self.run_node:   # another node merged in between: a new run starts
            self.run_node, self.run_delta = result.name, {}
        for part, keys in (("shared", shared), ("bn", bn)):
            n = _norm(sd, pulled_sd, keys)
            b = self._bound(part, result.name)
            self.hist[part].append((result.name, n))
            s = 1.0 if b is None or n <= b else b / n
            rec[part] = {"norm": n, "bound": b, "scale": s}
            if run:
                s0, s = s, self._run_scale(part, sd, pulled_sd, keys, s, b)
                rec[part].update({"run_length": len(self.run_delta.get("_n", [])) + 1, "scale": s,
                                  "scale_single": s0})
                d = {k: s * (sd[k].float() - pulled_sd[k].float()) for k in keys}
                R = self.run_delta.get(part)
                self.run_delta[part] = d if R is None else {k: R[k] + d[k] for k in keys}
            if s < 1.0:
                for k in keys:
                    base = pulled_sd[k].float()
                    new = base + s * (sd[k].float() - base)
                    if k.endswith("running_var"):
                        new = new.clamp_min(0.0)
                    sd[k] = new.to(sd[k].dtype)
        rows = {}
        for c in result.owned_global_ids:
            n = _row_norm(sd, pulled_sd, self.pairs, c)
            b = self._bound("row", result.name)
            s = 1.0 if b is None or n <= b else b / n
            rows[c] = s
            if s < 1.0:
                for p in self.pairs:
                    for k in p:
                        t = sd[k].clone()
                        t[c] = pulled_sd[k][c] + s * (t[c] - pulled_sd[k][c])
                        sd[k] = t
        for c in result.owned_global_ids:   # appended after the loop: rows of one upload never bound each other
            self.hist["row"].append((result.name, _row_norm(result.state_dict, pulled_sd, self.pairs, c)))
        if run:
            self.run_delta.setdefault("_n", []).append(self.n_seen)
        rec["rows_scaled"] = {str(c): s for c, s in rows.items() if s < 1.0}
        result.state_dict = sd
        self._log(rec)
        return result

    def process(self, result, pulled_sd, async_push=False):
        """One upload: the attack (attacker only), then the defence. `async_push` for the
        asynchronous / sequential servers, which merge one upload at a time."""
        return self.screen(self.attack(result, pulled_sd), pulled_sd, async_push)

    def trimmed_shared(self, results, prev_global, keys):
        """Parallel sync, defence 'ora': coordinate-wise trimmed mean of the shared deltas."""
        t = self.rc.trim
        out = {}
        for k in keys:
            ref = prev_global[k]
            if not torch.is_floating_point(ref):
                out[k] = results[0].state_dict[k]
                continue
            d = torch.stack([r.state_dict[k].float() - ref.float() for r in results])
            if t > 0 and len(results) > 2 * t:
                d = d.sort(dim=0).values[t:len(results) - t]
            new = ref.float() + d.mean(dim=0)
            if k.endswith("running_var"):
                new = new.clamp_min(0.0)
            out[k] = new.to(ref.dtype)
        return out
