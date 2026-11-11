#!/usr/bin/env python3
"""Checks that the candidate defences keep COFY's rules (paper, Problem setting):
  R1  nodes learn about other nodes' classes only through the model they download;
  R2  the server holds no labelled data and trains nothing: it may only combine weights and statistics;
  R3  each class has exactly one owner, and only the owner can change that class's output parameters.

It drives the real server code on the WBC architecture (CPU only, no training, no data labels):
a sequence of adversarial uploads -- every node also changes the rows of every class and claims them,
one node uploads several times in a row -- goes through the defence (ownership + norm bounds + P4)
and the asynchronous merge, then P3's check with scripted coverage reports. It asserts:
  R3  a merge changes the rows of a class only when the class's owner uploaded; P4 changes no row
      (rows are identical with P4 on and off); P3 only ever sets a class's rows to values the rows
      already had at an earlier version (its owner's, or the initial ones), and changes nothing else;
      P1 (median BatchNorm pooling) changes BatchNorm statistics only;
  vote a liar (the attacker reports every class it does not own as covering everything) cannot get an
      honest class rolled back; the sign-flipped class is rolled back after two flagged checks and then
      stays frozen however often its owner uploads;
  P3b (exclusion, through the real asynchronous push and the parallel-sync merge steps) only the attacker
      is excluded; its later uploads change nothing and create no version; honest uploads are still merged;
      rows still change only on their owner's merges; a targeted liar is outvoted, and once excluded its
      reports no longer count; in parallel sync its classes keep the previous rows;
  R2  P3's node step reads images only (no label file), and the server receives one number per class;
  defaults: with the default configuration every candidate defence is off.

    CUDA_VISIBLE_DEVICES= python tools/check_rules.py
"""
import collections, inspect, sys, tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cofy.calibrate import pool_bn_stats
from cofy.client import NodeRoundResult
from cofy.config import FederationConfig, RobustConfig, load_config
from cofy.model import build_model, per_class_param_names
from cofy import peercheck
from cofy.peercheck import PeerCheck
from cofy.robust import Robust
from cofy.server import AsyncFedServer

FAILS = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        FAILS.append(what)


def rows(sd, pairs, c):
    return torch.cat([sd[k][c].flatten().float() for p in pairs for k in p])


def run(history_bound, coverage_script):
    """One adversarial sequence; returns per-step global state dicts and the check results."""
    import dataclasses
    cfg = load_config(ROOT / "configs/variants/rb_conc_signflip_ora_p3_budget/wbc_s0.yaml")
    rc = dataclasses.replace(cfg.federation.robust, history_bound=history_bound, peer_check=True)
    out = Path(tempfile.mkdtemp(prefix="check_rules_"))
    cfg = dataclasses.replace(cfg, output_dir=str(out),
                              federation=dataclasses.replace(cfg.federation, robust=rc))
    torch.manual_seed(0)
    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=False, class_names=cfg.global_classes)
    srv = AsyncFedServer.__new__(AsyncFedServer)
    srv.cfg, srv.model, srv.out_dir = cfg, model, out
    srv._global_sd = {k: v.clone() for k, v in model.state_dict().items()}
    srv._tower_keys, srv._applied, srv._version = {}, collections.deque(maxlen=4), 0
    srv._robust = Robust(cfg, model, out)
    orig_images = peercheck.node_images
    peercheck.node_images = lambda node, cfg, n, seed: torch.zeros(1, 3, 8, 8, dtype=torch.uint8)  # no data needed here
    srv._peer = PeerCheck(cfg, model, srv._global_sd, out)
    peercheck.node_images = orig_images
    pairs = per_class_param_names(model)
    per_class = {k for p in pairs for k in p}
    owner = {c: n.name for n in cfg.nodes for c in n.owned_global_ids(cfg.global_classes)}
    att = srv._peer.attacker
    order = ["node_A", "node_A", "node_B", "node_C", "node_D", "node_A", "node_A", "node_A", "node_B",
             "node_C", "node_D", "node_B", "node_A", "node_C", "node_D", "node_A", "node_A", "node_B"]
    history = {c: [rows(srv._global_sd, pairs, c)] for c in range(cfg.nc)}
    gen = torch.Generator().manual_seed(1)
    res = {"merge_only_owner": True, "p3_only_earlier_values": True, "p3_only_rows": True,
           "frozen_after_quarantine": True, "states": []}
    for step, nm in enumerate(order):
        pulled = {k: v.clone() for k, v in srv._global_sd.items()}
        up = {}
        for k, v in pulled.items():
            if torch.is_floating_point(v):
                noise = torch.randn(v.shape, generator=gen) * 0.01
                up[k] = (v.float() + noise).clamp_min(0).to(v.dtype) if k.endswith("running_var") else (v.float() + noise).to(v.dtype)
            else:
                up[k] = v.clone()
        r = NodeRoundResult(name=nm, state_dict=up, num_images=100, class_counts={}, optimizer_state={},
                            owned_global_ids=list(range(cfg.nc)))          # claims every class
        before = {k: v.clone() for k, v in srv._global_sd.items()}
        r = srv._robust.process(r, pulled, async_push=True)
        srv._aggregate_async(r, pulled, 0)
        srv._version += 1
        for c in range(cfg.nc):
            changed = not torch.equal(rows(before, pairs, c), rows(srv._global_sd, pairs, c))
            if changed and owner[c] != nm:
                res["merge_only_owner"] = False
        merged = {k: v.clone() for k, v in srv._global_sd.items()}
        srv._peer._rates = lambda sd, s=step: coverage_script(s, cfg, owner, att)
        q_before = set(srv._peer.quarantined)
        srv._peer.after_merge(srv._global_sd, [], srv._version)
        for k in srv._global_sd:
            if k not in per_class and not torch.equal(merged[k], srv._global_sd[k]):
                res["p3_only_rows"] = False
        for c in range(cfg.nc):
            now = rows(srv._global_sd, pairs, c)
            if not torch.equal(now, rows(merged, pairs, c)):
                if not any(torch.equal(now, h) for h in history[c]):
                    res["p3_only_earlier_values"] = False
            if c in q_before and not torch.equal(now, history[c][-1]):
                res["frozen_after_quarantine"] = False
            history[c].append(now)
        res["states"].append({k: v.clone() for k, v in srv._global_sd.items() if k in per_class})
    res["quarantined"] = dict(srv._peer.quarantined)
    res["attacker_classes"] = sorted(c for c, o in owner.items() if o == att)
    return res


def run_exclusion():
    """P3b through the real asynchronous push(): the sign-flipped class floods from the 6th upload on;
    after the rollback its owner is excluded. The attacker lies in a targeted way throughout (node_B's
    class reported as flooding, one vote of three); from the 12th upload on, one honest node (node_C)
    reports the same, so that the excluded liar's vote would decide if it still counted."""
    import dataclasses, threading
    cfg = load_config(ROOT / "configs/variants/rb_conc_signflip_ora_p3_budget/wbc_s0.yaml")
    rc = dataclasses.replace(cfg.federation.robust, peer_check=True, peer_exclude=True, peer_lie=False)  # lie scripted
    out = Path(tempfile.mkdtemp(prefix="check_rules_p3b_"))
    cfg = dataclasses.replace(cfg, output_dir=str(out), federation=dataclasses.replace(cfg.federation, robust=rc))
    torch.manual_seed(0)
    model = build_model(cfg.model.arch, cfg.nc, cfg.model.imgsz, pretrained=False, class_names=cfg.global_classes)
    srv = AsyncFedServer.__new__(AsyncFedServer)
    srv.cfg, srv.model, srv.out_dir = cfg, model, out
    srv._global_sd = {k: v.clone() for k, v in model.state_dict().items()}
    srv._tower_keys, srv._applied, srv._version = {}, collections.deque(maxlen=4), 0
    srv._lock, srv._ca2fl, srv._swa, srv._last_by_node = threading.Lock(), None, [], {}
    srv._submission_count, srv._last_ckpt_path, srv._progress, srv._pending_eval = 0, None, None, None
    srv.eval_enabled = False
    srv._robust = Robust(cfg, model, out)
    orig_images = peercheck.node_images
    peercheck.node_images = lambda node, cfg, n, seed: torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
    srv._peer = PeerCheck(cfg, model, srv._global_sd, out)
    peercheck.node_images = orig_images
    pairs = per_class_param_names(model)
    owner = {c: n.name for n in cfg.nodes for c in n.owned_global_ids(cfg.global_classes)}
    att = srv._peer.attacker
    b_class = next(c for c, o in owner.items() if o == "node_B")
    order = ["node_A", "node_B", "node_C", "node_D"] * 6
    srv._total_submissions = len(order)
    gen = torch.Generator().manual_seed(2)
    res = {"merge_only_owner": True, "excluded_changes_nothing": True, "honest_still_merged": True,
           "version_unchanged_on_discard": True, "n_discarded": 0, "targeted_lie_outvoted": True,
           "attacker": att, "attacker_classes": sorted(c for c, o in owner.items() if o == att), "b_class": b_class}
    for step, nm in enumerate(order):
        def script(sd, s=step):
            rep = {n.name: [0.4 if owner[c] == att and s >= 6 else 0.0005 for c in range(cfg.nc)] for n in cfg.nodes}
            rep[att] = [0.4 if c == b_class else 0.0005 for c in range(cfg.nc)]       # targeted lie
            if s >= 12:
                rep["node_C"] = [0.4 if c == b_class else x for c, x in enumerate(rep["node_C"])]
            return rep
        srv._peer._rates = script
        pulled = {k: v.clone() for k, v in srv._global_sd.items()}
        up = {k: ((v.float() + torch.randn(v.shape, generator=gen) * 0.01).clamp_min(0).to(v.dtype)
                  if k.endswith("running_var") else (v.float() + torch.randn(v.shape, generator=gen) * 0.01).to(v.dtype))
              if torch.is_floating_point(v) else v.clone() for k, v in pulled.items()}
        r = NodeRoundResult(name=nm, state_dict=up, num_images=100, class_counts={}, optimizer_state={},
                            owned_global_ids=list(range(cfg.nc)))
        excluded_before, v_before = set(srv._peer.excluded), srv._version
        before = {k: v.clone() for k, v in srv._global_sd.items()}
        srv.push(r, pulled, srv._version)
        if step < 12 and srv._peer.strikes[b_class] > 0:
            res["targeted_lie_outvoted"] = False
        if nm in excluded_before:
            res["n_discarded"] += 1
            if any(not torch.equal(before[k], srv._global_sd[k]) for k in before):
                res["excluded_changes_nothing"] = False
            if srv._version != v_before:
                res["version_unchanged_on_discard"] = False
        else:
            shared = [k for k in before if torch.is_floating_point(before[k]) and not k.endswith(("running_mean", "running_var"))
                      and all(k not in p for p in pairs)]
            if excluded_before and all(torch.equal(before[k], srv._global_sd[k]) for k in shared):
                res["honest_still_merged"] = False
        for c in range(cfg.nc):
            if owner[c] != nm and not torch.equal(rows(before, pairs, c), rows(srv._global_sd, pairs, c)):
                # P3's rollback may reset a row to an earlier value of its owner; any other change breaks R3
                if c not in srv._peer.quarantined:
                    res["merge_only_owner"] = False
    res["excluded"] = dict(srv._peer.excluded)
    res["quarantined"] = dict(srv._peer.quarantined)

    # parallel sync merge steps (server.py, FedServer.run): excluded nodes are left out of the round and
    # their classes keep the previous rows
    from cofy.model import aggregate
    prev = {k: v.clone() for k, v in srv._global_sd.items()}
    ups = []
    for n in cfg.nodes:
        if n.name in srv._peer.excluded:
            continue
        sd = {k: (v.float() + 0.05).to(v.dtype) if torch.is_floating_point(v) and not k.endswith("running_var") else v.clone()
              for k, v in prev.items()}
        ups.append(NodeRoundResult(name=n.name, state_dict=sd, num_images=100, class_counts={}, optimizer_state={},
                                   owned_global_ids=n.owned_global_ids(cfg.global_classes)))
    new = aggregate(node_state_dicts=[u.state_dict for u in ups], node_num_images=[100] * len(ups),
                    node_owned_global_ids=[u.owned_global_ids for u in ups], node_class_counts=[{}] * len(ups),
                    model=model, nc=cfg.nc, prev_global=prev, rule="mean")
    srv._peer.keep_excluded_rows(new, prev)
    exc_classes = [c for c, o in owner.items() if o in srv._peer.excluded]
    res["sync_excluded_rows_kept"] = bool(exc_classes) and all(torch.equal(rows(new, pairs, c), rows(prev, pairs, c))
                                                               for c in exc_classes)
    res["sync_others_updated"] = all(not torch.equal(rows(new, pairs, c), rows(prev, pairs, c))
                                     for c in owner if c not in exc_classes)
    return res


def coverage(step, cfg, owner, att):
    """Scripted reports: honest nodes see every class at 0.05 % coverage, except the attacker's classes,
    which flood from step 6 on (sign flip); the attacker lies about every class it does not own."""
    rep = {}
    for n in cfg.nodes:
        if n.name == att:
            rep[n.name] = [1.0 if owner[c] != att else 0.0005 for c in range(cfg.nc)]
        else:
            rep[n.name] = [0.4 if owner[c] == att and step >= 6 else 0.0005 for c in range(cfg.nc)]
    return rep


def main():
    print("R3 and the vote: adversarial upload sequence through the defence, the merge, P4 and P3")
    on = run("run", coverage)
    off = run("none", coverage)
    check(on["merge_only_owner"] and off["merge_only_owner"],
          "a merge changes a class's rows only when its owner uploads (uploads claim every class)")
    check(all(torch.equal(a[k], b[k]) for a, b in zip(on["states"], off["states"]) for k in a),
          "P4 changes no class row (rows identical with history_bound run and none)")
    check(on["p3_only_rows"], "P3 changes no shared weight and no BatchNorm statistic")
    check(on["p3_only_earlier_values"], "P3 sets a class's rows only to values they had at an earlier version")
    check(set(on["quarantined"]) == set(on["attacker_classes"]),
          f"only the attacker's class is rolled back (rolled back: {on['quarantined']}, "
          f"attacker's: {on['attacker_classes']}); the liar alone flags nothing")
    check(all(v == 8 for v in on["quarantined"].values()), "rolled back at the second flagged check (patience 2)")
    check(on["frozen_after_quarantine"], "a quarantined class stays frozen while its owner keeps uploading")

    print("P3b: exclusion of the rolled-back class's owner (real asynchronous push, parallel sync merge steps)")
    ex = run_exclusion()
    check(ex["targeted_lie_outvoted"],
          "a targeted liar (one honest class reported as flooding) is outvoted before the exclusion")
    check(set(ex["excluded"]) == {ex["attacker"]} and set(ex["quarantined"]) == set(ex["attacker_classes"]),
          f"only the attacker is excluded (excluded: {ex['excluded']}, rolled back: {ex['quarantined']})")
    check(ex["n_discarded"] > 0 and ex["excluded_changes_nothing"],
          f"every upload of the excluded node after the exclusion changes nothing ({ex['n_discarded']} discarded)")
    check(ex["version_unchanged_on_discard"], "a discarded upload creates no new version (no extra staleness)")
    check(ex["honest_still_merged"], "the honest nodes' uploads are still merged after the exclusion")
    check(ex["merge_only_owner"], "with exclusion, a merge still changes a class's rows only when its owner uploads")
    check(ex["b_class"] not in ex["quarantined"],
          f"an excluded liar's reports no longer count: with one honest node also reporting node_B's class as "
          f"flooding, the class is not rolled back (rolled back: {ex['quarantined']})")
    check(ex["sync_excluded_rows_kept"] and ex["sync_others_updated"],
          "parallel sync: the excluded node is left out of the round, its classes keep the previous rows, "
          "the other classes are merged from their owners")

    print("R3 and R2: P1 median pooling")
    model = build_model(load_config(ROOT / "configs/variants/rb_conc_clean_ora/wbc_s0.yaml").model.arch, 7, 640,
                        pretrained=False, class_names=[str(i) for i in range(7)])
    sd = model.state_dict()
    bn = {n: (torch.randn(m.num_features), torch.rand(m.num_features)) for n, m in model.named_modules()
          if isinstance(m, torch.nn.BatchNorm2d)}
    stats = [({n: (a + i, b + i) for n, (a, b) in bn.items()}, 10 * (i + 1)) for i in range(4)]
    for rule in ("mean", "median"):
        out = pool_bn_stats(sd, stats, rule)
        changed = {k for k in sd if not torch.equal(sd[k], out[k])}
        check(all(k.endswith(("running_mean", "running_var")) for k in changed) and changed,
              f"pooling '{rule}' changes BatchNorm statistics only ({len(changed)} tensors)")

    print("R2: what leaves a node in P3")
    import builtins, cv2
    cfg = load_config(ROOT / "configs/variants/rb_seq_clean_ora_p3_budget/wbc_s0.yaml")
    opened, real_open, real_imread, real_read = [], builtins.open, cv2.imread, Path.read_text
    builtins.open = lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1]
    cv2.imread = lambda f, *a, **k: (opened.append(str(f)), real_imread(f, *a, **k))[1]
    Path.read_text = lambda self, *a, **k: (opened.append(str(self)), real_read(self, *a, **k))[1]
    try:
        imgs = peercheck.node_images(cfg.nodes[0], cfg, 32, 0)
    finally:
        builtins.open, cv2.imread, Path.read_text = real_open, real_imread, real_read
    bad = [f for f in opened if "/labels/" in f or f.endswith((".txt", ".cache"))]
    check(len(imgs) == 32 and not bad and all(f.endswith(("data.yaml", ".jpg", ".jpeg", ".png", ".bmp")) for f in opened),
          f"P3's node step opens only its data.yaml and its own images: {len(opened)} files, "
          f"{sum(f.endswith('.jpg') for f in opened)} images, no label file")
    sig = inspect.signature(PeerCheck.after_merge)
    check(list(sig.parameters) == ["self", "sd", "stored", "version"],
          "the server-side step takes the model and its stored versions; each node reports one number per class")

    print("defaults")
    r, f = RobustConfig(), FederationConfig()
    check(r.history_bound == "none" and not r.peer_check and not r.calib_attack and f.calibration_bn_pool == "mean",
          "every candidate defence is off by default (registered behaviour unchanged)")
    print("\nALL CHECKS PASSED" if not FAILS else f"\n{len(FAILS)} CHECK(S) FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
