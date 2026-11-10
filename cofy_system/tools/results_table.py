#!/usr/bin/env python3
"""Fed:Cent / Fed:Oracle table for the primary study and every variant, from the runs finished so far.

    python tools/results_table.py                       # print
    python tools/results_table.py --out FILE            # also write it to FILE
    watch -n 300 python tools/results_table.py          # live

Denominators are the centralized and oracle mAP50 of the same dataset and seed:
the primary study's (seeds 0-2, and kfm12 seed 3), else runs/variants/_baseline
(centralized only, trained by tools/run_baseline.py for extra seeds). CI-low =
lower bound of the two-sided 95% CI of the mean Fed:Cent over seeds
(main criterion since 2026-10-08: PASS if >= 85%, user decision, post hoc; pre-registered:
90%, PRE90 column). "vs primary" is the mean mAP50 difference
against the primary study's same mode, paired by seed.

The same functions are used by tools/search_rules.py (winners, extra seeds).
"""
import json, re, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "runs/variants"
DATASETS, MARGIN, N_MAX = ["kfm12", "wbc", "chestx"], 85.0, 12   # MARGIN: main threshold (was 90, see PREREGISTRATION deviations)
PRE_MARGIN = 90.0   # the pre-registered threshold, still reported
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093}
KEY = {"sync": "federated_sync", "async": "federated_async"}
ORDER_CHECK = re.compile(r"^(order_r\d+|shufswa_r[1-9]\d*)$")   # node-order checks, not candidates
PRIMARY_SEEDS = [0, 1, 2]
SEARCH_PREFIX = "v2_"   # rows of the current search protocol (LR decay + plateau budget, FOLLOWUP_PLAN §8.6):
                        # only these can win or get extra seeds; their reference is v2_central
CONFIRM_PREFIX = "confirm_v2"   # pre-registered confirmation of the two final schemes (analysis/CONFIRMATORY_PREREG.md):
                                # protocol v2, fresh seeds, reference v2_central; never searched, never winners
BASELINE_PREFIX = "bl_"   # standard competitors under protocol v2 (FedProx, FedBuff, CA2FL): reference rows,
                          # shown with the v2 search but never candidates, winners or given extra seeds
CONFIRM_ROWS = ("confirm_v2sync+cal", "confirm_v2async+acalr")   # the two pre-registered rows (sync, async)


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_plan():
    """[(name, mode, [seeds])] from runs/variants/_plan.txt."""
    out = []
    f = VAR / "_plan.txt"
    for line in (f.read_text().splitlines() if f.exists() else []):
        p = line.split("#", 1)[0].split()
        if p:
            out.append((p[0], p[1].split("@", 1)[0], [int(s) for s in p[2:]]))   # '@ds' restriction dropped
    return out


def v2_protocol(name):
    """Rows trained with protocol v2 (search or confirmation): their reference is v2_central."""
    return name.startswith((SEARCH_PREFIX, CONFIRM_PREFIX, BASELINE_PREFIX))


def variant_cfg_text(src):
    """Text of one config of variant `src` (seed-0 kfm12 if present, else the first one): the
    settings shared by all its seeds and datasets (scheme, clock, calibration options)."""
    if src in ("primary", "primary_ext"):
        return (ROOT / "configs/study/kfm12_s0.yaml").read_text()
    d = ROOT / f"configs/variants/{src}"
    first = d / "kfm12_s0.yaml"
    if not first.exists():
        first = next(iter(sorted(d.glob("*.yaml"))), None)
    return first.read_text() if first else ""


def search_log():
    return load(VAR / "_search_log.json") or {}


def split_row(name):
    """'<src>+cal' / '<src>+acal' / '<src>' -> (src, suffix). +cal = the calibration round
    (synchronous: all nodes once more at the end); +acal = the asynchronous progressive
    calibration, complete once every node has contributed (virtual-clock runs)."""
    for suf in ("+acalr", "+acal", "+cal"):
        if name.endswith(suf):
            return name[:-len(suf)], suf
    return name, ""


def reference(ds, s, ref="centralized", row=""):
    """Centralized / oracle mAP50 of the same seed and protocol as `row`. Same post-processing
    rule (user decision 2026-10-03): a '+cal' row is compared with the centralized baseline
    given the same final calibration round (tools/calibrate_checkpoints.py)."""
    if ref == "centralized" and split_row(row)[1]:
        src = "v2_centralized" if v2_protocol(row) else "centralized"
        d = load(VAR / f"calib/{src}/{ds}/s{s}/centralized.json")
        return d["bn+head"]["map50"] if d and "bn+head" in d else None
    paths = ([VAR / f"v2_central/{ds}/s{s}/summary.json"] if v2_protocol(row) else
             [ROOT / f"runs/study/{ds}/s{s}/summary.json", VAR / f"_baseline/{ds}/s{s}/summary.json"])
    for p in paths:
        d = load(p)
        if d and ref in d and "map50" in d[ref]:
            return d[ref]["map50"]
    return None


def candidates():
    """[(row name, mode, declared seeds)]: every table row. '<src>+cal' rows come from
    posthoc plan lines; primary seeds beyond 0-2 come from the primary_ext variant."""
    plan = read_plan()
    ext = {m: seeds for n, m, seeds in plan if n == "primary_ext"}
    cal_ext = {}
    for n, m, seeds in plan:
        if n == "calib_primary_ext":
            for md in m.split(":")[2].split(","):
                cal_ext[md] = seeds
    rows = [("primary", md, sorted(set(PRIMARY_SEEDS + ext.get(md, [])))) for md in ("sync", "async")]
    for n, m, seeds in plan:
        if (n in ("primary_ext", "calib_primary_ext") or m.startswith("baseline") or m == "budget"
                or n.startswith(("rb_", "rbsmoke_"))):   # robustness study: tools/robust_eval.py
            continue
        if m.startswith("posthoc:"):
            _, src, modes = m.split(":")
            if src not in ("centralized", "v2_centralized"):
                rows += [(f"{src}+cal", md,
                          sorted(set(seeds + cal_ext.get(md, []))) if src == "primary" else seeds)
                         for md in modes.split(",")]
                cfgt = variant_cfg_text(src)
                if "async_clock: virtual" in cfgt:
                    rows += [(f"{src}+acal", md, seeds) for md in modes.split(",")]
                    if "progressive_refresh: true" in cfgt:
                        rows += [(f"{src}+acalr", md, seeds) for md in modes.split(",")]
        else:
            rows.append((n, m, seeds))
    merged = {}   # one row per (name, mode): a variant may have one plan line per seed (confirmation, seed-major)
    for name, mode, seeds in rows:
        merged.setdefault((name, mode), []).extend(s for s in seeds if s not in merged.get((name, mode), []))
    return [(name, mode, seeds) for (name, mode), seeds in merged.items()]


def value(name, mode, ds, s):
    src, suf = split_row(name)
    stage = {"+cal": "bn+head", "+acal": "acal", "+acalr": "acal_refresh"}.get(suf)
    srcs = [src, "primary_ext"] if src == "primary" else [src]
    for sr in srcs:
        if stage:
            d = load(VAR / f"calib/{sr}/{ds}/s{s}/{mode}.json")
            if d and stage in d:
                return d[stage]["map50"]
        else:
            p = (ROOT / f"runs/study/{ds}/s{s}/summary.json" if sr == "primary"
                 else VAR / f"{sr}/{ds}/s{s}/summary.json")
            d = load(p)
            if d and KEY[mode] in d and "map50" in d[KEY[mode]]:
                return d[KEY[mode]]["map50"]
    return None


def n_required(mean, sd, n_now):
    """Smallest n (n_now < n <= N_MAX) whose projected CI-low (same mean and sd) is >= MARGIN, else None."""
    if mean < MARGIN:
        return None
    for n in range(max(n_now + 1, 2), N_MAX + 1):
        if mean - T975[n - 1] * sd / n ** 0.5 >= MARGIN:
            return n
    return None


def stats(name, mode, ds, seeds):
    vals = {s: v for s in seeds if (v := value(name, mode, ds, s)) is not None and reference(ds, s, row=name)}
    if not vals:
        return None
    ss = sorted(vals)
    fc = [vals[s] / reference(ds, s, row=name) * 100 for s in ss]
    fo = [vals[s] / reference(ds, s, "oracle", name) * 100 for s in ss if reference(ds, s, "oracle", name)]
    n = len(fc)
    sd = statistics.stdev(fc) if n > 1 else float("nan")
    lo = statistics.mean(fc) - T975[n - 1] * sd / n ** 0.5 if n > 1 else float("nan")
    # +cal rows: vs the same runs uncalibrated; other primary-protocol rows: vs the primary study; v2 rows: none
    base = (split_row(name)[0] if split_row(name)[1] else
            (None if v2_protocol(name) else "primary"))
    prim = {s: value(base, mode, ds, s) if base else None for s in ss}
    paired = [vals[s] - prim[s] for s in ss if prim[s] is not None]
    return {"seeds": ss, "map50": [vals[s] for s in ss], "n": n, "complete": n == len(seeds),
            "fc": statistics.mean(fc), "sd": sd, "lo": lo, "fo": statistics.mean(fo) if fo else float("nan"),
            "pass": n >= 3 and lo >= MARGIN, "vs": statistics.mean(paired) if paired else None}


def eligible(name, mode):
    """(True, '') if the row can win: sync, or async with a node order that varies."""
    src, suf = split_row(name)
    wd = VAR / "_withdrawn.txt"
    if wd.exists() and name in wd.read_text().split():
        return False, "withdrawn: failed its confirmation on fresh seeds (FOLLOWUP_PLAN §8.8)"
    if ORDER_CHECK.match(src):
        return False, "node-order check, not a candidate"
    if mode == "async":
        text = variant_cfg_text(src)
        if "async_clock: virtual" in text and suf == "+cal":   # user decision 2026-10-04
            return False, "calibration needs a closing synchronous round: reference only (see +acal)"
        if "async_clock: virtual" not in text and "async_order: shuffle" not in text:
            return False, "fixed node order: cannot win (async must not depend on the order)"
    return True, ""


def scheme(name, mode):
    """Which winner a row competes for. Sequential runs with server-scheduled turns (async engine,
    async_order: shuffle, one node at a time) count as SYNC (user decision 2026-10-08): the server
    coordinates the turns, every cycle needs all nodes, and the output model merges every node's
    last version. Virtual-clock runs (nodes concurrent, order from timing) are ASYNC."""
    if mode != "async":
        return mode
    text = variant_cfg_text(split_row(name)[0])
    return "sync" if "async_order: shuffle" in text and "async_clock: virtual" not in text else "async"


def winners():
    """{"sync"/"async": [row names]} of eligible rows passing on all datasets with the same, complete
    n >= 3 (grouped by scheme(): sequential turn-based rows count as sync)."""
    out = {}
    for name, mode, seeds in candidates():   # any protocol: a test already scheduled may still produce a winner
        if name.startswith(("confirm_", BASELINE_PREFIX)) or not eligible(name, mode)[0]:   # confirmations, baselines: not candidates
            continue
        st = [stats(name, mode, ds, seeds) for ds in DATASETS]
        if all(x and x["complete"] and x["pass"] for x in st) and len({x["n"] for x in st}) == 1:
            out.setdefault(scheme(name, mode), []).append(name)
    return out


def now_running():
    """One line on the run in progress (tools/queue_status.py), or None."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import queue_status as q
        cur = q.current_run()
        if not cur:
            return None
        name, ds, s, _ = cur
        return name, ds, s, q.phase(name, ds, s)
    except Exception:
        return None


def note(name, mode, ds, st, log, running):
    parts = []
    src_, suf_ = split_row(name)
    if running and running[0] == (("calib_" + src_) if suf_ else name) and running[1] == ds:
        parts.append(f"<< RUNNING NOW: s{running[2]}, {running[3]}")
    ok, why = eligible(name, mode)
    decided = [e["reason"] for k in ("stage0", "stage1", "stage2") if (e := log.get(f"{name}|{mode}|{k}"))]
    if decided and ds == DATASETS[0]:
        parts += decided
    if ok and mode == "async" and scheme(name, mode) == "sync" and ds == DATASETS[0]:
        parts.append("sequential, server-scheduled turns: competes as SYNC")
    if name.startswith(BASELINE_PREFIX):
        return " | ".join(parts + (["standard competitor under protocol v2 (seeds of the search): reference only"]
                                   if ds == DATASETS[0] else []))
    if name.startswith(CONFIRM_PREFIX) and ds == DATASETS[0]:
        parts.append("pre-registered confirmation (analysis/CONFIRMATORY_PREREG.md): fresh seeds, tested once at the "
                     "fixed n by tools/confirm_analysis.py" + ("" if name in CONFIRM_ROWS else "; secondary row"))
    elif name.startswith("confirm_") and ds == DATASETS[0]:
        parts.append("confirmation of the async winner on seeds never used for selection; tested once at this n")
    elif not ok:
        if ds == DATASETS[0]:
            parts.append(why)
    elif st and st["n"] >= 3 and not st["pass"] and not decided:
        nr = n_required(st["fc"], st["sd"], st["n"])
        parts.append(f"mean below {MARGIN:.0f}%: more seeds cannot make it pass" if st["fc"] < MARGIN else
                     f"CI too wide: projected to pass at n={nr}" if nr else
                     f"CI too wide: would need more than {N_MAX} seeds")
    return " | ".join(parts)


def main():
    log, running = search_log(), now_running()
    out = [f"Fed:Cent / Fed:Oracle -- {time.strftime('%Y-%m-%d %H:%M')} -- denominators = centralized/oracle of the same seed"]
    out.append(f">> NOW RUNNING: {running[0]} {running[1]} s{running[2]} ({running[3]})" if running
               else ">> NOW RUNNING: no run in progress")
    for mode, names in winners().items():
        out.append(f">> {mode.upper()} WINNER: {', '.join(names)}")
    try:
        import confirm_analysis as ca   # pre-registered confirmation: its own verdict, never the search's
        out += [f">> {line}" for line in ca.headline()]
    except Exception as e:
        out.append(f">> CONFIRMATION: status not available ({e})")
    for name, mode, seeds in candidates():
        if name.startswith("confirm_") and not name.startswith(CONFIRM_PREFIX) and name.endswith("+cal"):
            st = [stats(name, mode, ds, seeds) for ds in DATASETS]
            done = all(x and x["complete"] for x in st)
            res = (("PASS" if all(x["pass"] for x in st) else "FAIL") if done else
                   f"in progress ({sum(x['n'] for x in st if x)}/{len(seeds) * len(DATASETS)} runs calibrated)")
            detail = "; ".join(f"{ds} {x['fc']:.1f}% (CI-low {x['lo']:.1f}%)" for ds, x in zip(DATASETS, st)
                               if x and x["n"] >= 2)
            out.append(f">> {mode.upper()} WINNER CONFIRMATION on fresh seeds {seeds[0]}-{seeds[-1]}: {res}"
                       + (f" -- {detail}" if detail else ""))
    out += ["", f"{'dataset':7s} {'method':18s} {'n':>2s}  {'mAP50 per seed':34s} {'mean':>6s} "
                f"{'vs base':>10s} {'Fed:Cent':>8s} {'CI-low':>7s} {'test':>4s} {'@90':>4s} {'Fed:Oracle':>10s}  note"]
    rows = candidates()
    conf = [r for r in rows if r[0].startswith(CONFIRM_PREFIX)]
    v2 = [r for r in rows if r[0].startswith((SEARCH_PREFIX, BASELINE_PREFIX))]
    blocks = ([("PRE-REGISTERED CONFIRMATION (analysis/CONFIRMATORY_PREREG.md): frozen configs, fresh seeds, protocol v2",
                "confirm")] if conf else []) + \
             ([("NEW PROTOCOL v2: LR decay to 1%, plateau budget kfm12 30 / WBC 20 / ChestX 56 epochs "
                "-- the search runs here", "v2")] if v2 else []) + \
             [("PRIMARY PROTOCOL: 20 epochs, constant LR (for reference)", "primary")]
    for title, block in blocks:
      is_v2 = block != "primary"
      group = conf if block == "confirm" else v2
      out += ["", "#" * 118, f"# {title}", "#" * 118]
      for sect in ("sync", "async"):
        out += ["", f"==================== {sect.upper()} " + "=" * 80]
        for ds in DATASETS:
            out.append("-" * 118)
            refs = ([("centralized v2", "centralized", sorted({s for r in group for s in r[2]}), "v2_x")] if is_v2 else
                    [("centralized", "centralized", PRIMARY_SEEDS, ""), ("oracle", "oracle", PRIMARY_SEEDS, "")])
            for label, ref, rseeds, rowkey in refs:
                rv = [v for s in rseeds if (v := reference(ds, s, ref, rowkey)) is not None]
                if rv:
                    out.append(f"{ds:7s} {'[' + label + ']':18s} {len(rv):2d}  {' '.join(f'{x:.4f}' for x in rv):34s} "
                               f"{statistics.mean(rv):6.3f}   (reference{', v2_central' if is_v2 else ', primary study seeds 0-2'})")
            cseeds = sorted({s for r in group for s in r[2]}) if is_v2 else PRIMARY_SEEDS
            cv = [v for s in cseeds if (v := reference(ds, s, "centralized", ("v2_x" if is_v2 else "x") + "+cal")) is not None]
            if cv:
                out.append(f"{ds:7s} {'[centralized+cal]':18s} {len(cv):2d}  {' '.join(f'{x:.4f}' for x in cv):34s} "
                           f"{statistics.mean(cv):6.3f}   (reference of the +cal rows: same calibration applied to centralized)")
            for name, mode, seeds in rows:
                if mode != sect or {"confirm": name.startswith(CONFIRM_PREFIX),
                                    "v2": name.startswith((SEARCH_PREFIX, BASELINE_PREFIX)),
                                    "primary": not v2_protocol(name)}[block] is False:
                    continue
                st = stats(name, mode, ds, seeds)
                nt = note(name, mode, ds, st, log, running)
                if not st:
                    out.append(f"{ds:7s} {name:18s}  0  (no finished run yet)" + (f"  {nt}" if nt else ""))
                    continue
                per = " ".join(f"{x:.3f}" for x in st["map50"])
                vs = "" if name == "primary" or st["vs"] is None else f"{st['vs']:+.3f}"
                lo = "" if st["lo"] != st["lo"] else f"{st['lo']:6.1f}%"
                test = "-" if st["n"] < 3 else ("PASS" if st["pass"] else "fail")
                t90 = "-" if st["n"] < 3 else ("PASS" if st["lo"] >= PRE_MARGIN else "fail")
                fo = "" if st["fo"] != st["fo"] else f"{st['fo']:9.1f}%"
                out.append(f"{ds:7s} {name:18s} {st['n']:2d}  {per:34s} {statistics.mean(st['map50']):6.3f} {vs:>10s} "
                           f"{st['fc']:7.1f}% {lo:>7s} {test:>4s} {t90:>4s} {fo:>10s}  {nt}")
    pil = [d for ds in DATASETS if (d := load(VAR / f"_budget/{ds}/s0/curve.json")) and d.get("curve")]
    if pil:
        out += ["", "==================== TRAINING-BUDGET PILOT (centralized, seed 0; FOLLOWUP_PLAN §8.4) " + "=" * 30]
        for d in pil:
            best = max(d["curve"], key=lambda c: c["map50"])
            at20 = next((c["map50"] for c in d["curve"] if c["epoch"] == 20), None)
            state = (f"budget {d['budget_epochs']} epochs" if "budget_epochs" in d
                     else f"running, epoch {d['curve'][-1]['epoch']}/{d['max_epochs']}")
            out.append(f"{d['dataset']:7s} best mAP50 {best['map50']:.4f} at epoch {best['epoch']:3d} | at 20 epochs "
                       f"{'-' if at20 is None else f'{at20:.4f}'} | {state} | curve: "
                       + " ".join(f"{c['epoch']}:{c['map50']:.3f}" for c in d["curve"]))
    out += ["", f"n = seeds finished. test = main criterion CI-low >= {MARGIN:.0f}% (user decision 2026-10-08, post hoc), "
            f"needs n >= 3; pre-registered: {PRE_MARGIN:.0f}%.",
            "<name>+cal = that run's final model after the final calibration round (cofy/calibrate.py), applied post hoc;",
            "  its Fed:Cent uses the centralized baseline given the same calibration (same post-processing on both sides).",
            "<name>+acal = asynchronous progressive calibration (virtual-clock runs): nobody waits, the released model improves",
            "  after each node's contribution; shown after all nodes contributed (partial states: runs/variants/calib/<name>/).",
            "<name>+acalr = the same, then each node re-fits its rows once more against the complete BatchNorm pool.",
            "Winner = first sync / async method passing on all three datasets with the same n; async only with a varying node order.",
            "vs base = paired mAP50 difference: <name>+cal rows vs the same runs uncalibrated; other rows vs the primary study.",
            "Extra seeds are decided once, after seeds 0-2, from the projected CI (note column); never added again after that."]
    text = "\n".join(out)
    print(text)
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(text + "\n")


if __name__ == "__main__":
    main()
