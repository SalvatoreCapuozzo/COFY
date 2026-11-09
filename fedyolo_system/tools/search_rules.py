#!/usr/bin/env python3
"""Decisions the variant queue takes on its own (runs/variants/_queue.sh calls this
after every plan entry). Rows and statistics come from tools/results_table.py.

1. Winner: the first sync / async row that passes the pre-registered test
   (CI-low of Fed:Cent >= 90%) on all three datasets with the same complete n.
   Async rows qualify only with a varying node order. Written to
   runs/variants/_WINNERS.txt; the queue then skips the remaining entries of
   that mode.
2. Promotion: a seed-0 screening row with Fed:Cent >= 95% on all three datasets
   gets seeds 1-2.
3. Extra seeds: a row with seeds 0-2 done, mean Fed:Cent >= 90% everywhere but
   CI-low < 90% somewhere, gets the seeds its projected CI needs (same mean and
   sd, at most N_MAX), the same count on every dataset. Centralized baselines
   for seeds the primary study never ran are queued first (tools/run_baseline.py).

Only rows of the current search protocol (results_table.SEARCH_PREFIX, "v2_") are
considered; the 20-epoch rows stay in the table for reference.

Each decision is taken once per row and stage and kept, with its reason, in
runs/variants/_search_log.json (shown in the note column of the table): seeds are
fixed from the first three, then the test is done once at that n.
"""
import json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_table as rt

ROOT, VAR = rt.ROOT, rt.VAR
PLAN, LOG, WIN = VAR / "_plan.txt", VAR / "_search_log.json", VAR / "_WINNERS.txt"
PROMOTE = rt.MARGIN + 5   # seed-0 screening bar (95 with the pre-registered 90, now 90)


def say(msg):
    print(f"{time.strftime('%m-%d %H:%M')} [search] {msg}", flush=True)


def set_seeds(lines, name, mode, seeds, before=None, after=None):
    """Make plan line `name` hold at least `seeds` (create it before line `before` / after line `after`
    / at the end, if missing)."""
    for i, line in enumerate(lines):
        p = line.split("#", 1)[0].split()
        if p and p[0] == name:
            have = [int(s) for s in p[2:]]
            lines[i] = f"{name:18s} {p[1]:28s} {' '.join(map(str, sorted(set(have) | set(seeds))))}"
            return
    new = f"{name:18s} {mode:28s} {' '.join(map(str, sorted(seeds)))}"
    first = lambda n: next((i for i, l in enumerate(lines) if l.split("#", 1)[0].split()[:1] == [n]), None)
    idx = first(before) if before else (first(after) + 1 if after and first(after) is not None else None)
    lines.insert(len(lines) if idx is None else idx, new)


def write_configs(src, seeds):
    import yaml
    for ds in rt.DATASETS:
        for s in seeds:
            if src == "primary_ext":
                base, out = ROOT / f"configs/study/{ds}_s0.yaml", ROOT / f"configs/variants/primary_ext/{ds}_s{s}.yaml"
            else:
                base, out = ROOT / f"configs/variants/{src}/{ds}_s0.yaml", ROOT / f"configs/variants/{src}/{ds}_s{s}.yaml"
            if out.exists():
                continue
            c = yaml.safe_load(base.read_text())
            c["seed"] = s
            if src == "primary_ext":
                c["output_dir"] = f"runs/variants/primary_ext/{ds}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(c, sort_keys=False))


def extend(lines, row, mode, seeds):
    """Queue `seeds` for the variant behind `row` (and its calibration line), plus missing baselines."""
    src = rt.split_row(row)[0]
    plan = rt.read_plan()
    if src == "primary":
        src, extra = "primary_ext", [s for s in seeds if s not in rt.PRIMARY_SEEDS]
        set_seeds(lines, "primary_ext", mode, extra, before=next((n for n, *_ in plan), None))
        set_seeds(lines, "calib_primary_ext", f"posthoc:primary_ext:{mode}", extra, after="primary_ext")
        seeds_for_cfg, anchor = extra, "primary_ext"
    else:
        set_seeds(lines, src, mode, seeds)
        for n, m, _ in plan:
            if m.startswith(f"posthoc:{src}:"):
                set_seeds(lines, n, m, seeds)
        seeds_for_cfg, anchor = seeds, src
    write_configs(src, seeds_for_cfg)
    v2 = row.startswith(rt.SEARCH_PREFIX)
    plain = rt.split_row(row)[0]
    missing = sorted({s for s in seeds for ds in rt.DATASETS if rt.reference(ds, s, row=plain) is None})
    if missing:   # centralized reference of the same protocol, trained before the variant's new seeds
        bname, bmode = ("v2_central", "baseline:v2") if v2 else ("baseline_ext", "baseline")
        old = [s for n, m, ss in plan if n == bname for s in ss]
        lines[:] = [l for l in lines if l.split("#", 1)[0].split()[:1] != [bname]]
        set_seeds(lines, bname, bmode, sorted(set(missing) | set(old)), before=anchor)
    if rt.split_row(row)[1]:   # same post-processing rule: the calibrated centralized reference too
        miss_cal = sorted({s for s in seeds for ds in rt.DATASETS if rt.reference(ds, s, row=row) is None})
        if miss_cal:
            cname, cmode = (("calib_v2_central", "posthoc:v2_centralized:-") if v2
                            else ("calib_central", "posthoc:centralized:-"))
            old = [s for n, m, ss in plan if n == cname for s in ss]
            lines[:] = [l for l in lines if l.split("#", 1)[0].split()[:1] != [cname]]
            set_seeds(lines, cname, cmode, sorted(set(miss_cal) | set(old)), before=anchor)


def lift(lines, src):
    """Drop the '@datasets' restriction from the plan lines of variant `src` and of its calibration."""
    for i, line in enumerate(lines):
        p = line.split("#", 1)[0].split()
        if p and "@" in p[1] and (p[0] == src or p[1].startswith(f"posthoc:{src}:")):
            lines[i] = f"{p[0]:18s} {p[1].split('@', 1)[0]:28s} {' '.join(p[2:])}"


def main():
    log = rt.load(LOG) or {}
    lines = PLAN.read_text().splitlines()
    before = list(lines)

    found = rt.winners()
    known = WIN.read_text() if WIN.exists() else ""
    for mode, names in found.items():
        if re.search(rf"\b{mode.upper()} WINNER", known):   # \b: "ASYNC WINNER" is not a sync winner
            continue
        name = names[0]
        rmode, rseeds = next((r[1], r[2]) for r in rt.candidates() if r[0] == name)
        detail = "; ".join(f"{ds} Fed:Cent {st['fc']:.1f}% (CI-low {st['lo']:.1f}%, n={st['n']})"
                           for ds in rt.DATASETS if (st := rt.stats(name, rmode, ds, rseeds)))
        if rmode != mode:
            detail += " -- sequential scheme (server-scheduled turns), counted as sync"
        msg = f"{mode.upper()} WINNER: {name} -- {detail}"
        with open(WIN, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {msg}\n")
        say(msg + f" -- remaining {mode} entries will be skipped")

    import queue_status as qs
    dsets = qs.plan_datasets()
    for row, mode, seeds in rt.candidates():
        if (not row.startswith(rt.SEARCH_PREFIX) or rt.scheme(row, mode) in found
                or not rt.eligible(row, mode)[0] or not seeds):
            continue
        src = rt.split_row(row)[0]
        only = dsets.get(src, rt.DATASETS)
        if set(only) != set(rt.DATASETS):
            # stage 0: screen restricted to some datasets (ChestX first); >= 95% there on seed 0
            # -> the restriction is lifted for the variant and its calibration lines
            sx = {ds: rt.stats(row, mode, ds, [0]) for ds in only}
            key = f"{row}|{mode}|stage0"
            if key in log or not all(x and x["complete"] for x in sx.values()):
                continue
            vals = ", ".join(f"{ds} {x['fc']:.1f}%" for ds, x in sx.items())
            ok = all(x["fc"] >= PROMOTE for x in sx.values())
            reason = (f"seed 0 >= {PROMOTE:.0f}% on {', '.join(only)} ({vals}): other datasets added" if ok else
                      f"screen stopped at seed 0 on {', '.join(only)} (needs >= {PROMOTE:.0f}%: {vals})")
            log[key] = {"reason": reason, "at": time.strftime("%Y-%m-%d %H:%M")}
            if ok:
                lift(lines, src)
            say(f"{row} ({mode}): {reason}")
            continue
        st = {ds: rt.stats(row, mode, ds, seeds) for ds in rt.DATASETS}
        if not all(x and x["complete"] for x in st.values()):
            continue
        fc = {ds: x["fc"] for ds, x in st.items()}
        if seeds == [0]:
            key = f"{row}|{mode}|stage1"
            if key in log:
                continue
            ok = all(v >= PROMOTE for v in fc.values())
            vals = ", ".join(f"{ds} {v:.1f}%" for ds, v in fc.items())
            reason = (f"seed 0 >= {PROMOTE:.0f}% on all datasets ({vals}): seeds 1-2 added" if ok else
                      f"screen stopped at seed 0 (needs >= {PROMOTE:.0f}% everywhere: {vals})")
            log[key] = {"reason": reason, "seeds": [0, 1, 2] if ok else [0], "at": time.strftime("%Y-%m-%d %H:%M")}
            if ok:
                extend(lines, row, mode, [0, 1, 2])
            say(f"{row} ({mode}): {reason}")
            continue
        key = f"{row}|{mode}|stage2"
        if key in log or len(seeds) < 3 or all(x["pass"] for x in st.values()):
            continue
        n_now = min(x["n"] for x in st.values())
        if any(v < rt.MARGIN for v in fc.values()):
            reason = f"mean below {rt.MARGIN:.0f}% on " + ", ".join(ds for ds, v in fc.items() if v < rt.MARGIN) + ": no extra seeds"
            n_new = None
        else:
            need = {ds: rt.n_required(x["fc"], x["sd"], x["n"]) if not x["pass"] else x["n"] for ds, x in st.items()}
            worst = min(st, key=lambda ds: st[ds]["lo"])
            w = st[worst]
            if any(v is None for v in need.values()):
                n_new = None
                reason = (f"CI-low {w['lo']:.1f}% on {worst} at n={w['n']} (mean {w['fc']:.1f}%, sd {w['sd']:.1f}): "
                          f"would need more than {rt.N_MAX} seeds: no extra seeds")
            else:
                n_new = max(need.values())
                reason = (f"CI-low {w['lo']:.1f}% on {worst} at n={w['n']} (mean {w['fc']:.1f}%, sd {w['sd']:.1f}); "
                          f"projected >= {rt.MARGIN:.0f}% on all datasets at n={n_new}: seeds {n_now}-{n_new - 1} added")
        log[key] = {"reason": reason, "seeds": list(range(n_new)) if n_new else seeds,
                    "at": time.strftime("%Y-%m-%d %H:%M")}
        if n_new:
            extend(lines, row, mode, list(range(n_new)))
        say(f"{row} ({mode}): {reason}")

    LOG.write_text(json.dumps(log, indent=1))
    if lines != before:
        PLAN.write_text("\n".join(lines) + "\n")
        say("plan updated")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:   # gate of the ChestX fix screens (analysis/CHESTX_CONCURRENT.md)
            import cx_gate
            cx_gate.main()
        except Exception as e:
            print(f"[cx_gate] error: {e}", flush=True)
