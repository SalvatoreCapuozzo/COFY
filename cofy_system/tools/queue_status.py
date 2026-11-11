#!/usr/bin/env python3
"""Where is the variant queue right now?

    python tools/queue_status.py          # full view: job in progress, sections, last completed, next
    python tools/queue_status.py --watch  # the same, live (redrawn every 2 s)
    python tools/queue_status.py --done   # every finished run, newest first
    python tools/queue_status.py --line   # one line (the queue appends this to _queue.log every 10 min)

Reads runs/variants/_plan.txt (the queue's own run list), each run's
summary.json (done or not), the live cofy.simulate process (which run is
training), the tail of that run's log (which round), and the durations of
finished runs (for the time estimate).
"""
import json, re, statistics, subprocess, sys, time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cofy.proc import pgrep_lines   # `pgrep -af` output on Linux and macOS
PLAN = ROOT / "runs/variants/_plan.txt"
DATASETS = ["kfm12", "wbc", "chestx"]
KEYS = {"sync": "federated_sync", "async": "federated_async"}
FALLBACK_MIN = {"kfm12": 41, "wbc": 25, "chestx": 51}
V2_FACTOR = {"kfm12": 30 / 20, "wbc": 20 / 20, "chestx": 56 / 20}   # v2 budgets vs the 20-epoch runs


def split_mode(mode):
    """A plan mode may restrict the datasets: 'sync@chestx' -> ('sync', ['chestx']);
    without '@' -> (mode, all datasets). Used for ChestX-first screens."""
    if "@" in mode:
        base, ds = mode.split("@", 1)
        return base, [d for d in ds.split(",") if d]
    return mode, list(DATASETS)


def read_plan():
    """[(name, mode without the '@' restriction, [seeds])]."""
    plan = []
    for line in PLAN.read_text().splitlines():
        line = line.split("#", 1)[0].split()
        if line:
            plan.append((line[0], split_mode(line[1])[0], [int(s) for s in line[2:]]))
    return plan


def plan_datasets():
    """{plan entry name: datasets it runs on}."""
    out = {}
    for line in PLAN.read_text().splitlines():
        line = line.split("#", 1)[0].split()
        if line:
            out[line[0]] = split_mode(line[1])[1]
    return out


def plan_lines():
    """[(name, mode, datasets, seeds, section)]: every plan line in order, mode without its '@'
    restriction, the datasets that line runs on, and its section (the last '# ---- TITLE' above it;
    held-out lines and robustness smoke tests are filed under their own study)."""
    out, section = [], "Exploratory search"
    for raw in PLAN.read_text().splitlines():
        s = raw.strip()
        if s.startswith("# ----"):
            title = s.lstrip("# -").split(" (")[0].split(":")[0].strip()
            if title[:1].isupper():
                section = title
            continue
        p = raw.split("#", 1)[0].split()
        if p:
            mode, dsets = split_mode(p[1])
            sec = ("HELD-OUT DENTEX confirmation" if dsets == ["dentex"] else
                   "ROBUSTNESS STUDY" if p[0].startswith("rbsmoke_") else section)
            out.append((p[0], mode, dsets, [int(x) for x in p[2:]], sec))
    return out


def all_runs(plan=None):
    """[(name, mode, dataset, seed)] of every plan line (per line: a name may have lines on other datasets)."""
    return [(name, mode, ds, s) for name, mode, dsets, seeds, _ in plan_lines() for s in seeds for ds in dsets]


def posthoc(mode):
    """'posthoc:<source>:<modes,>' plan entries (tools/calibrate_checkpoints.py) -> (source, [modes]) or None."""
    if not mode.startswith("posthoc:"):
        return None
    _, src, modes = mode.split(":")
    return src, (["centralized"] if src in ("centralized", "v2_centralized") else modes.split(","))


def calib_stages(src):
    """Stages tools/calibrate_checkpoints.py computes for `src`: the calibration round (before,
    bn, head, bn+head); for virtual-clock async sources: before, the synchronous round (bn+head),
    and the asynchronous progressive calibration after 1..N-1 node contributions (acal_k) and all
    N (acal), see cofy/calibrate.py calibrate_progressive."""
    from results_table import variant_cfg_text
    text = variant_cfg_text(src)
    if "async_clock: virtual" not in text:
        return ["before", "bn", "head", "bn+head"]
    import yaml
    n = len(yaml.safe_load(text)["nodes"])
    return (["before", "bn+head"] + [f"acal_{k}" for k in range(1, n)] + ["acal"]
            + (["acal_refresh"] if "progressive_refresh: true" in text else []))


def is_done(name, mode, ds, s):
    if mode == "budget":     # tools/budget_pilot.py (seed s, every dataset)
        return "budget_epochs" in (load_json(ROOT / f"runs/variants/_budget/{ds}/s{s}/curve.json") or {})
    if mode in ("baseline:swa", "baseline:local", "baseline:budgeteval", "baseline:bnpool", "baseline:cleanse"):   # tools/followup_runs.py
        return bool((load_json(ROOT / f"runs/variants/{name}/{ds}/s{s}/summary.json") or {}).get("complete"))
    if mode.startswith("baseline"):   # tools/run_baseline.py: centralized for a seed / protocol
        paths = ([ROOT / f"runs/variants/v2_central/{ds}/s{s}/summary.json"] if mode == "baseline:v2" else
                 [ROOT / f"runs/study/{ds}/s{s}/summary.json", ROOT / f"runs/variants/_baseline/{ds}/s{s}/summary.json"])
        for p in paths:
            if "map50" in ((load_json(p) or {}).get("centralized") or {}):
                return True
        return False
    if posthoc(mode):
        src, modes = posthoc(mode)
        for m in modes:
            d = load_json(ROOT / f"runs/variants/calib/{src}/{ds}/s{s}/{m}.json") or {}
            if not all(k in d for k in calib_stages(src)):
                return False
        return True
    if name == "primary_ext" and "map50" in ((load_json(ROOT / f"runs/study/{ds}/s{s}/summary.json") or {}).get(KEYS[mode]) or {}):
        return True   # extra seeds of the primary study that it already ran (kfm12 s3)
    try:
        d = json.loads((ROOT / f"runs/variants/{name}/{ds}/s{s}/summary.json").read_text())
        return "map50" in d.get(KEYS[mode], {})
    except (OSError, json.JSONDecodeError):
        return False


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def durations():
    """(variant, dataset) -> list of finished-run minutes, from every _runner.log."""
    out = {}
    for log in (ROOT / "runs/variants").glob("*/_runner.log"):
        for m in re.finditer(r"DONE\s+(\w+) s\d+ in (\d+) min", log.read_text()):
            out.setdefault((log.parent.name, m.group(1)), []).append(int(m.group(2)))
        for m in re.finditer(r"DONE\s+\S+ (\w+) s\d+ \w+ in (\d+) min", log.read_text()):   # post-hoc, per mode
            out.setdefault((log.parent.name, m.group(1)), []).append(int(m.group(2)))
    return out


ALIAS = {"confirm_v2sync": "v2_async", "confirm_v2async": "v2_vclock",   # frozen copies: same durations
         "calib_confirm_v2sync": "calib_v2_async", "calib_confirm_v2async": "calib_v2_vclock",
         "calib_confirm_central": "calib_v2_central"}


def typical(name, ds, durs, mode=""):
    if not durs.get((name, ds)) and ALIAS.get(name) in {n for n, _ in durs}:
        name = ALIAS[name]
    if mode == "budget":
        return 4.5 * FALLBACK_MIN.get(ds, 50)
    if durs.get((name, ds)) and not posthoc(mode):
        return statistics.median(durs[(name, ds)])
    if mode.startswith("baseline") or name.startswith("v2_"):
        same_ds = [x for (n, d), v in durs.items() if d == ds and not n.startswith("v2_") for x in v]
        t = statistics.median(same_ds) if same_ds else FALLBACK_MIN.get(ds, 50)
        return t * (V2_FACTOR.get(ds, 1.5) if mode == "baseline:v2" or name.startswith("v2_") else 1)
    if posthoc(mode):
        per_mode = statistics.median(durs[(name, ds)]) if durs.get((name, ds)) else 6
        return per_mode * len(posthoc(mode)[1])
    if durs.get((name, ds)):
        return statistics.median(durs[(name, ds)])
    same_ds = [x for (n, d), v in durs.items() if d == ds for x in v]
    return statistics.median(same_ds) if same_ds else FALLBACK_MIN.get(ds, 50)


RUNNER = re.compile(r"tools/(run_variant|run_baseline|budget_pilot|calibrate_checkpoints)[.]py")
WHAT = {   # name prefix -> what the job does (first match wins)
    "confirm_central": "centralized reference, protocol v2: one model trained on every node's data pooled "
                       "(the Fed:Cent denominator of this seed)",
    "calib_confirm_central": "calibration of the centralized reference (same post-processing as the federated runs)",
    "confirm_v2sync": "federated SYNC scheme, pre-registered confirmation: nodes train in server-scheduled turns "
                      "(random order each cycle); final model = average of the last 4 versions",
    "calib_confirm_v2sync": "closing calibration of the sync runs: BatchNorm statistics pooled, each node re-fits "
                            "its own class rows",
    "confirm_v2async": "federated ASYNC scheme, pre-registered confirmation: nodes train concurrently on a virtual "
                       "clock, each upload merged on arrival with its staleness weight",
    "calib_confirm_v2async": "progressive calibration of the async runs (no barrier: the model improves after "
                             "each node's contribution)",
    "confirm_pilot": "training-length pilot: centralized up to 100 epochs, evaluated every 5, stops at the "
                     "plateau; fixes the held-out dataset's budget",
    "rbsmoke_": "1-round smoke test of the attack / defence code",
    "rb_": "robustness study: one malicious node (node seed mod 4)",
    "bl_": "standard competitor under protocol v2",
    "calib_bl_": "calibration of a competitor's runs (same post-processing)",
}


FOLLOWUP_WHAT = {   # tools/followup_runs.py protocols (they print '⏱ <entry> <ds> s<seed> <unit> k/n done' lines)
    "baseline:swa": "centralized reference with weight averaging (follow-up)",
    "baseline:local": "local-only baseline: each node trains alone (follow-up)",
    "baseline:budgeteval": "per-class detection budget on clean runs (follow-up)",
    "baseline:bnpool": "candidate defence P1: median pooling of BatchNorm statistics",
    "baseline:cleanse": "candidate defence P2: input screening at the nodes (Detector Cleanse)",
}


def describe(name, mode):
    """What a plan entry does, in one sentence."""
    for k in sorted(WHAT, key=len, reverse=True):
        if name.startswith(k):
            text = WHAT[k]
            break
    else:
        text = FOLLOWUP_WHAT.get(mode) or (
               "centralized reference" if mode.startswith("baseline") else "training-length pilot" if mode == "budget"
                else "post-hoc calibration" if posthoc(mode) else f"federated {mode} training (exploratory variant)")
    m = re.match(r"rb_(seq|conc|par)_([a-z]+)(_ora)?(_noenf)?(_p3x|_p3|_p4)?(_budget)?$", name)
    if m:
        text += (f", {dict(seq='sequential sync', conc='concurrent async', par='parallel FedAvg')[m[1]]}, "
                 f"attack '{m[2]}', defence {'ON' if m[3] else 'off'}" + (", ownership not enforced" if m[4] else "")
                 + {"_p3": " + candidate P3 (peer validation, per-class rollback)",
                    "_p3x": " + candidate P3b (peer validation, rollback and exclusion of the flagged node)",
                    "_p4": " + candidate P4 (consecutive merges bounded jointly)"}.get(m[5] or "", "")
                 + (", final model kept" if m[6] else ""))
    return text


def _queue_entry():
    """(name, mode, [seeds], started 'MM-DD HH:MM') of the last entry the queue launched (runs/variants/_queue.log)."""
    log = ROOT / "runs/variants/_queue.log"
    try:
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 300_000))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    ent = re.findall(r"^(\d\d-\d\d \d\d:\d\d) \[queue\] variant (\S+) \(mode (\S+), seeds ([\d ]*)\)", text, re.M)
    return (ent[-1][1], ent[-1][2], [int(x) for x in ent[-1][3].split()], ent[-1][0]) if ent else None


def _tail(path, n=3_000_000):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - n))
            return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", f.read().decode("utf-8", "replace"))
    except OSError:
        return ""


def _stamp(s):
    """'MM-DD HH:MM' or 'YYYY-MM-DD HH:MM:SS' -> datetime (this year)."""
    return (datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S") if len(s) >= 19 and s[4] == "-"
            else datetime.strptime(f"{datetime.now().year}-{s}", "%Y-%m-%d %H:%M"))


def current_job():
    """The job the queue runs now, in detail, or None: its plan entry (from the queue's own log), the
    runner process, dataset and seed (last START of the entry's runner log), what it does, its
    progress in its own unit, timing, and a health check (is output still coming?)."""
    out = pgrep_lines(RUNNER.pattern)
    procs = [l for l in out.splitlines() if RUNNER.search(l) and "python" in l.split()[1]]
    if not procs:
        return None
    pid, cmd = procs[0].split(None, 1)
    ent = _queue_entry()
    if ent and (ent[0] in cmd or RUNNER.search(cmd)):
        name, mode, seeds, qstart = ent
    else:   # the queue's line for this entry is missing (log overwritten by a relaunch) or a runner started by hand:
        # read the entry from the runner's own arguments
        m = re.search(r"--name (\S+)", cmd)
        md = re.search(r"--mode (\S+)", cmd)
        pr = re.search(r"tools/run_baseline[.]py(?: --protocol (\S+))?", cmd)
        sd = re.search(r"--seeds ((?:\d+ ?)+)", cmd)
        seeds = [int(x) for x in sd[1].split()] if sd else []
        mode = md[1] if md else (f"baseline:{pr[1]}" if pr and pr[1] else "baseline" if pr else "?")
        name = m[1] if m else next((n for n, md_, _ in read_plan() if md_ == mode), RUNNER.search(cmd)[1])
        qstart = None
    runner_log = ROOT / f"runs/variants/{name}/_runner.log"
    text = _tail(runner_log)
    starts = list(re.finditer(r"^(\d\d-\d\d \d\d:\d\d) START\s+(?:(\S+)\s+)?([a-z0-9]+) s(\d+)\b", text, re.M))
    if not starts:
        return {"name": name, "mode": mode, "pid": pid, "ds": None, "seed": None, "what": describe(name, mode),
                "unit": "", "done": 0, "total": 1, "elapsed": 0.0, "left": None, "health": "starting", "ok": True,
                "seeds": seeds, "detail": "setting up"}
    st = starts[-1]
    ds, seed, t0 = st[3], int(st[4]), _stamp(st[1])
    after = text[st.end():]
    now = datetime.now()
    el = (now - t0).total_seconds() / 60
    job = {"name": name, "mode": mode, "pid": pid, "ds": ds, "seed": seed, "seeds": seeds, "started": t0,
           "elapsed": el, "what": describe(name, mode), "detail": ""}
    if mode in FOLLOWUP_WHAT:   # follow-up tools: progress from their own '⏱ ... k/n done' lines
        ticks = re.findall(r"^⏱ \S+ \S+ s\d+ (?:model|node|epoch) (\d+)/(\d+) done", after, re.M)
        stamps = re.findall(r"^(\d\d-\d\d \d\d:\d\d) ", after, re.M)
        done, total = (int(ticks[-1][0]), int(ticks[-1][1])) if ticks else (0, 1)
        unit = re.findall(r"^⏱ \S+ \S+ s\d+ (model|node|epoch) ", after, re.M)
        job.update(unit=unit[-1] if unit else "step", done=done, total=total,
                   last=_stamp(stamps[-1]) if stamps else t0)
        expect = el / done if done else 20
    elif mode.startswith("baseline") or mode == "budget":
        ep = re.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \[INFO\] === centralized epoch (\d+)/(\d+) ===", after)
        done, total = (int(ep[-1][1]) - 1, int(ep[-1][2])) if ep else (0, 1)
        last = _stamp(ep[-1][0]) if ep else t0
        job.update(unit="epoch", done=done, total=total, last=last)
        if mode == "budget":
            c = (load_json(ROOT / f"runs/variants/_budget/{ds}/s{seed}/curve.json") or {}).get("curve") or []
            if c:
                best = max(c, key=lambda x: x["map50"])
                job["detail"] = (f"last evaluation epoch {c[-1]['epoch']}: mAP50 {c[-1]['map50']:.3f} "
                                 f"(best {best['map50']:.3f} at epoch {best['epoch']}); may stop before {total}")
        expect = el / done if done else 10
    elif posthoc(mode):
        src = posthoc(mode)[0]
        stages = re.findall(r"^(\d\d-\d\d \d\d:\d\d)\s+(\S+)\s+mAP50 ([\d.]+)\s*$", after, re.M)
        total = len(calib_stages(src))
        job.update(unit="stage", done=len(stages), total=total, last=_stamp(stages[-1][0]) if stages else t0,
                   detail=(f"last stage {stages[-1][1]}: mAP50 {float(stages[-1][2]):.3f}" if stages else
                           "first stage (re-evaluating the model as saved)"))
        expect = el / len(stages) if stages else 15
    else:
        ph = phase(name, ds, seed)
        m = re.search(r"(async submission|sync round) (\d+)/(\d+)", ph)
        done = (int(m[2]) - (1 if "round" in m[1] else 0)) if m else 0
        log = ROOT / f"runs/variants/{name}/_log_{ds}_s{seed}.log"
        last = datetime.fromtimestamp(log.stat().st_mtime) if log.exists() else t0
        job.update(unit="update" if m and "submission" in m[1] else "round", done=done,
                   total=int(m[3]) if m else 1, last=last, detail=ph)
        expect = 5
    job["left"] = (el * (job["total"] - job["done"]) / job["done"]) if job["done"] else None
    age = (now - job["last"]).total_seconds() / 60
    limit = max(20.0, 3 * expect)
    job["ok"] = age <= limit
    job["health"] = (f"runner pid {pid} alive, last progress {age:.0f} min ago" +
                     ("" if job["ok"] else f" -- WARNING: nothing new for {age:.0f} min (expected every "
                                           f"~{expect:.0f} min): check {runner_log.relative_to(ROOT)}"))
    return job


def gpu_util():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        u, mem, pw = [x.strip() for x in r.splitlines()[0].split(",")]
        return f"GPU {u} %, {float(mem) / 1024:.1f} GB, {float(pw):.0f} W"
    except Exception:
        return "GPU ?"


def completed_runs():
    """Every finished run, newest first: (time, entry, dataset, seed, minutes, result text)."""
    rows = []
    for log in (ROOT / "runs/variants").glob("*/_runner.log"):
        name = log.parent.name
        for m in re.finditer(r"^(\d\d-\d\d \d\d:\d\d) DONE\s+(?:(\S+)\s+)?([a-z0-9]+) s(\d+)(?: (\w+))? in (\d+) min(.*)$",
                             _tail(log, 5_000_000), re.M):
            when, src, ds, s, md, mins, rest = m.groups()
            res = re.search(r"mAP50 ([\d.]+)", rest)
            bud = re.search(r"budget (\d+) epochs", rest)
            if bud:
                result = f"budget {bud[1]} epochs" + (f", best mAP50 {res[1]}" if res else "")
            elif res:
                result = f"mAP50 {float(res[1]):.3f}"
            elif src and md:   # calibration: the scheme's final stage
                d = load_json(ROOT / f"runs/variants/calib/{src}/{ds}/s{s}/{md}.json") or {}
                stage = calib_stages(src)[-1] if src not in ("centralized", "v2_centralized") else "bn+head"
                result = f"{stage} mAP50 {d[stage]['map50']:.3f}" if stage in d else ""
            else:
                d = load_json(ROOT / f"runs/variants/{name}/{ds}/s{s}/summary.json") or {}
                v = next((x["map50"] for x in d.values() if isinstance(x, dict) and "map50" in x), None)
                result = f"mAP50 {v:.3f}" if v is not None else ""
            rows.append((when, name, ds, int(s), int(mins), result))
    return sorted(rows, reverse=True)


def queue_eta(job=None):
    """(runs done, runs total, minutes left, {section: [done, total, minutes left]}) with the current job's
    own estimate for its remaining time."""
    durs, won = durations(), winner_modes()
    secs, n_done, n_tot, left = {}, 0, 0, 0.0
    for name, mode, dsets, seeds, sec in plan_lines():
        for s in seeds:
            for ds in dsets:
                d = skipped(name, mode, won) or is_done(name, mode, ds, s)
                x = secs.setdefault(sec, [0, 0, 0.0])
                x[1] += 1
                n_tot += 1
                if d:
                    x[0] += 1
                    n_done += 1
                    continue
                t = typical(name, ds, durs, mode)
                if job and (name, ds, s) == (job["name"], job["ds"], job["seed"]) and mode == job["mode"]:
                    t = job["left"] if job.get("left") is not None else max(t - job["elapsed"], 1)
                x[2] += t
                left += t
    return n_done, n_tot, left, secs


def upcoming(n=6):
    won, out = winner_modes(), []
    for name, mode, dsets, seeds, sec in plan_lines():
        if skipped(name, mode, won):
            continue
        todo = [(ds, s) for s in seeds for ds in dsets if not is_done(name, mode, ds, s)]
        if todo:
            out.append((name, mode, dsets, seeds, len(todo)))
        if len(out) >= n + 1:
            break
    return out[1:] if out else []   # the first one is the job in progress


def current_run():
    """(variant, dataset, seed, pid) of the job in progress, or None (see current_job)."""
    j = current_job()
    return (j["name"], j["ds"], j["seed"], j["pid"]) if j and j["ds"] else None


def _old_current_run():
    """(variant, dataset, seed, pid) of the training process, or None."""
    out = pgrep_lines("(cofy|fedyolo)[.]simulate")
    for line in out.splitlines():
        m = re.search(r"configs/variants/(\w+)/(\w+)_s(\d+)\.yaml", line)
        if m:
            return m.group(1), m.group(2), int(m.group(3)), line.split()[0]
    for tool, kind in (("budget_pilot", "budget"), ("run_baseline", "baseline")):
      out = pgrep_lines(f"tools/{tool}[.]py")
      for line in out.splitlines():
        k = "baseline:v2" if "--protocol v2" in line else kind
        name = next((n for n, mode, _ in read_plan() if mode == k), None)
        log = ROOT / f"runs/variants/{name}/_runner.log"
        starts = re.findall(r"START (\w+) s(\d+) ", log.read_text()) if name and log.exists() else []
        if starts:
            return name, starts[-1][0], int(starts[-1][1]), line.split()[0]
    out = pgrep_lines("tools/calibrate_checkpoints[.]py")
    for line in out.splitlines():
        src = re.search(r"--source (\S+)", line)
        name = next((n for n, mode, _ in read_plan() if posthoc(mode) and posthoc(mode)[0] == (src and src.group(1))), None)
        log = ROOT / f"runs/variants/{name}/_runner.log"
        starts = re.findall(r"START \S+ (\w+) s(\d+) ", log.read_text()) if name and log.exists() else []
        if starts:
            return name, starts[-1][0], int(starts[-1][1]), line.split()[0]
    return None


def started_at(name, ds, s):
    log = ROOT / f"runs/variants/{name}/_runner.log"
    last = None
    for m in re.finditer(rf"(\d\d-\d\d \d\d:\d\d) START (?:\S+ )?{ds} s{s}\b", log.read_text() if log.exists() else ""):
        last = m.group(1)
    return datetime.strptime(f"{datetime.now().year}-{last}", "%Y-%m-%d %H:%M") if last else None


def winner_modes():
    """Modes ("sync"/"async") that already have a winner (runs/variants/_WINNERS.txt)."""
    f = ROOT / "runs/variants/_WINNERS.txt"
    text = f.read_text() if f.exists() else ""
    # \b: "ASYNC WINNER" must not count as a sync winner
    return {m for m in ("sync", "async") if re.search(rf"\b{m.upper()} WINNER", text)}


def skipped(name, mode, won):
    """Entries of a mode that already has a winner are not run any more (baselines, the
    confirmation entries confirm_* / calib_confirm_* and the robustness study rb_* always are)."""
    if name.startswith(("confirm_", "calib_confirm_", "rb_", "rbsmoke_", "bl_", "calib_bl_")):
        return False
    if mode in won:
        return True
    return (bool(posthoc(mode)) and posthoc(mode)[0] not in ("centralized", "v2_centralized")
            and set(posthoc(mode)[1]) <= won)


def next_entry(attempted):
    """First plan line not done, not skipped, and not already attempted by this queue session."""
    won = winner_modes()
    for line in PLAN.read_text().splitlines():
        p = line.split("#", 1)[0].split()
        if not p:
            continue
        name, mode, seeds = p[0], p[1], [int(s) for s in p[2:]]
        entry = " ".join(p)
        mode, dsets = split_mode(mode)
        if entry in attempted or skipped(name, mode, won):
            continue
        if not all(is_done(name, mode, ds, s) for s in seeds for ds in dsets):
            return entry
    return ""


def phase(name, ds, s):
    plan_mode = next((mode for n, mode, _ in read_plan() if n == name), "")
    if plan_mode.startswith("baseline"):
        p = epoch_progress(name, ds, s)
        return f"centralized baseline training, epoch {p[0] + 1}/{p[1]}" if p else "centralized baseline training"
    if plan_mode == "budget":
        c = (load_json(ROOT / f"runs/variants/_budget/{ds}/s{s}/curve.json") or {}).get("curve") or []
        return f"centralized pilot, last evaluation at epoch {c[-1]['epoch']}" if c else "centralized pilot, before epoch 5"
    if posthoc(plan_mode):
        lines = (ROOT / f"runs/variants/{name}/_runner.log").read_text().splitlines()
        stages = [l.split()[2] for l in lines if re.search(r"\d  +(before|bn|head|bn\+head) +mAP50", l)]
        return f"post-hoc calibration (last stage done: {stages[-1] if stages else 'none'})"
    log = ROOT / f"runs/variants/{name}/_log_{ds}_s{s}.log"
    if not log.exists():
        return "starting"
    with open(log, "rb") as f:
        f.seek(max(0, log.stat().st_size - 3_000_000))
        text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", f.read().decode("utf-8", "replace"))
    rounds = re.findall(r"=== round (\d+)/(\d+)\b", text)
    subs = re.findall(r"sub\s+(\d+)/(\d+) \|", text)
    evals = "valuat" in text[-20000:] and not rounds and not subs
    if subs:
        return f"async submission {subs[-1][0]}/{subs[-1][1]}"
    if rounds:
        return f"sync round {rounds[-1][0]}/{rounds[-1][1]}"
    return "final evaluation" if evals else "setting up"


def epoch_progress(name, ds, s):
    """(epochs finished, total epochs) of a centralized job (budget pilot, baseline) from its log, or None."""
    log = ROOT / f"runs/variants/{name}/_runner.log"
    if not log.exists():
        return None
    with open(log, "rb") as f:
        f.seek(max(0, log.stat().st_size - 3_000_000))
        text = f.read().decode("utf-8", "replace")
    starts = [m.end() for m in re.finditer(rf"START {ds} s{s}\b", text)]
    if not starts:
        return None
    ep = re.findall(r"=== centralized epoch (\d+)/(\d+) ===", text[starts[-1]:])
    return (int(ep[-1][0]) - 1, int(ep[-1][1])) if ep else None


def queue_left(cur_key=None, cur_left=0.0):
    """('run i/N' of the current run, minutes left in the whole queue). The current run
    (name, ds, seed) counts cur_left minutes instead of its typical duration."""
    runs, durs, won = all_runs(read_plan()), durations(), winner_modes()
    left, pos = 0.0, ""
    for r in runs:
        name, mode, ds, s = r
        if cur_key and (name, ds, s) == tuple(cur_key):
            left, pos = left + cur_left, f"run {runs.index(r) + 1}/{len(runs)}"
        elif not (skipped(name, mode, won) or is_done(*r)):
            left += typical(name, ds, durs, mode)
    return pos, left


def eta_text(cur_key, elapsed_min, left_min):
    """'run: X min in, ~Y min left (~HH:MM) | queue: run i/N, ~Z h left (~MM-DD HH:MM)' for ⏱ lines."""
    try:
        pos, total = queue_left(cur_key, left_min)
        now = datetime.now()
        return (f"run: {elapsed_min:.0f} min in, ~{left_min:.0f} min left (~{now + timedelta(minutes=left_min):%H:%M}) | "
                f"queue: {pos}, ~{total / 60:.1f} h left (~{now + timedelta(minutes=total):%m-%d %H:%M})")
    except Exception:
        return f"run: {elapsed_min:.0f} min in, ~{left_min:.0f} min left"


def entry_for(kind, source=None):
    """Plan entry name of a job type: 'budget', 'baseline', or the posthoc entry of `source` --
    the entry the queue is running, if it is of that type, else the first such plan line."""
    ent = _queue_entry()
    if ent and (ent[1] == kind or (source and posthoc(ent[1]) and posthoc(ent[1])[0] == source)):
        return ent[0]
    for n, mode, _ in read_plan():
        if mode == kind or (source and posthoc(mode) and posthoc(mode)[0] == source):
            return n
    return None


def live_bits():
    """(current node's progress bar as plain text, last ⏱ line) from the log of the job in progress."""
    link = ROOT / "runs/variants/_current.log"
    try:
        target = link.resolve()
        with open(target, "rb") as f:
            f.seek(max(0, target.stat().st_size - 400_000))
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return "", ""
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", raw)
    bar = ""
    for seg in reversed(re.split(r"[\r\n]", text)):   # the bar redraws in place with \r
        if re.search(r"\d+/\d+", seg) and ("Ep " in seg or "Passed" in seg):
            bar = " ".join(seg.replace("━", "").replace("╸", "").split())
            break
    etas = [l for l in text.splitlines() if l.startswith("⏱")]
    return bar, (etas[-1] if etas else "")


def _status_numbers(line: str) -> dict:
    """Numbers out of the --line status: runs done/total, the run's own progress and time left."""
    g = lambda pat: re.search(pat, line)
    out = {}
    if (m := g(r"(\d+)/(\d+) runs done")):
        out["q"] = (int(m[1]), int(m[2]))
    if (m := g(r"now: (\S+ \S+ s\d+)")):
        out["name"] = m[1]
    if (m := g(r"(?:async submission|sync round) (\d+)/(\d+)")):
        out["run"] = (int(m[1]), int(m[2]))
    if (m := g(r"(\d+) of ~(\d+) min, ~(\d+) min left")):
        out["run_min"] = (int(m[1]), int(m[2]), int(m[3]))
    if (m := g(r"queue: ~([\d.]+) h left, ends ~(\S+ \S+)")):
        out["q_left"] = (float(m[1]), m[2])
    if (m := g(r"\| ([a-z][^|]*?) \| \d+ of ~")):
        out["phase"] = m[1]
    return out


def _job_lines(job, now):
    """Plain-text lines on the job in progress (shared by the full view and the watcher)."""
    if not job:
        return ["no job running (queue finished, stopped, or between two jobs -- see runs/variants/_queue.log)"]
    where = f"{job['ds']} seed {job['seed']}" if job["ds"] else "starting"
    left = (f"~{job['left']:.0f} min left (~{now + timedelta(minutes=job['left']):%H:%M})"
            if job.get("left") is not None else "time left: after the first step")
    return [f"{job['name']} ({job['mode']}) -- {where}",
            f"what: {job['what']}",
            f"progress: {job['unit']} {job['done']}/{job['total']} done, {job['elapsed']:.0f} min in, {left}"
            + (f" | {job['detail']}" if job.get("detail") else ""),
            f"health: {job['health']} | {gpu_util()}"]


def watch(every: float) -> None:
    """Live view, redrawn in place: the job in progress (what it does, its progress in its own unit,
    a health check), the inner training bar, the queue by section, the last completed runs and
    what comes next. Ctrl+C to stop."""
    from rich.console import Group
    from rich.live import Live
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    def bar(done, total, width=30):
        return ProgressBar(total=max(total, 1), completed=min(done, max(total, 1)), width=width)

    state, last = {}, 0.0

    def render():
        nonlocal last
        now = datetime.now()
        if time.time() - last > 20:
            job = current_job()
            state.update(job=job, eta=queue_eta(job), done=completed_runs()[:12], next=upcoming(6), gpu=gpu_util())
            last = time.time()
        job, (n_done, n_tot, left, secs) = state["job"], state["eta"]
        head = Table.grid(padding=(0, 1))
        head.add_row(Text("QUEUE", style="bold"), Text(f"{now:%m-%d %H:%M}  {n_done}/{n_tot} runs done "
                                                         f"({100 * n_done / max(n_tot, 1):.0f}%)", style="bold"))
        hb = Table.grid(padding=(0, 1))
        hb.add_row(bar(n_done, n_tot, 40), Text(f"~{left / 60:.1f} h left, ends ~{now + timedelta(minutes=left):%m-%d %H:%M}"))
        head.add_row("", hb)
        cur = Table.grid(padding=(0, 1))
        if job:
            lines = _job_lines(job, now)
            cur.add_row(Text("NOW", style="bold green"), Text(lines[0], style="bold"))
            cur.add_row("", Text(lines[1]))
            prog = Table.grid(padding=(0, 1))
            prog.add_row(bar(job["done"], job["total"]), Text(lines[2].replace("progress: ", "")))
            cur.add_row("", prog)
            nb, _ = live_bits()
            m = re.search(r"(\S+) (Ep \d+/\d+).*?(\d+)/(\d+).*?Passed: (\S+).*?Left: (\S+)", nb)
            if m:
                inner = Table.grid(padding=(0, 1))
                inner.add_row(bar(int(m[3]), int(m[4])), Text(f"{m[1]} {m[2]}: batch {m[3]}/{m[4]}, left {m[6]}"))
                cur.add_row("", inner)
            cur.add_row("", Text(lines[3].replace("health: ", ""), style="green" if job["ok"] else "bold red"))
        else:
            cur.add_row(Text("NOW", style="bold yellow"), Text(_job_lines(None, now)[0]))
        sec = Table(title="by section", title_justify="left", show_edge=False, box=None, padding=(0, 1))
        for c in ("section", "", "runs", "time left"):
            sec.add_column(c)
        for s, (d, tot, lm) in secs.items():
            sec.add_row(s, bar(d, tot, 20), f"{d}/{tot}", f"~{lm / 60:.1f} h" if d < tot else "done")
        done = Table(title="last completed", title_justify="left", show_edge=False, box=None, padding=(0, 1))
        for c in ("finished", "entry", "dataset", "seed", "min", "result"):
            done.add_column(c)
        for when, name, ds, s, mins, res in state["done"]:
            done.add_row(when, name, ds, str(s), str(mins), res)
        nxt = Table(title="next", title_justify="left", show_edge=False, box=None, padding=(0, 1))
        for c in ("entry", "mode", "runs", "what it does"):
            nxt.add_column(c)
        for name, mode, dsets, seeds, k in state["next"]:
            nxt.add_row(name, mode, str(k), describe(name, mode)[:90])
        return Group(head, Text(""), cur, Text(""), sec, Text(""), done, Text(""), nxt)

    try:
        with Live(render(), refresh_per_second=1, transient=False) as live:
            while True:
                time.sleep(every)
                live.update(render())
    except KeyboardInterrupt:
        pass


def main():
    if "--watch" in sys.argv:   # live view (see watch)
        i = sys.argv.index("--watch")
        watch(float(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1][0].isdigit() else 2)
        return
    if "--next" in sys.argv:   # used by runs/variants/_queue.sh
        f = Path(sys.argv[sys.argv.index("--next") + 1])
        print(next_entry(set(f.read_text().splitlines()) if f.exists() else set()))
        return
    now = datetime.now()
    job = current_job()
    n_done, n_tot, left, secs = queue_eta(job)
    eta = now + timedelta(minutes=left)
    if "--line" in sys.argv:
        if job and job["ds"]:
            cur = (f"{job['name']} {job['ds']} s{job['seed']} -- {job['what'].split(':')[0].split(',')[0]}, "
                   f"{job['unit']} {job['done']}/{job['total']} | {job['elapsed']:.0f} min in"
                   + (f", ~{job['left']:.0f} min left" if job.get("left") is not None else "")
                   + ("" if job["ok"] else " | WARNING: no progress, check the runner log"))
        else:
            cur = "no job running"
        print(f"{'⏱ ' if '--eta' in sys.argv else ''}{now:%m-%d %H:%M} [status] {n_done}/{n_tot} runs done "
              f"({100 * n_done / n_tot:.0f}%) | now: {cur} | queue: ~{left / 60:.1f} h left, "
              f"ends ~{eta:%m-%d %H:%M}", flush=True)
        return
    if "--done" in sys.argv:   # every finished run, newest first
        for r in completed_runs():
            print(f"{r[0]}  {r[1]:26s} {r[2]:7s} s{r[3]:<4d} {r[4]:4d} min  {r[5]}")
        return
    print(f"Queue status  {now:%Y-%m-%d %H:%M}")
    print(f"Overall: {n_done}/{n_tot} runs done ({100 * n_done / n_tot:.0f}%) -- queue ends ~{eta:%m-%d %H:%M} "
          f"({left / 60:.1f} h left)\n")
    print("NOW\n  " + "\n  ".join(_job_lines(job, now)))
    print("\nBY SECTION")
    for s, (d, tot, lm) in secs.items():
        print(f"  {s:40s} {d:4d}/{tot:<4d} " + ("done" if d == tot else f"~{lm / 60:.1f} h left"))
    print("\nLAST COMPLETED (all: --done)")
    for r in completed_runs()[:12]:
        print(f"  {r[0]}  {r[1]:26s} {r[2]:7s} s{r[3]:<4d} {r[4]:4d} min  {r[5]}")
    print("\nNEXT")
    for name, mode, dsets, seeds, k in upcoming(6):
        print(f"  {name:26s} {mode:30s} {k} runs -- {describe(name, mode)[:100]}")


if __name__ == "__main__":
    main()
