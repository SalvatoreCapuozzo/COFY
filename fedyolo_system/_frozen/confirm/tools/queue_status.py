#!/usr/bin/env python3
"""Where is the variant queue right now?

    python tools/queue_status.py          # full view
    python tools/queue_status.py --line   # one line (the queue appends this to _queue.log every 10 min)

Reads runs/variants/_plan.txt (the queue's own run list), each run's
summary.json (done or not), the live fedyolo.simulate process (which run is
training), the tail of that run's log (which round), and the durations of
finished runs (for the time estimate).
"""
import json, re, statistics, subprocess, sys, time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


def all_runs(plan):
    dsets = plan_datasets()
    return [(name, mode, ds, s) for name, mode, seeds in plan for s in seeds for ds in dsets.get(name, DATASETS)]


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
    N (acal), see fedyolo/calibrate.py calibrate_progressive."""
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


def current_run():
    """(variant, dataset, seed, pid) of the training process, or None."""
    out = subprocess.run(["pgrep", "-af", "fedyolo[.]simulate"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.search(r"configs/variants/(\w+)/(\w+)_s(\d+)\.yaml", line)
        if m:
            return m.group(1), m.group(2), int(m.group(3)), line.split()[0]
    for tool, kind in (("budget_pilot", "budget"), ("run_baseline", "baseline")):
      out = subprocess.run(["pgrep", "-af", f"tools/{tool}[.]py"], capture_output=True, text=True).stdout
      for line in out.splitlines():
        k = "baseline:v2" if "--protocol v2" in line else kind
        name = next((n for n, mode, _ in read_plan() if mode == k), None)
        log = ROOT / f"runs/variants/{name}/_runner.log"
        starts = re.findall(r"START (\w+) s(\d+) ", log.read_text()) if name and log.exists() else []
        if starts:
            return name, starts[-1][0], int(starts[-1][1]), line.split()[0]
    out = subprocess.run(["pgrep", "-af", "tools/calibrate_checkpoints[.]py"], capture_output=True, text=True).stdout
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
    """Entries of a mode that already has a winner are not run any more (baselines and the
    winner's confirmation entries, named confirm_* / calib_confirm_*, always are)."""
    if name.startswith(("confirm_", "calib_confirm_")):
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
    """Plan entry name of a job type: 'budget', 'baseline', or the posthoc entry of `source`."""
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


def watch(every: float) -> None:
    """Live view with bars redrawn in place: the current node's training, the run in progress
    and the whole queue, each with its time left. Ctrl+C to stop."""
    from rich.live import Live
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
    prog = Progress(TextColumn("{task.description}", justify="left"), BarColumn(bar_width=40),
                    MofNCompleteColumn(), TextColumn("{task.fields[info]}"), expand=False)
    t_node = prog.add_task("node ", total=1, info="")
    t_run = prog.add_task("run  ", total=1, info="")
    t_q = prog.add_task("queue", total=1, info="")
    nums, last_status = {}, 0.0
    try:
        with Live(prog, refresh_per_second=4, transient=False):
            while True:
                if time.time() - last_status > 30:
                    line = subprocess.run([sys.executable, __file__, "--line"], capture_output=True, text=True).stdout
                    nums, last_status = _status_numbers(line), time.time()
                bar, eta = live_bits()
                m = re.search(r"(\S+) (Ep \d+/\d+).*?(\d+)/(\d+).*?Passed: (\S+).*?Left: (\S+)", bar)
                if m:
                    prog.update(t_node, completed=int(m[3]), total=int(m[4]),
                                description=f"node  {m[1]} {m[2]}", info=f"passed {m[5]}  left {m[6]}")
                else:
                    prog.update(t_node, completed=0, total=1, description="node ", info=nums.get("phase", ""))
                run_done, run_total = nums.get("run", (0, 1))
                e = re.search(r"run: (\d+) min in, ~(\d+) min left \(~(\S+)\)", eta)
                if e:
                    info = f"{e[1]} min in, ~{e[2]} min left (~{e[3]})"
                elif "run_min" in nums:
                    info = f"{nums['run_min'][0]} min in, ~{nums['run_min'][2]} min left"
                else:
                    info = ""
                prog.update(t_run, completed=run_done, total=max(run_total, 1),
                            description=f"run   {nums.get('name', '-')}", info=info)
                q_done, q_total = nums.get("q", (0, 1))
                ql = nums.get("q_left")
                prog.update(t_q, completed=q_done, total=max(q_total, 1),
                            info=(f"~{ql[0]:.1f} h left (ends ~{ql[1]})" if ql else ""))
                time.sleep(every)
    except KeyboardInterrupt:
        pass


def main():
    if "--watch" in sys.argv:   # live view (see watch)
        i = sys.argv.index("--watch")
        watch(float(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 1)
        return
    if "--next" in sys.argv:   # used by runs/variants/_queue.sh
        f = Path(sys.argv[sys.argv.index("--next") + 1])
        print(next_entry(set(f.read_text().splitlines()) if f.exists() else set()))
        return
    plan, durs, now = read_plan(), durations(), datetime.now()
    runs = all_runs(plan)
    won = winner_modes()
    done = [skipped(r[0], r[1], won) or is_done(*r) for r in runs]
    cur = current_run()
    left_min = 0.0
    cur_txt = "no training process running"
    for (name, mode, ds, s), d in zip(runs, done):
        if d:
            continue
        t = typical(name, ds, durs, mode)
        if cur and (name, ds, s) == cur[:3]:
            st = started_at(name, ds, s)
            el = (now - st).total_seconds() / 60 if st else 0
            rem = max(t - el, 1)
            prog = epoch_progress(name, ds, s) if mode == "budget" or mode.startswith("baseline") else None
            if prog and prog[0] >= 1:   # measured epoch rate (a pilot may also stop early)
                rem = max((prog[1] - prog[0]) * el / prog[0], 1)
                t = el + rem
            left_min += rem
            idx = runs.index((name, mode, ds, s)) + 1
            vruns = [r for r in runs if r[0] == name]
            vidx = vruns.index((name, mode, ds, s)) + 1
            cur_txt = (f"{name} {ds} s{s} = run {vidx}/{len(vruns)} of '{name}', run {idx}/{len(runs)} overall | "
                       f"{phase(name, ds, s)} | {el:.0f} of ~{t:.0f} min, ~{rem:.0f} min left")
        else:
            left_min += t
    n_done = sum(done)
    eta = now + timedelta(minutes=left_min)
    if "--line" in sys.argv:
        print(f"{'⏱ ' if '--eta' in sys.argv else ''}{now:%m-%d %H:%M} [status] {n_done}/{len(runs)} runs done "
              f"({100*n_done/len(runs):.0f}%) | now: {cur_txt} | queue: ~{left_min / 60:.1f} h left, "
              f"ends ~{eta:%m-%d %H:%M}", flush=True)
        return
    print(f"Queue status  {now:%Y-%m-%d %H:%M}")
    print(f"Overall: {n_done}/{len(runs)} runs done ({100*n_done/len(runs):.0f}%) -- queue ends ~{eta:%m-%d %H:%M} "
          f"({left_min/60:.1f} h left)\n")
    for name, mode, seeds in plan:
        vr = [(r, d) for r, d in zip(runs, done) if r[0] == name]
        k = sum(d for _, d in vr)
        tag = "done" if k == len(vr) else ("RUN " if cur and cur[0] == name else "wait")
        print(f"  [{tag}] {name:10s} {mode:5s} seeds {' '.join(map(str, seeds)):6s} {k}/{len(vr)} runs")
    print(f"\nCurrent: {cur_txt}")


if __name__ == "__main__":
    main()
