"""One-line ETA printed after every node's local round (run log), e.g.

    ⏱ node_B round 3/10 done (11/40 node-rounds) | run: 14 min in, ~31 min left (~16:58) | queue: run 53/93, ~27.3 h left (~09-30 19:40)

Run progress = node-rounds finished / total node-rounds of this federated run
(nodes train one at a time, in config order). Elapsed time is measured from the
start of the cofy.simulate process; the final evaluation after training
(~1-3 min) is not included. Queue ETA = this run's remaining time + the typical
duration of every run still pending in runs/variants/_plan.txt (tools/queue_status.py).
Never raises: any failure just skips the line.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_start() -> float | None:
    import psutil
    p = psutil.Process(os.getpid())
    while p is not None:
        if "cofy.simulate" in " ".join(p.cmdline()):
            return p.create_time()
        p = p.parent()
    return None


def _queue_part(cfg, run_left_min: float) -> str:
    sys.path.insert(0, str(ROOT / "tools"))
    import queue_status as qs
    parts = Path(cfg.output_dir).resolve().relative_to(ROOT / "runs/variants").parts   # (name, ds, mode)
    pos, left = qs.queue_left((parts[0], parts[1], cfg.seed), run_left_min)
    eta = datetime.now() + timedelta(minutes=left)
    return f" | queue: {pos}, ~{left / 60:.1f} h left (~{eta:%m-%d %H:%M})"


def eta_line(cfg, node, round_idx: int, done_total: tuple[int, int] | None = None) -> str:
    try:
        async_mode = cfg.federation.mode == "async"
        n_rounds = cfg.federation.async_node_cycles if async_mode else cfg.federation.rounds
        k = len(cfg.nodes)
        pos = [n.name for n in cfg.nodes].index(node.name)
        done, total = done_total or (round_idx * k + pos + 1, n_rounds * k)
        start = _run_start()
        if start is None:
            return ""
        el = (time.time() - start) / 60
        left = el * (total / done - 1)
        unit = "cycle" if async_mode else "round"
        head = (f"⏱ {node.name} submission {round_idx + 1} done ({done}/{total} submissions) | " if done_total else
                f"⏱ {node.name} {unit} {round_idx + 1}/{n_rounds} done ({done}/{total} node-rounds) | ")
        line = (head +
                f"run: {el:.0f} min in, ~{left:.0f} min left (~{datetime.now() + timedelta(minutes=left):%H:%M})")
        try:
            line += _queue_part(cfg, left)
        except Exception:
            pass
        return line
    except Exception:
        return ""
