"""GPU/process cleanup, invoked automatically by simulate.py and
simulate_to_target.py (see --auto-cleanup-oom, on by default):

  - cleanup_after_crash(): called from an except block if THIS run crashes.
    Kills only this process's own descendant processes (leftover DataLoader
    workers, or ProcessPoolExecutor workers in sync mode, that can survive
    a parent's OOM) and clears orphaned shared-memory segments. Uses the OS
    process tree (psutil children(recursive=True)) rather than a command-
    line pattern match, so it can NEVER touch an unrelated process -- only
    genuine descendants of the crashing run.

  - preflight_cleanup(): called at the start of main(), before building any
    model. REPORTS (does not kill) any other fedyolo process it finds still
    running, plus current GPU memory. It deliberately does not auto-kill:
    a pattern match on the command line can't tell "orphaned from a crashed
    session 9 hours ago" apart from "someone's legitimate concurrent run" --
    guessing wrong there is worse than an occasional manual cleanup. Use
    tools/free_gpu_after_oom.sh yourself (with --dry-run first) if you want
    that more aggressive, pattern-based sweep.

Command-line equivalent for manual/aggressive use: tools/free_gpu_after_oom.sh
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess

log = logging.getLogger("fedyolo.gpu_cleanup")

# Matches both `fedyolo.simulate` and `fedyolo.simulate_to_target` (the
# latter contains the former as a substring). Used for REPORTING only here.
_PATTERN = "fedyolo.simulate"


def _kill_own_descendants() -> list[int]:
    """Kill only processes that are genuine OS-level children of this
    process (recursively) -- e.g. leftover DataLoader workers or
    ProcessPoolExecutor workers. Can never affect an unrelated process."""
    try:
        import psutil
    except ImportError:
        log.warning("[gpu-cleanup] psutil not installed -- skipping descendant cleanup")
        return []

    me = psutil.Process(os.getpid())
    children = me.children(recursive=True)
    pids = [c.pid for c in children]
    for c in children:
        try:
            c.terminate()
        except psutil.NoSuchProcess:
            pass
    if children:
        _, alive = psutil.wait_procs(children, timeout=3)
        for c in alive:
            try:
                c.kill()
            except psutil.NoSuchProcess:
                pass
    return pids


def _clean_stale_shm() -> None:
    for pattern in ("torch_*", "pytorch_*"):
        for path in glob.glob(f"/dev/shm/{pattern}"):
            try:
                os.remove(path)
            except OSError:
                pass


def _report_gpu_memory() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if out:
            log.info(f"[gpu-cleanup] GPU memory: {out}")
    except Exception:
        pass  # no GPU / nvidia-smi unavailable -- not fatal, just skip reporting


def _report_other_processes() -> None:
    try:
        out = subprocess.run(["pgrep", "-af", _PATTERN], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return
    others = [line for line in out.splitlines() if line.strip() and not line.split()[0] == str(os.getpid())]
    if others:
        log.warning(
            f"[gpu-cleanup] {len(others)} other fedyolo process(es) currently running -- "
            f"not touching them automatically (could be legitimate concurrent work). "
            f"If one is actually a leftover from a crash, clean it up yourself with "
            f"tools/free_gpu_after_oom.sh (--dry-run first):\n" + "\n".join(f"    {o}" for o in others)
        )


def preflight_cleanup() -> None:
    _clean_stale_shm()
    _report_other_processes()
    _report_gpu_memory()


def cleanup_after_crash() -> None:
    log.warning("[gpu-cleanup] run crashed -- cleaning up this run's own leftover processes")
    killed = _kill_own_descendants()
    if killed:
        log.warning(f"[gpu-cleanup] killed {len(killed)} leftover descendant process(es): {killed}")
    _clean_stale_shm()
    _report_gpu_memory()
