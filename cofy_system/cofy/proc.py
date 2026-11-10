"""Finding running processes by their command line, the same way on Linux and macOS.

`pgrep -af` prints each match's command line on Linux, but on macOS `-a` means "include ancestors"
and only the PIDs are printed; `pgrep -lf` is the macOS spelling, while on Linux `-l` prints the
process name only. psutil (in requirements.txt) gives the full command line on both; `pgrep` is the
fallback when psutil is missing.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def find_processes(pattern: str) -> list[tuple[int, str]]:
    """[(pid, full command line)] of the processes whose command line matches the regular expression
    `pattern` (searched, as `pgrep -f` does), this process excluded."""
    rx, me = re.compile(pattern), os.getpid()
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        out = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
            except (psutil.Error, TypeError):
                continue
            if p.info["pid"] != me and cmd and rx.search(cmd):
                out.append((p.info["pid"], cmd))
        return out
    flags = "-lf" if sys.platform == "darwin" else "-af"
    try:
        res = subprocess.run(["pgrep", flags, pattern], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    out = []
    for line in res.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit() and int(pid) != me:
            out.append((int(pid), cmd))
    return out


def pgrep_lines(pattern: str) -> str:
    """The matches as 'pid command line' lines, like `pgrep -af` on Linux."""
    return "\n".join(f"{pid} {cmd}" for pid, cmd in find_processes(pattern))
