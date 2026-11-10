#!/usr/bin/env python3
"""Launch a cofy run in its own session so it survives the parent exiting.

Plain `nohup ... &` is not enough here: when the launching process (a shell,
or an agent session) is torn down, the whole process GROUP is signalled and
the training dies with it -- which is exactly how an M5 run was lost at round
5 of 10. macOS has no `setsid`, so this uses subprocess's start_new_session
(a setsid() in the child before exec) to detach properly.

    python3 tools/launch_detached.py runs/_m5.log -- python3 -m cofy.simulate ...
"""
import subprocess, sys, os, time

log = sys.argv[1]
cmd = sys.argv[sys.argv.index("--") + 1:]
# append, never overwrite: a relaunch must not erase the log of a process still writing to it
# (the queue status reads the entry in progress from runs/variants/_queue.log)
os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
with open(log, "ab") as f:
    f.write(f"==== {time.strftime('%m-%d %H:%M')} launched: {' '.join(cmd)} ====\n".encode())
    f.flush()
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=os.getcwd())
print(f"pid={p.pid} log={log}")
