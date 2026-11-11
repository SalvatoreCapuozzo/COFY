# Variant queue — control and monitoring commands

Run everything from `~/COFY/cofy_system`. Commands that start Python need the
`cofy` environment:

```bash
cd ~/COFY/cofy_system
conda activate cofy
```

The queue runs the variants listed in `runs/variants/_plan.txt`, in order, 3 datasets
per seed. Finished runs are never redone. An interrupted run restarts from its
beginning with the same seed.

---

## 1. See where it is

```bash
# One-line status every 10 minutes (overall run N/M, current run, round, time left, end time)
tail -f runs/variants/_queue.log

# Full overview on demand, refreshed every minute
watch -n 60 python tools/queue_status.py

# START / DONE / FAIL of every run, all variants
tail -f runs/variants/*/_runner.log

# Live view, redrawn in place (Ctrl+C to stop):
#  - the job in progress: what it does, its progress in its own unit (epoch, calibration stage,
#    federated round / update), time left, and a health check (runner alive, time since its
#    last progress, GPU load), shown in red if nothing new has come for too long;
#  - the queue by section; the last completed runs with their results; what comes next.
python tools/queue_status.py --watch
python tools/queue_status.py --done       # every finished run, newest first

# Raw training output of the run in progress, live progress bar included
# (first line = its position in the queue). Don't pipe it into grep: the bar redraws
# in place on one line, so grep only sees it when a node's training ends.
tail -F runs/variants/_current.log

# Is anything running?
pgrep -fl "_queue[.]sh|tools/(run_variant|calibrate_checkpoints|run_baseline)[.]py|cofy[.]simulate"

# Results so far: Fed:Cent / Fed:Oracle table, every method x dataset. First line =
# the run in progress; its row is marked "<< RUNNING NOW". The note column gives the
# reason for extra seeds (projected confidence interval) or why a row cannot pass.
python tools/results_table.py
watch -n 300 python tools/results_table.py      # refreshing every 5 min
cat runs/variants/_results.txt                  # same table, rewritten after every finished run

# GPU and disk (the runner refuses to start a run below 5 GB free)
nvidia-smi -l 5
df -h .

# Decisions taken by the queue: winners, promoted screens, extra seeds (with reasons)
cat runs/variants/_WINNERS.txt
grep -a "\[search\]" runs/variants/_queue.log
```

---

## 1b. Confirmatory study, held-out dataset, robustness, competitors (from 2026-10-28)

```bash
# Pre-registered confirmation: status, and the one-time test once all 10 seeds are done
python tools/confirm_analysis.py            # --write: analysis/CONFIRMATORY_RESULTS.md
# Have the frozen files changed since registration? (code read from _frozen/confirm; amended files listed apart)
python tools/confirm_manifest.py --check

# Held-out DENTEX: done 2026-11-01 (budget 20 epochs, amendment 3); 60 plan lines queued
python tools/make_heldout_configs.py --dataset dentex --budget 20 --amendment 3

# Robustness study (rb_*) and its report; fairness and cost reports
python tools/robust_eval.py --report         # analysis/ROBUSTNESS.md
#   (both forms first evaluate any checkpoint not yet scored, ON THE GPU: run only when the queue is idle)
python tools/fairness_analysis.py --confirm --out analysis/FAIRNESS_CONFIRM.md
python tools/cost_report.py --out analysis/COST.md

# GPU power log (one line every 10 s; started 10-28, runs until killed)
tail -3 runs/variants/_power.csv
pgrep -fl "nvidia-smi --query-gpu"          # kill <pid> after the study
```

Entries named `confirm_*` / `calib_confirm_*` run on the frozen code in
`_frozen/confirm/`. Everything else runs on the working tree.

---

## 2. Pause without losing anything (freeze)

Freezes the run in place. It keeps its GPU memory (~5.6 GB) while paused.
Don't reboot or suspend the machine while frozen.

```bash
# pause
pkill -STOP -f "tools/run_variant[.]py|cofy[.]simulate|envs/cofy/bin/python -s -c from multiprocessing"

# resume
pkill -CONT -f "tools/run_variant[.]py|cofy[.]simulate|envs/cofy/bin/python -s -c from multiprocessing"
```

The 10-minute status lines keep appearing while frozen; ignore them.

---

## 3. Stop completely (frees the GPU)

The run in progress is lost and will restart from its beginning. To lose almost
nothing, stop right after a `DONE` line in `tail -f runs/variants/*/_runner.log`.

Order matters: queue manager first, so it can't start the next variant.

```bash
pkill -f "runs/variants/_queue[.]sh"                           # 1. queue manager (and its status ticker)
pkill -f "tools/(run_variant|calibrate_checkpoints|run_baseline)[.]py"   # 2. runner (training, calibration or baseline)
pkill -f "cofy[.]simulate"                                  # 3. training run in progress
sleep 5; pkill -f "envs/cofy/bin/python -s -c from multiprocessing"   # 4. leftover worker processes
nvidia-smi                                                     # GPU should be back to ~9 MiB
```

If `nvidia-smi` still shows memory in use, find the process and kill it by PID:
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, then `kill <pid>`.

---

## 4. Resume (after a stop, a crash, or a reboot)

A reboot kills the queue: it does **not** restart on its own.

```bash
cd ~/COFY/cofy_system && conda activate cofy
pgrep -fl "_queue[.]sh|tools/(run_variant|calibrate_checkpoints|run_baseline)[.]py|cofy[.]simulate"   # must print nothing
python tools/launch_detached.py runs/variants/_queue.log -- /bin/bash runs/variants/_queue.sh
tail -f runs/variants/_queue.log
```

`launch_detached.py` starts the queue in its own session, so closing the terminal
or VS Code does not stop it.

---

## 5. Change what runs next

Edit `runs/variants/_plan.txt` (one line per entry: `name mode seeds...`). The queue
re-reads it before every entry, so no restart is needed. Modes:

- `sync` / `async`: training runs; configs in `configs/variants/<name>/<dataset>_s<seed>.yaml`.
- `posthoc:<source>:<modes,>`: final calibration applied to finished runs of `<source>`
  (`tools/calibrate_checkpoints.py`; rows `<source>+cal` in the table).
- `baseline`: centralized baselines for seeds the primary study never ran (`tools/run_baseline.py`).

The queue edits the plan itself (`tools/search_rules.py`, after every entry):
- it promotes a seed-0 screen to seeds 1–2 if it reaches at least 95% on all datasets;
- it adds the seeds a confidence interval needs (decided once, after seeds 0–2);
- once a sync or async winner exists, it skips the remaining entries of that mode.

To run skipped entries anyway, delete `runs/variants/_WINNERS.txt`.

---

## Notes

- `pkill -f PATTERN` also matches any shell whose own command line contains
  PATTERN. Typed in a terminal this is fine. Inside a script or `bash -c "..."`
  that contains the same text, kill by PID instead (`ps -eo pid,args | grep ...`).
- If a variant fails twice in a row, the queue halts on purpose. The reason is in
  `runs/variants/<name>/_log_<dataset>_s<seed>.log`. Fix it, then resume (section 4).
- Main study (seeds 0–2, finished): `python tools/analyze_study.py`.
