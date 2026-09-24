# DAFYOLO — Disjoint Asynchronous Federated YOLO

Federated object detection where nodes hold disjoint or overlapping class
subsets. Two implementations live in this repo:

## [`fedyolo_system/`](fedyolo_system/) — current

A class-decomposed rewrite: every per-class output channel aggregates only
across the nodes that own that class, so training doesn't need a merge
heuristic. Two federation strategies (sync: round-barrier FedAvg; async:
staleness-discounted delta aggregation), a centralized baseline and a true
full-annotation oracle baseline for comparison, and support for
YOLOv8/YOLO26 (classic + native end2end)/YOLOE. Validated exclusively on
real data — no synthetic/toy dataset in this repo.
Start with [`fedyolo_system/README.md`](fedyolo_system/README.md).

## [`legacy/`](legacy/) — prior approach, kept for reference

Client/server scripts (`client_updated.py` + `server_updated.py`, plus
earlier `client_v2.py`/`server_v2.py` and `run_experiments.py`) implementing
a selectable-strategy merge approach (FedHead, Stitch, TIES, FedAvg,
FedProx, YOLO-Inc, DFKD...). `client_updated.py`/`server_updated.py` are the
most current pair in this set; `client_v2.py`/`server_v2.py` are kept only
for the experimental DFKD strategy, which isn't ported anywhere else.
Machine-specific paths (remote GPU box vs. this Mac) live in
[`legacy/server_paths.yaml`](legacy/server_paths.yaml).

## [`dafyolo_field_report.html`](dafyolo_field_report.html)

The full audit trail — every run, every bug found and fixed, root-cause
analysis, and a macOS compatibility audit. Grows with each update; open it
directly in a browser (no external dependencies besides a Google Fonts
stylesheet). Also published at
https://claude.ai/code/artifact/2a78ddc2-3ee6-41ce-987f-1328dc4005fa.

## [`dafyolo_current_state.html`](dafyolo_current_state.html)

A living summary: current-best config, current numbers, open problems,
and a schematic of how sync/async aggregation lines up with the
YOLOv8/YOLO26/YOLOE architectures. Overwritten in place on each update
rather than appended to — read this first if you just want "where things
stand," and the field report above only if you want the history. Also
published at https://claude.ai/code/artifact/caa25a33-1b24-4b0e-be7d-6fcde03d54a4.

## Install

Each implementation has its own `requirements.txt`:
```bash
pip install -r legacy/requirements.txt          # legacy client/server scripts
pip install -r fedyolo_system/requirements.txt   # fedyolo_system
```
