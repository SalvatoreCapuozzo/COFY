# COFY — Class-Owned Federated YOLO (formerly DAFYOLO)

Federated object detection for sites that annotate different classes: every class has exactly one
owner node, only the owner changes that class's output rows, and the server holds no data. Sequential
(server-scheduled turns) and concurrent (asynchronous) schemes, a barrier-free calibration, a secured
client-server implementation and ownership-aware robust aggregation.

**Try it:** `cd fedyolo_system && python demo/run_demo.py --use-case wbc --scheme sequential` runs a real
four-node federation over HTTP on one machine, with small samples of the three use cases
(parasite eggs, white blood cells, dental radiographs). See [`fedyolo_system/demo/`](fedyolo_system/demo/).
Or run it online for free: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SalvatoreCapuozzo/DAFYOLO/blob/master/fedyolo_system/demo/COFY_demo.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/SalvatoreCapuozzo/DAFYOLO)

**Licence:** code under the GNU AGPL-3.0 ([`LICENSE`](LICENSE)), the licence of the Ultralytics
library it builds on. The demo images keep their own licences
([`fedyolo_system/demo/data/README.md`](fedyolo_system/demo/data/README.md)).


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
