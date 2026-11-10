# legacy/ — earlier work, kept for reference

Nothing here is used by COFY (`cofy_system/`). It is the history of the project before the current system
and before the study, moved out of the way on 2026-11-10.

| Folder | What it holds |
|---|---|
| `prototype/` | The first COFY prototype: client/server scripts with selectable merge strategies (FedHead, Stitch, TIES, FedAvg, FedProx, YOLO-Inc, DFKD), its experiment runner and notes. `client_updated.py` / `server_updated.py` are the most recent pair; `client_v2.py` / `server_v2.py` are kept only for the experimental DFKD strategy. |
| `reports/` | HTML reports of the early COFY development (`early_*.html`), the ISBI workshop draft and old result logs. |
| `cofy_early/` | Early COFY material: the first system description (`early_system_description.md`), the experiment-matrix runner (`run_matrix.sh`, `tools/run_matrix.py`, superseded by the study's queue), its configs (`configs/`) and the GPU-limit calibration results. |
| `data/` | Local data and outputs of the prototype (client datasets, global validation sets, global models, `runs/`, `datasets/`, weights). Not in git (`.gitignore`). |

The prototype scripts expect their data at the repository root; to run one again, run it from
`legacy/data/` or adjust the paths in `prototype/server_paths.yaml`.
