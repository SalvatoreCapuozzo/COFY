#!/bin/bash
# One command, resumable: run it, interrupt it anytime (Ctrl-C, closing
# the terminal, anything), and run this exact same command again later to
# pick up wherever it stopped. See tools/run_matrix.py's own docstring
# for --dry-run / --only / --list-deformabledetr.
cd "$(dirname "$0")"

# Pin to the dl_env conda environment's own interpreter explicitly --
# do NOT rely on whatever "python3" happens to be first on PATH. This bit
# for real once already: launched from a shell sitting in (base), every
# one of the 53 planned runs died instantly on "No module named 'torch'"
# (base has no torch, only dl_env does), burning through the whole
# matrix in seconds with nothing actually trained.
DL_ENV_PY="$HOME/anaconda3/envs/dl_env/bin/python3"
if [ -x "$DL_ENV_PY" ]; then
    PY="$DL_ENV_PY"
else
    echo "=== dl_env python3 not found at $DL_ENV_PY -- falling back to plain python3, but this is exactly how the torch-missing failure loop happened before. Fix the path or activate dl_env yourself. ===" >&2
    PY="python3"
fi

# RT-DETR's decoder hits aten::grid_sampler_2d_backward, which MPS
# doesn't implement -- confirmed for real (a crash mid-matrix, not
# assumed): every other architecture here trains fine natively on MPS,
# so this is scoped to whatever RT-DETR's own backward pass needs, not a
# general MPS problem. This fallback only engages for ops MPS genuinely
# lacks -- everything already running natively on MPS keeps doing so --
# so it's safe to set for the whole matrix rather than special-casing
# just the RT-DETR legs.
export PYTORCH_ENABLE_MPS_FALLBACK=1

"$PY" tools/run_matrix.py "$@"
