#!/usr/bin/env bash
# Clean up after a training process OOM-crashes.
#
# NOTE on what this does and doesn't do:
#   - When a process actually exits (crash or otherwise), the NVIDIA driver
#     already reclaims its VRAM automatically -- nothing to do there.
#   - torch.cuda.empty_cache() does NOT help after a crash: it only returns
#     cached-but-unused blocks back to the driver from WITHIN a still-running
#     process. It can't reach into a dead process's memory.
#   - This script exists for the cases where cleanup does NOT happen cleanly:
#     a hung/zombie process, an orphaned CUDA-context-holding child that
#     outlived its parent, or leftover /dev/shm segments from crashed
#     DataLoader workers (a common source of confusing follow-on failures).
#
# Usage:
#   bash tools/free_gpu_after_oom.sh            # clean up
#   bash tools/free_gpu_after_oom.sh --dry-run  # show what would be killed/removed, do nothing

set -uo pipefail
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

echo "=== GPU memory before ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo
echo "=== processes currently holding GPU memory ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "(none)"

echo
echo "=== lingering fedyolo training processes ==="
# matches both `fedyolo.simulate` and `fedyolo.simulate_to_target`
# (the latter contains the former as a substring)
PIDS=$(pgrep -f "fedyolo\.simulate" || true)
if [ -n "$PIDS" ]; then
    echo "found: $PIDS"
    ps -o pid,etime,cmd -p $PIDS 2>/dev/null
    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry-run] would SIGTERM, then SIGKILL if still alive after 3s"
    else
        kill -TERM $PIDS 2>/dev/null
        sleep 3
        STILL=$(pgrep -f "fedyolo\.simulate" || true)
        if [ -n "$STILL" ]; then
            echo "still alive after SIGTERM, sending SIGKILL: $STILL"
            kill -KILL $STILL 2>/dev/null
        fi
    fi
else
    echo "none found"
fi

echo
echo "=== any process still holding GPU memory after that ==="
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
if [ -n "$GPU_PIDS" ]; then
    for pid in $GPU_PIDS; do
        if [ "$DRY_RUN" = 1 ]; then
            echo "[dry-run] would SIGKILL PID $pid ($(ps -o cmd= -p "$pid" 2>/dev/null))"
        else
            echo "killing PID $pid ($(ps -o cmd= -p "$pid" 2>/dev/null))"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    [ "$DRY_RUN" = 0 ] && sleep 2
else
    echo "none"
fi

echo
echo "=== orphaned torch shared-memory segments in /dev/shm ==="
STALE_SHM=$(find /dev/shm -maxdepth 1 -user "$(whoami)" \( -name "torch_*" -o -name "pytorch_*" \) 2>/dev/null || true)
if [ -n "$STALE_SHM" ]; then
    echo "$STALE_SHM"
    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry-run] would delete the above"
    else
        find /dev/shm -maxdepth 1 -user "$(whoami)" \( -name "torch_*" -o -name "pytorch_*" \) -delete 2>/dev/null
    fi
else
    echo "none"
fi

echo
echo "=== GPU memory after ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "(none)"
