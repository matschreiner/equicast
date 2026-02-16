i#!/bin/bash
# Interactive shell on Leonardo
# Usage: ./scripts/leonardo_interactive.sh [gpus] [cpus]
#   gpus: number of GPUs (default: 0)
#   cpus: number of CPUs (default: gpu*8 if gpus>0, else 1)

GPUS=${1:-0}
CPUS=${2:-$(( GPUS > 0 ? GPUS * 8 : 1 ))}

GPU_FLAG=""
if [ "$GPUS" -gt 0 ]; then
    GPU_FLAG="--gres=gpu:$GPUS"
fi

srun --nodes=1 --ntasks=1 $GPU_FLAG --cpus-per-task=$CPUS --time=01:00:00 --partition=boost_usr_prod --account=deste_340_26 --pty bash
