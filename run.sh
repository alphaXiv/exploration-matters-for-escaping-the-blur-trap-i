#!/usr/bin/env bash
set -euo pipefail

echo "ORX ExploreGS controlled reproduction"
echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "node=$(hostname) completion_index=${JOB_COMPLETION_INDEX:-0}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

node_rank="${JOB_COMPLETION_INDEX:-0}"
nnodes="${ORX_NNODES:-2}"
master_addr="${ORX_MASTER_ADDR:-127.0.0.1}"
master_port="${ORX_MASTER_PORT:-29500}"
gpus_per_node="$(python -c 'import torch; print(torch.cuda.device_count())')"

if [ "$gpus_per_node" -lt 1 ]; then
  echo "fatal: no CUDA devices visible" >&2
  exit 2
fi

echo "distributed nnodes=$nnodes node_rank=$node_rank gpus_per_node=$gpus_per_node master=$master_addr:$master_port"
torchrun \
  --nnodes "$nnodes" \
  --nproc-per-node "$gpus_per_node" \
  --node-rank "$node_rank" \
  --master-addr "$master_addr" \
  --master-port "$master_port" \
  reproduce.py

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
