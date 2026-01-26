#!/bin/bash
#SBATCH --job-name=train_fsdp_2node
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/sharedfs/nebius-demoday-test
#SBATCH --output=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.err

set -euo pipefail
set -x

mkdir -p /mnt/sharedfs/nebius-demoday-test/slurm-logs

# ---- NCCL / GLOO networking ----
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Keep the same deterministic TCP path you validated
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export GLOO_USE_IPV6=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=2

# HF cache on shared FS (required for rank0 tokenization cache)
export HF_HOME=/mnt/sharedfs/nebius-demoday-test/.cache/huggingface
mkdir -p "$HF_HOME"

# ---- rendezvous ----
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE="${NODES[0]}"

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc \
  "ip -4 -o addr show dev eth0 | awk '{print \$4}' | cut -d/ -f1")
MASTER_PORT=29500
RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
export MASTER_ADDR MASTER_PORT RDZV_ENDPOINT

echo "NODES=${NODES[*]}"
echo "HEAD_NODE=$HEAD_NODE"
echo "RDZV_ENDPOINT=$RDZV_ENDPOINT"

nvidia-smi -L

# Launch one torchrun per node
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  set -x

  cd /mnt/sharedfs/nebius-demoday-test
  source training/env.sh || true
  source /mnt/sharedfs/nebius-demoday-test/.venv/bin/activate

  echo "Running on $(hostname) SLURM_NODEID=$SLURM_NODEID"
  echo "IP: $(hostname -I)"
  nvidia-smi -L

  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint="$RDZV_ENDPOINT" \
    training/src/train_sft_fsdp_multinode.py \
      --run_name sft_fsdp_2node_smoke \
      --max_steps 10 \
      --seq_len 512 \
      --bsz 1 \
      --grad_accum 1 \
      --bf16 True
'

