#!/bin/bash
#SBATCH --job-name=train_ddp_2node_lora
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

source training/env.sh || true
source /mnt/sharedfs/nebius-demoday-test/.venv/bin/activate

# ---- NCCL / networking ----
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# HF cache on shared FS (important for multi-node)
export HF_HOME=/mnt/sharedfs/nebius-demoday-test/.cache/huggingface
mkdir -p "$HF_HOME"

# rendezvous
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_ADDR="${NODES[0]}"
MASTER_PORT=29500

echo "NODES=${NODES[*]}"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# launch one torchrun per node
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  echo "Running on $(hostname)  SLURM_NODEID=$SLURM_NODEID"

  torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR"':'"$MASTER_PORT"' \
    training/src/train_sft_ddp_min_2node.py \
      --run_name sft_ddp_2node_lora_smoke \
      --max_steps 10 \
      --seq_len 512 \
      --bsz 1 \
      --grad_accum 1 \
      --bf16 True \
      --gradient_checkpointing True \
      --use_lora \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05
'

