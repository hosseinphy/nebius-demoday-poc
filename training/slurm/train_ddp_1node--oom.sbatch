#!/bin/bash
#SBATCH --job-name=train_ddp_1node
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/sharedfs/nebius-demoday-test
#SBATCH --output=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.err

set -euo pipefail
set -x

mkdir -p /mnt/sharedfs/nebius-demoday-test/slurm-logs

# env + venv
source training/env.sh || true
source /mnt/sharedfs/nebius-demoday-test/.venv/bin/activate

# debug (useful when something breaks)
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# torchrun uses 8 ranks for 8 GPUs
torchrun \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  training/src/train_sft_ddp_min.py \
    --run_name sft_ddp_1node_smoke \
    --max_steps 10 \
    --seq_len 512

