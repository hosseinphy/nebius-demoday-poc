#!/bin/bash
#SBATCH --job-name=train_fsdp_1node
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/sharedfs/nebius-demoday-test
#SBATCH --output=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.err

set -euo pipefail
set -x

mkdir -p /mnt/sharedfs/nebius-demoday-test/slurm-logs

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nvidia-smi -L

srun --ntasks=1 bash -lc '
  set -euo pipefail
  set -x

  cd /mnt/sharedfs/nebius-demoday-test
  source training/env.sh || true
  source /mnt/sharedfs/nebius-demoday-test/.venv/bin/activate

  echo "Running on $(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi -L

  torchrun \
    --standalone \
    --nproc_per_node=8 \
    training/src/train_sft_fsdp_multinode.py \
      --run_name sft_fsdp_1node_fullmodel_smoke \
      --max_steps 10 \
      --seq_len 512 \
      --bsz 1 \
      --grad_accum 1 \
      --bf16 True
'

