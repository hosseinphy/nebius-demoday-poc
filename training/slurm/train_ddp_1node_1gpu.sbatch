#!/bin/bash
#SBATCH --job-name=ddp1gpu-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/shared/ubuntu/nebius-demoday
#SBATCH --output=/mnt/shared/ubuntu/nebius-demoday/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/shared/ubuntu/nebius-demoday/slurm-logs/%x-%j.err

set -euo pipefail
set -x

source training/env.sh
source .venv/bin/activate

torchrun \
  --nproc_per_node=1 \
  training/src/train_sft_ddp_min.py \
    --run_name sft_ddp_1gpu_test \
    --max_steps 10 \
    --seq_len 512

