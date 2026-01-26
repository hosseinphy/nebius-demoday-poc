#!/bin/bash
#SBATCH --job-name=train_1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/sharedfs/nebius-demoday-test
#SBATCH --output=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.err

set -euo pipefail
set -x

# Ensure log dir exists
mkdir -p /mnt/sharedfs/nebius-demoday-test/slurm-logs

# Repo env (keep if you already rely on it for paths)
source training/env.sh

VENV="/mnt/sharedfs/nebius-demoday-test/.venv"
source "$VENV/bin/activate"


# Sanity prints
hostname
pwd
which python
python -V

# Run single-GPU training smoke
python training/src/train_sft_min.py \
  --run_name sft_1gpu_smoke \
  --max_steps 10 \
  --seq_len 512


