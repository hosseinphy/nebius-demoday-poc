#!/bin/bash
#SBATCH --job-name=cuda-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --chdir=/mnt/sharedfs/nebius-demoday-test
#SBATCH --output=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.err

set -euo pipefail
set -x

# Ensure log dir exists (safe even if already there)
mkdir -p /mnt/sharedfs/nebius-demoday-test/slurm-logs

# Activate your repo env (if it sets paths)
source training/env.sh || true

# Activate repo venv (shared)
VENV="/mnt/sharedfs/nebius-demoday-test/.venv"
if [ -d "$VENV" ]; then
  source "$VENV/bin/activate"
fi

hostname
pwd
nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("gpu0:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda_matmul_ok:", y.mean().item())
PY

