#!/bin/bash
#SBATCH -J fsdp-2node-ib
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -p gpu
#SBATCH -t 00:30:00
#SBATCH -o /mnt/sharedfs/nebius-demoday-test/training/slurm-logs/train_fsdp_2node_ib-%j.out
#SBATCH -e /mnt/sharedfs/nebius-demoday-test/training/slurm-logs/train_fsdp_2node_ib-%j.err

set -euo pipefail

# Repo root MUST match where src/ actually lives
REPO="/mnt/sharedfs/nebius-demoday-test/training"
cd "$REPO"

LOG_DIR="$REPO/slurm-logs"
mkdir -p "$LOG_DIR"

# venv at project root (adjust if different)
VENV_DIR="/mnt/sharedfs/nebius-demoday-test/.venv"
TORCHRUN="$VENV_DIR/bin/torchrun"
PY="$VENV_DIR/bin/python"

# Training entrypoint relative to REPO
TRAIN_PY="src/train_sft_fsdp_multinode_ib.py"
TRAIN_ARGS=""

# rendezvous
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD_NODE="${NODES[0]}"
export MASTER_ADDR="$HEAD_NODE"
export MASTER_PORT="${MASTER_PORT:-20046}"
export RDZV_ID="${SLURM_JOB_ID}"

# control-plane sockets
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

# NCCL / IB
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,IB}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

NNODES="${SLURM_JOB_NUM_NODES}"
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"

echo "JobID=${SLURM_JOB_ID}"
echo "REPO=${REPO}"
echo "TRAIN=${TRAIN_PY}"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT} RDZV_ID=${RDZV_ID}"
echo "NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo

# sanity on worker0
srun -N1 -w "$HEAD_NODE" bash -lc "set -e; hostname; ip -br -4 addr; [ -x '$PY' ]; [ -x '$TORCHRUN' ]; ls -lah '$REPO/$TRAIN_PY'"

# launch: ONE torchrun per node (node_rank from SLURM_PROCID)
srun --ntasks="$NNODES" --ntasks-per-node=1 bash -lc "
set -euo pipefail
NODE_RANK=\"\$SLURM_PROCID\"
echo \"HOST=\$(hostname) NODE_RANK=\$NODE_RANK\"
'$TORCHRUN' \
  --nnodes='$NNODES' \
  --nproc_per_node='$NPROC_PER_NODE' \
  --node_rank=\"\$NODE_RANK\" \
  --rdzv_backend=c10d \
  --rdzv_id='$RDZV_ID' \
  --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' \
  '${TRAIN_PY}' ${TRAIN_ARGS}
"

