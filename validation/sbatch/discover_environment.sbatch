#!/usr/bin/env bash
#SBATCH --job-name=env-discovery
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/sharedfs/nebius-demoday-test/validation/results/environment/%x-%j.out
#SBATCH --error=/mnt/sharedfs/nebius-demoday-test/validation/results/environment/%x-%j.err

set -euo pipefail

OUTDIR="/mnt/sharedfs/nebius-demoday-test/validation/results/environment"
mkdir -p "$OUTDIR"

echo "=== Time (UTC) ==="
date -u +%FT%TZ

echo
echo "=== Hostname ==="
hostname -s

echo
echo "=== Slurm basics ==="
scontrol ping || true
sinfo -N -l || true
scontrol show node "$(hostname -s)" | egrep 'NodeName=|Arch=|CPU|RealMemory|State=|Gres=|CfgTRES=|Partitions=' || true

echo
echo "=== IP + routes ==="
ip -br -4 addr || true
ip -4 route || true

echo
echo "=== GPU information (nvidia-smi) ==="
nvidia-smi || true

echo
echo "=== GPU topology ==="
nvidia-smi topo -m || true

echo
echo "=== Shared filesystem check (/mnt/sharedfs) ==="
findmnt /mnt/sharedfs || true
df -hT /mnt/sharedfs || true
ls -lah /mnt/sharedfs || true

echo
echo "=== Filesystems (df -hT) ==="
df -hT || true

echo
echo "=== Mounts (filtered) ==="
findmnt -o TARGET,SOURCE,FSTYPE,SIZE,OPTIONS \
| egrep -v "tmpfs|overlay|proc|sysfs|cgroup|devtmpfs|mqueue|devpts|securityfs|pstore|bpf|tracefs" || true

echo
echo "=== InfiniBand devices ==="
ls -lah /sys/class/infiniband 2>/dev/null || echo "No IB devices visible"

echo
echo "=== ibstat (if available) ==="
command -v ibstat >/dev/null 2>&1 && ibstat || echo "ibstat not available"

echo
echo "=== Net devices ==="
ip -br link || true

echo
echo "DONE"
echo "Output path:"
echo "  $SLURM_JOB_OUTPUT"

