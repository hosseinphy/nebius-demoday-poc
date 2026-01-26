#!/usr/bin/env bash
set -euo pipefail

# ==================================
# NCCL + monitoring harness (cleaned)
# ==================================
#
# Runs an NCCL test under Slurm and logs:
# - NCCL stdout/stderr
# - GPU metrics (nvidia-smi sampled)
# - CPU/RAM metrics (/proc-based)
#
# Results are written to:
#   validation/results/nccl/run_<RUN_ID>/
#
# Assumptions:
# - Slurm partition "gpu"
# - Shared FS at /mnt/sharedfs
# - nccl-tests at /mnt/sharedfs/tools/nccl-tests/build/all_reduce_perf

# --------------------
# Repo / result layout
# --------------------
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
VALIDATION_ROOT="${REPO_ROOT}/validation"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${VALIDATION_ROOT}/results/nccl/run_${RUN_ID}"

mkdir -p "${OUT_DIR}"

echo "RUN_ID=${RUN_ID}"
echo "Results directory: ${OUT_DIR}"
echo

# ---------
# Arguments
# ---------
NODES="${1:-1}"              # 1 or 2
NTASKS_TOTAL="${2:-8}"       # 8 (1 node), 16 (2 nodes)
PARTITION="${PARTITION:-gpu}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

# NCCL test params
NCCL_BIN="${NCCL_BIN:-/mnt/sharedfs/tools/nccl-tests/build/all_reduce_perf}"
NCCL_B="${NCCL_B:-8}"
NCCL_E="${NCCL_E:-1G}"
NCCL_F="${NCCL_F:-2}"
NCCL_G="${NCCL_G:-1}"

# Sampling interval (seconds)
INTERVAL="${INTERVAL:-1}"

# ----------------
# Basic validation
# ----------------
if [[ ! -x "${NCCL_BIN}" ]]; then
  echo "ERROR: NCCL binary not found: ${NCCL_BIN}" >&2
  exit 1
fi

# ----------------
# Metadata
# ----------------
{
  echo "run_id=${RUN_ID}"
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "user=$(whoami)"
  echo "partition=${PARTITION}"
  echo "nodes=${NODES}"
  echo "ntasks_total=${NTASKS_TOTAL}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "cpus_per_task=${CPUS_PER_TASK}"
  echo "nccl_bin=${NCCL_BIN}"
  echo "nccl_args=-b ${NCCL_B} -e ${NCCL_E} -f ${NCCL_F} -g ${NCCL_G}"
  echo "interval_sec=${INTERVAL}"
  echo "slurm_jobid=${SLURM_JOB_ID:-NA}"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} | tee "${OUT_DIR}/meta.txt" >/dev/null

# --------------------------------
# Per-node monitoring loop snippet
# --------------------------------
read -r -d '' MONITOR_SNIPPET <<'EOF'
set -euo pipefail
OUT_DIR="$1"
INTERVAL="$2"

HOST="$(hostname -s)"

GPU_OUT="${OUT_DIR}/node_${HOST}_gpu.csv"
CPU_OUT="${OUT_DIR}/node_${HOST}_cpu.csv"
TOP_OUT="${OUT_DIR}/node_${HOST}_top.txt"

echo "ts_utc,host,gpu,util_gpu,util_mem,mem_used_mb,mem_total_mb,temp_c,power_w,pcie_rx,pcie_tx" > "${GPU_OUT}"
echo "ts_utc,host,load1,load5,load15,mem_total_kb,mem_avail_kb,swap_total_kb,swap_free_kb,cpu_user,cpu_system,cpu_idle" > "${CPU_OUT}"

while true; do
  TS="$(date -u +%FT%TZ)"

  read -r L1 L5 L15 _ < /proc/loadavg
  MEM_TOTAL="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
  MEM_AVAIL="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
  SWAP_TOTAL="$(awk '/SwapTotal/ {print $2}' /proc/meminfo)"
  SWAP_FREE="$(awk '/SwapFree/ {print $2}' /proc/meminfo)"

  read -r _ U N S I IO IRQ SIRQ ST _ < /proc/stat
  TOT=$((U+N+S+I+IO+IRQ+SIRQ+ST))
  cpu_user=$(( (U+N) * 100 / (TOT==0?1:TOT) ))
  cpu_system=$(( (S+IRQ+SIRQ) * 100 / (TOT==0?1:TOT) ))
  cpu_idle=$(( (I+IO) * 100 / (TOT==0?1:TOT) ))

  echo "${TS},${HOST},${L1},${L5},${L15},${MEM_TOTAL},${MEM_AVAIL},${SWAP_TOTAL},${SWAP_FREE},${cpu_user},${cpu_system},${cpu_idle}" >> "${CPU_OUT}"

  nvidia-smi \
    --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,pcie.rx_throughput,pcie.tx_throughput \
    --format=csv,noheader,nounits \
  | awk -v ts="${TS}" -v host="${HOST}" -F', *' \
    '{print ts "," host "," $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 "," $9}' \
    >> "${GPU_OUT}" || true

  sleep "${INTERVAL}"
done
EOF

# -----------------------------
# 1) Start monitors (background)
# -----------------------------
echo "Starting monitors on ${NODES} node(s)..."
MONITOR_LOG="${OUT_DIR}/monitor_launcher.log"

srun -p "${PARTITION}" \
     --nodes="${NODES}" \
     --ntasks-per-node=1 \
     --cpus-per-task=1 \
     --gres=gpu:0 \
     --exclusive \
     bash -lc "${MONITOR_SNIPPET} '${OUT_DIR}' '${INTERVAL}'" \
     > "${MONITOR_LOG}" 2>&1 &
MONITOR_PID=$!

sleep 3

# -------------------------
# 2) Run NCCL test
# -------------------------
echo "Running NCCL all_reduce_perf..."
NCCL_LOG="${OUT_DIR}/nccl.log"

NTASKS_PER_NODE=$((NTASKS_TOTAL / NODES))

srun -p "${PARTITION}" \
     --nodes="${NODES}" \
     --ntasks-per-node="${NTASKS_PER_NODE}" \
     --gres="gpu:${GPUS_PER_NODE}" \
     --cpus-per-task="${CPUS_PER_TASK}" \
     --exclusive \
     --export=ALL \
     "${NCCL_BIN}" -b "${NCCL_B}" -e "${NCCL_E}" -f "${NCCL_F}" -g "${NCCL_G}" \
     2>&1 | tee "${NCCL_LOG}"

# -------------------------
# 3) Stop monitors
# -------------------------
echo "Stopping monitors..."
kill "${MONITOR_PID}" >/dev/null 2>&1 || true

echo
echo "NCCL test completed successfully."
echo "Results written to: ${OUT_DIR}"

