#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-gpu}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"

WORKDIR="${WORKDIR:-/mnt/sharedfs/tools/nccl-tests}"
BIN="${BIN:-./build/all_reduce_perf}"

MIN_BYTES="${MIN_BYTES:-32M}"
MAX_BYTES="${MAX_BYTES:-1G}"
FACTOR="${FACTOR:-2}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-3}"

DO_2NODE="${DO_2NODE:-0}"
PIN_NODES="${PIN_NODES:-}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
VALIDATION_ROOT="${REPO_ROOT}/validation"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTDIR="${VALIDATION_ROOT}/results/nccl_scaling/run_${RUN_ID}"
mkdir -p "${OUTDIR}"

echo "Writing results to: ${OUTDIR}"
echo "Using nccl-tests workdir: ${WORKDIR}"
echo "Partition: ${PARTITION}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "CPUS_PER_TASK: ${CPUS_PER_TASK}"
echo

if [[ ! -x "${WORKDIR}/build/all_reduce_perf" ]]; then
  echo "ERROR: ${WORKDIR}/build/all_reduce_perf not found or not executable." >&2
  exit 1
fi

{
  echo "run_id=${RUN_ID}"
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "partition=${PARTITION}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "cpus_per_task=${CPUS_PER_TASK}"
  echo "workdir=${WORKDIR}"
  echo "bin=${BIN}"
  echo "args=-b ${MIN_BYTES} -e ${MAX_BYTES} -f ${FACTOR} -g 1 -w ${WARMUP} -n ${ITERS}"
  echo "pin_nodes=${PIN_NODES:-none}"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "${OUTDIR}/meta.txt"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"

summarize_log () {
  local logfile="$1"
  awk '
    $1 ~ /^[0-9]+$/ && NF >= 13 {
      size=$1;
      oop_time=$6; oop_algbw=$7; oop_busbw=$8;
      ip_time=$11; ip_algbw=$12; ip_busbw=$13;
      last=size "  oop_time(us)=" oop_time "  oop_algbw(GB/s)=" oop_algbw "  oop_busbw(GB/s)=" oop_busbw \
                "  ip_time(us)=" ip_time "  ip_algbw(GB/s)=" ip_algbw "  ip_busbw(GB/s)=" ip_busbw;
    }
    END { if (last!="") print last; }
  ' "${logfile}" || true
}

submit_case () {
  local case_name="$1"
  local nodes="$2"
  local ntasks="$3"
  local ntasks_per_node="$4"
  local gpus_per_node="$5"

  local logfile="${OUTDIR}/${case_name}.log"
  local sbout="${OUTDIR}/${case_name}.sbatch.out"

  local wopt=""
  if [[ -n "${PIN_NODES}" ]]; then
    wopt="--nodelist=${PIN_NODES}"
  fi

  echo "=============================="
  echo "CASE: ${case_name}"
  echo "LOG : ${logfile}"
  echo "SB  : ${sbout}"
  echo "nodes=${nodes} ntasks=${ntasks} ntasks_per_node=${ntasks_per_node} gpus_per_node=${gpus_per_node} cpus_per_task=${CPUS_PER_TASK}"
  echo "=============================="

  local wrap_cmd
  wrap_cmd="$(cat <<EOF
bash -lc '
set -euo pipefail
cd "${WORKDIR}"

echo "hostlist: \$(scontrol show hostnames "\${SLURM_JOB_NODELIST}")"
echo "time_utc: \$(date -u +%FT%TZ)"
echo "SLURM_JOB_CPUS_PER_NODE=\${SLURM_JOB_CPUS_PER_NODE:-NA}"
echo "SLURM_NTASKS=\${SLURM_NTASKS:-NA}"
echo "SLURM_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK:-NA}"
echo "SLURM_GPUS_ON_NODE=\${SLURM_GPUS_ON_NODE:-NA}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"

# IMPORTANT: do NOT pass --mpi=pmix_v3; this cluster does not have that plugin.
srun --ntasks=${ntasks} --ntasks-per-node=${ntasks_per_node} \
     --cpus-per-task=${CPUS_PER_TASK} \
     ${BIN} -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${FACTOR}" -g 1 -w "${WARMUP}" -n "${ITERS}"
'
EOF
)"

  local jid
  jid=$(sbatch --parsable -J "${case_name}" -p "${PARTITION}" -N "${nodes}" ${wopt} \
               --ntasks="${ntasks}" --ntasks-per-node="${ntasks_per_node}" \
               --cpus-per-task="${CPUS_PER_TASK}" \
               --gres="gpu:${gpus_per_node}" \
               --output="${sbout}" --time=00:15:00 --mem=0 \
               --wrap="${wrap_cmd}")

  echo "Submitted JobID=${jid}"
  while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do sleep 2; done

  {
    echo "===== SBATCH STDOUT (${case_name}) ====="
    cat "${sbout}" || true
    echo
    echo "===== SUMMARY LINE (parsed last row) ====="
    summarize_log "${sbout}" || true
  } > "${logfile}"

  echo "Done: ${logfile}"
  echo
}

submit_case "01_1node_1gpu" 1 1 1 1
submit_case "02_1node_${GPUS_PER_NODE}gpu_ddpstyle" 1 "${GPUS_PER_NODE}" "${GPUS_PER_NODE}" "${GPUS_PER_NODE}"

if [[ "${DO_2NODE}" == "1" ]]; then
  NTASKS_TOTAL=$((2 * GPUS_PER_NODE))
  submit_case "03_2node_${NTASKS_TOTAL}gpu_ddpstyle" 2 "${NTASKS_TOTAL}" "${GPUS_PER_NODE}" "${GPUS_PER_NODE}"
fi

echo "=============================="
echo "SUMMARY (tail of each combined log)"
echo "=============================="
for f in "${OUTDIR}"/*.log; do
  echo "--- $(basename "$f") ---"
  tail -n 30 "$f" || true
  echo
done

echo "Done. Results are in: ${OUTDIR}"

