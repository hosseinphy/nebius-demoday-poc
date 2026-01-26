#!/usr/bin/env bash
set -euo pipefail

P="${P:-gpu}"
SHARED_ROOT="${SHARED_ROOT:-/mnt/sharedfs/nebius-demoday-test}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-${SHARED_ROOT}/validation/results/infra-smoke-${RUN_ID}}"
mkdir -p "$OUT"

echo "Writing results to: $OUT"
echo "Partition: $P"
echo

phase () {
  local name="$1" file="$2"
  echo "=== ${name} ===" | tee "${OUT}/${file}"
}

run_block () {
  local file="$1"; shift
  { "$@"; } |& tee -a "${OUT}/${file}"
}

# Submit a tiny sbatch job and wait, returning output file path
submit_and_wait () {
  local name="$1" out_file="$2" wrap_cmd="$3"
  local jid
  jid=$(sbatch --parsable -J "$name" -p "$P" --mem=64M -t 00:05:00 \
        --output="$out_file" --wrap="$wrap_cmd")
  echo "$jid"
  while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 2; done
}

phase "Phase 1: Slurm basics" "phase1_slurm_basics.txt"
run_block "phase1_slurm_basics.txt" bash -lc "
  set -e
  scontrol ping
  echo
  sinfo -N -l
  echo
  sinfo -R || true
  echo
  scontrol show partition ${P} || true
"

phase "Phase 2: Shared filesystem (/mnt/sharedfs)" "phase2_sharedfs.txt"
run_block "phase2_sharedfs.txt" bash -lc "
  set -e
  findmnt /mnt/sharedfs || true
  df -h /mnt/sharedfs || true
  echo

  rm -f '${OUT}/fs_test.'* 2>/dev/null || true

  # Use sbatch so this works even without an interactive allocation
  J1=\$(sbatch --parsable -J fswrite1 -p ${P} --mem=64M -t 00:02:00 \
      --output='${OUT}/_fswrite1-%j.out' \
      --wrap=\"echo shared-test-1 from \\\$(hostname) at \\\$(date -u +%FT%TZ) > '${OUT}/fs_test.\\\$(hostname).txt'\")
  while squeue -j \"\$J1\" -h 2>/dev/null | grep -q .; do sleep 2; done

  # One task per node (2 nodes) – proves shared FS across workers
  J2=\$(sbatch --parsable -J fswrite2 -p ${P} -N 2 --ntasks-per-node=1 --mem=64M -t 00:03:00 \
      --output='${OUT}/_fswrite2-%j.out' \
      --wrap=\"srun --ntasks=2 --nodes=2 bash -lc 'echo shared-test-2 from \\\$(hostname) rank=\\\$SLURM_PROCID at \\\$(date -u +%FT%TZ) > \\\"${OUT}/fs_test.\\\$(hostname).rank\\\$SLURM_PROCID.txt\\\"'\")
  while squeue -j \"\$J2\" -h 2>/dev/null | grep -q .; do sleep 2; done

  echo
  echo 'Files created on shared FS:'
  ls -lah '${OUT}'/fs_test.*.txt || true

  echo
  echo 'Contents:'
  for f in '${OUT}'/fs_test.*.txt; do
    echo \"--- \$f\"
    cat \"\$f\" || true
  done
"

phase "Phase 3: Slurm runtime" "phase3_slurm_runtime.txt"
run_block "phase3_slurm_runtime.txt" bash -lc "
  set -e

  # Single-node srun
  J=\$(sbatch --parsable -J srun1 -p ${P} --mem=64M -t 00:03:00 \
      --output='${OUT}/_srun1-%j.out' \
      --wrap='srun --ntasks=1 hostname')
  while squeue -j \"\$J\" -h 2>/dev/null | grep -q .; do sleep 2; done
  cat '${OUT}/_srun1-'\"\$J\"'.out' || true
  echo

  # Two tasks same node
  J=\$(sbatch --parsable -J srun2 -p ${P} --mem=64M -t 00:03:00 \
      --output='${OUT}/_srun2-%j.out' \
      --wrap=\"srun --ntasks=2 bash -lc 'echo rank=\\\$SLURM_PROCID host=\\\$(hostname)'\")
  while squeue -j \"\$J\" -h 2>/dev/null | grep -q .; do sleep 2; done
  cat '${OUT}/_srun2-'\"\$J\"'.out' || true
  echo

  # Two nodes, one task per node
  J=\$(sbatch --parsable -J srun2n -p ${P} -N 2 --ntasks-per-node=1 --mem=64M -t 00:03:00 \
      --output='${OUT}/_srun2n-%j.out' \
      --wrap=\"srun --nodes=2 --ntasks=2 bash -lc 'echo rank=\\\$SLURM_PROCID host=\\\$(hostname)'\")
  while squeue -j \"\$J\" -h 2>/dev/null | grep -q .; do sleep 2; done
  cat '${OUT}/_srun2n-'\"\$J\"'.out' || true
  echo

  # Cancel test
  K=\$(sbatch --parsable -J canceltest -p ${P} --mem=64M -t 00:05:00 \
      --output='${OUT}/cancel-%j.out' \
      --wrap='sleep 300')
  sleep 2
  scancel \"\$K\" || true
  squeue -j \"\$K\" || true
"

phase "Phase 4: GPU" "phase4_gpu.txt"
run_block "phase4_gpu.txt" bash -lc "
  set -e

  J=\$(sbatch --parsable -J gpucheck -p ${P} --mem=64M --gres=gpu:1 -t 00:03:00 \
      --output='${OUT}/_gpu-%j.out' \
      --wrap=\"nvidia-smi; echo; echo host=\\\$(hostname) CVD=\\\${CUDA_VISIBLE_DEVICES:-unset}; nvidia-smi -L\")
  while squeue -j \"\$J\" -h 2>/dev/null | grep -q .; do sleep 2; done
  cat '${OUT}/_gpu-'\"\$J\"'.out' || true
"

phase "Phase 5: InfiniBand quick check" "phase5_ib.txt"
run_block "phase5_ib.txt" bash -lc "
  set -e

  J=\$(sbatch --parsable -J ibcheck -p ${P} -N 2 --ntasks-per-node=1 --mem=64M -t 00:03:00 \
      --output='${OUT}/_ib-%j.out' \
      --wrap=\"srun --nodes=2 --ntasks=2 bash -lc 'echo host=\\\$(hostname); ls /sys/class/infiniband 2>/dev/null || echo none'; echo; (command -v ibstat >/dev/null 2>&1 && ibstat) || echo ibstat-not-installed\")
  while squeue -j \"\$J\" -h 2>/dev/null | grep -q .; do sleep 2; done
  cat '${OUT}/_ib-'\"\$J\"'.out' || true
"

echo
echo "DONE"
echo "Results folder: $OUT"

