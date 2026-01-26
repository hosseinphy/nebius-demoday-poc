#!/usr/bin/env bash

# Activate repo venv (shared)
VENV="/mnt/sharedfs/nebius-demoday-test/.venv"
if [ -d "$VENV" ]; then
  source "$VENV/bin/activate"
fi


set -euo pipefail

REPO_ROOT=/mnt/sharedfs/nebius-demoday-test
export RUNS_ROOT=$REPO_ROOT/results/training
export HF_HOME=$REPO_ROOT/.cache/huggingface

mkdir -p "$RUNS_ROOT" "$HF_HOME"

echo "REPO_ROOT=$REPO_ROOT"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "HF_HOME=$HF_HOME"




