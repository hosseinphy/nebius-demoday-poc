# Nebius GPU Cluster Validation & Distributed Training Experiments

## Overview

This repository documents a **systematic validation of a multi-node, multi-GPU NVIDIA H100 cluster on Nebius**, followed by **incremental distributed training experiments using PyTorch**.

The goal is **not** to train a production model, but to:

- Verify **hardware correctness**
- Validate **Slurm scheduling and resource binding**
- Confirm **shared filesystem behavior**
- Measure **NCCL collective performance**
- Demonstrate **single-node and multi-node distributed training readiness**
  - DDP (Data Distributed Parallel)
  - FSDP (Fully Sharded Data Parallel)

This repo is designed as:
- A **reproducible infrastructure validation record**
- A **reference for bringing up distributed training on a fresh cluster**
- A **Demo Day / technical deep-dive artifact**

---

## Problem Statement

Before running large-scale LLM or ML workloads on a new GPU cluster, we must answer:

1. Are **GPUs visible, correctly bound, and isolated** per task?
2. Does **Slurm schedule GPUs, CPUs, and memory correctly**?
3. Is the **shared filesystem truly shared across nodes**?
4. Are **interconnects (NVLink / InfiniBand)** usable by NCCL?
5. Can **PyTorch distributed training** run:
   - On 1 GPU
   - On 8 GPUs (single node)
   - Across 2 nodes (multi-node)
6. Are **failure modes observable and diagnosable**?

This repository answers those questions **step by step**, with scripts, logs, and documentation.

---

## Repository Structure

```
.
├── docs/
│   ├── 01-environment.md
│   ├── 02-infrastructure-validation.md
│   ├── 03-training-single-node.md
│   ├── 04-training-ddp-design.md
│   ├── 05-training-fsdp-design.md
│   └── 05-training-fsdp-multiple-node.md
│
├── infrastructure-validation/
│   ├── scripts/
│   │   ├── discover-environment.sbatch
│   │   ├── smoke_test.sh
│   │   ├── nccl_scaling.sh
│   │   └── nccl_run_with_monitoring.sh
│   └── README.md
│
├── training/
│   ├── src/
│   │   ├── train_single_gpu.py
│   │   ├── train_ddp_single_node.py
│   │   ├── train_ddp_multi_node.py
│   │   ├── train_fsdp_single_node.py
│   │   └── train_fsdp_multi_node.py
│   │
│   ├── slurm/
│   │   ├── run_1gpu.sbatch
│   │   ├── run_ddp_1node.sbatch
│   │   ├── run_ddp_2node.sbatch
│   │   ├── run_fsdp_1node.sbatch
│   │   └── run_fsdp_2node.sbatch
│
├── validation/
│   └── results/        # ignored by git (generated outputs)
│
├── slurm-logs/         # ignored by git
├── .gitignore
└── README.md
```

---

## Phase 1 — Environment Discovery

**Location:**  
`infrastructure-validation/scripts/discover-environment.sbatch`  
`docs/01-environment.md`

What is validated:

- Node topology
- CPU layout (sockets, cores, SMT)
- GPU model, driver, CUDA version
- MIG status
- Filesystem mounts
- InfiniBand visibility

This establishes **ground truth** for the cluster.

---

## Phase 2 — Infrastructure Smoke Tests

**Location:**  
`infrastructure-validation/scripts/smoke_test.sh`  
`docs/02-infrastructure-validation.md`

Validates:

- Slurm controller health
- Node availability
- Partition configuration
- Shared filesystem consistency across nodes
- Basic `srun` / `sbatch` behavior
- GPU visibility inside Slurm jobs

This phase answers:  
> *Can Slurm reliably schedule work on this cluster?*

---

## Phase 3 — NCCL & Interconnect Validation

**Location:**  
`infrastructure-validation/scripts/nccl_scaling.sh`  
`validation/results/nccl_scaling/`

Validated scenarios:

1. **1 GPU, 1 node**
2. **8 GPUs, 1 node (DDP-style)**
3. **16 GPUs, 2 nodes (DDP-style)**

Metrics collected:

- AllReduce latency
- Algorithm bandwidth
- Bus bandwidth
- GPU utilization (optional monitoring)

This confirms:
- Correct NCCL initialization
- GPU-to-GPU communication
- Readiness for distributed training

---

## Phase 4 — Training: Single GPU Baseline

**Location:**  
`training/src/train_single_gpu.py`  
`training/slurm/run_1gpu.sbatch`  
`docs/03-training-single-node.md`

Purpose:

- Establish a known-good baseline
- Validate dataloading, optimizer, checkpoint logic
- Confirm CUDA correctness

---

## Phase 5 — Training: DDP (Data Parallel)

**Location:**  
`training/src/train_ddp_single_node.py`  
`training/src/train_ddp_multi_node.py`  
`training/slurm/run_ddp_*`  
`docs/04-training-ddp-design.md`

Validated:

- `torchrun` + Slurm integration
- Rank/world-size correctness
- DistributedSampler behavior
- Failure propagation when a rank exits
- CPU/GPU binding correctness

This phase demonstrates **production-style data parallelism**.

---

## Phase 6 — Training: FSDP (Model Sharding)

**Location:**  
`training/src/train_fsdp_single_node.py`  
`training/src/train_fsdp_multi_node.py`  
`training/slurm/run_fsdp_*`  
`docs/05-training-fsdp-*.md`

Validated:

- FSDP wrapping policies
- Parameter sharding
- CPU offload (where applicable)
- Multi-node synchronization behavior
- Memory scaling characteristics

This phase demonstrates **large-model readiness**.

---

## Results & Artifacts

All generated outputs are written to:

```
validation/results/
training/results/
slurm-logs/
```

These directories are **intentionally excluded from git** via `.gitignore`.

The repository focuses on:
- **Reproducible procedures**
- **Clear documentation**
- **Failure analysis**
—not raw data dumps.

---

## What This Repository Demonstrates

✔ End-to-end cluster bring-up  
✔ Slurm + GPU correctness  
✔ NCCL scaling behavior  
✔ DDP and FSDP training readiness  
✔ Realistic failure modes  
✔ Production-style validation discipline  

---

## Intended Audience

- GPU infrastructure engineers  
- ML platform engineers  
- Distributed systems engineers  
- Technical reviewers (Demo Day, interviews)  

---

## How to Use This Repo

1. Read `docs/01-environment.md`
2. Run infra validation scripts
3. Review NCCL results
4. Execute training phases incrementally
5. Use docs as a **validation playbook**

---

## Status

✔ Infrastructure validated  
✔ Single-node training validated  
✔ Multi-node distributed training validated  
✔ Ready for scale-up workloads  



