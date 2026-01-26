
# Infrastructure Validation Report
**Nebius GPU Cluster – Fabric 2**

This document consolidates all infrastructure validation runs performed on the Nebius
GPU cluster (Fabric 2). The goal is to establish confidence in:

- Slurm scheduling correctness
- Shared filesystem consistency
- GPU visibility and health
- NCCL collective performance (single- and multi-node)
- Readiness for PyTorch DDP and FSDP training

All validation runs write results to a shared, reproducible directory under:

```

validation/results/

```

Each subsection below references:
- the script used
- the run output location
- the conclusion drawn
```

---

## 1. Slurm & Cluster Baseline Validation

### Purpose

Verify that:

* Slurm controller and compute nodes are healthy
* GPU partition is configured correctly
* Nodes are idle and schedulable

### Script

```
validation/scripts/30_smoke/smoke_test.sh
```

### Relevant Output

```
validation/results/infra-smoke-<RUN_ID>/phase1_slurm_basics.txt
```

### Checks Performed

* `scontrol ping`
* `sinfo -N -l`
* `scontrol show partition gpu`

### Outcome

* Slurm controller reachable
* Two GPU nodes (`worker0`, `worker1`) available and idle
* GPU partition configured as default and active

**Cluster baseline is healthy**

---

## 2. Shared Filesystem Validation

### Purpose

Verify that the shared filesystem:

* Is mounted consistently on all nodes
* Supports concurrent writes from multiple nodes
* Is suitable for checkpoints, logs, and training artifacts

### Script

```
validation/scripts/30_smoke/smoke_test.sh
```

### Relevant Output

```
validation/results/infra-smoke-<RUN_ID>/phase2_sharedfs.txt
```

### Checks Performed

* `findmnt /mnt/sharedfs`
* `df -h /mnt/sharedfs`
* Write tests from:

  * single node
  * two nodes simultaneously
* Read-back verification from login node

### Outcome

* `/mnt/sharedfs` mounted as virtiofs
* Files written on compute nodes visible immediately on login node
* No inconsistencies observed

**Shared filesystem is functional and consistent**

---

## 3. Slurm Runtime Behavior

### Purpose

Verify Slurm runtime semantics:

* `srun` task placement
* multi-task and multi-node launches
* job submission, completion, and cancellation

### Script

```
validation/scripts/30_smoke/smoke_test.sh
```

### Relevant Output

```
validation/results/infra-smoke-<RUN_ID>/phase3_slurm_runtime.txt
```

### Checks Performed

* `srun` single task
* `srun` multi-task (same node)
* `srun` multi-node
* `sbatch` submit / wait
* `scancel` cancellation

### Outcome

* Task ranks and hostnames matched expectations
* Multi-node placement correct
* Job lifecycle operations behaved correctly

**Slurm runtime behavior validated**

---

## 4. GPU Visibility & Health

### Purpose

Verify that:

* GPUs are visible to Slurm jobs
* CUDA device assignment is correct
* GPUs are usable inside scheduled jobs

### Script

```
validation/scripts/30_smoke/smoke_test.sh
```

### Relevant Output

```
validation/results/infra-smoke-<RUN_ID>/phase4_gpu.txt
```

### Checks Performed

* `nvidia-smi`
* `nvidia-smi -L`
* CUDA_VISIBLE_DEVICES inspection inside jobs

### Outcome

* All GPUs visible and enumerated correctly
* CUDA device assignment matches Slurm allocation

**GPU health and visibility confirmed**

---

## 5. InfiniBand / Fabric Presence Check

### Purpose

Verify that:

* InfiniBand devices are present on compute nodes
* Basic IB tooling is visible

### Script

```
validation/scripts/30_smoke/smoke_test.sh
```

### Relevant Output

```
validation/results/infra-smoke-<RUN_ID>/phase5_ib.txt
```

### Checks Performed

* `/sys/class/infiniband` presence
* `ibstat` availability (if installed)

### Outcome

* IB devices detected on compute nodes
* Fabric available for NCCL communication

**Fabric presence confirmed**

---

## 6. NCCL Scaling Validation (DDP-Style)

### Purpose

Validate NCCL collective performance in configurations that mirror
PyTorch Distributed Data Parallel (DDP) training.

This directly validates the communication layer used by:

* PyTorch DDP
* PyTorch FSDP (gradient synchronization)

### Script

```
validation/scripts/20_nccl/nccl_scaling.sh
```

### Test Cases

| Case | Configuration                |
| ---- | ---------------------------- |
| 01   | 1 node, 1 GPU                |
| 02   | 1 node, 8 GPUs (DDP-style)   |
| 03   | 2 nodes, 16 GPUs (DDP-style) |

### Relevant Output

```
validation/results/nccl_scaling/run_<RUN_ID>/
```

### Outcome

* NCCL all-reduce completes successfully in all configurations
* Intra-node scaling confirms NVLink / PCIe health
* Inter-node scaling confirms fabric usability for distributed training

**NCCL communication validated for DDP-style workloads**

---

## 7. NCCL + System Monitoring Validation

### Purpose

Confirm that NCCL collectives:

* Saturate GPUs correctly
* Do not cause CPU or memory bottlenecks
* Maintain stable performance over time

### Script

```
validation/scripts/20_nccl/nccl_run_with_monitoring.sh
```

### Relevant Output

```
validation/results/nccl/run_<RUN_ID>/
```

### Metrics Captured

* GPU utilization, memory, power
* PCIe RX/TX throughput
* CPU load and memory availability

### Outcome

* GPUs remain highly utilized during collectives
* No CPU starvation or memory pressure observed
* Stable behavior throughout test duration

**System health under NCCL load validated**

---

## 8. Readiness for Distributed Training

Based on the above validation runs:

* Slurm scheduling is stable
* Shared filesystem is reliable
* GPUs are healthy and visible
* NCCL scales correctly intra- and inter-node
* System remains stable under collective load

 **The cluster is ready for PyTorch DDP and FSDP training workloads.**



