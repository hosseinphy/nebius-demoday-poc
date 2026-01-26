
# Fully Sharded Data Parallel (FSDP) Training — Design and Expected Behavior

## Objective

The objective of this step is to design and prepare a **Fully Sharded Data Parallel (FSDP)** training configuration for large language model (LLM) training on a GPU cluster.

FSDP is evaluated as a complement to Distributed Data Parallel (DDP) to:
- reduce per-GPU memory footprint
- enable training of larger models and/or larger effective batch sizes
- demonstrate advanced distributed training capabilities on the cluster

This document describes:
- how FSDP differs from DDP
- how FSDP is configured in this environment
- how training would be executed
- what runtime behavior and performance characteristics are expected

At the time of writing, GPU worker nodes were unavailable in Slurm, so this document focuses on preparation and expected outcomes.

---

## FSDP vs DDP: Conceptual Overview

### Distributed Data Parallel (DDP)

- Each GPU process holds a **full replica** of:
  - model parameters
  - gradients
  - optimizer state
- Gradients are synchronized via **all-reduce** during backward pass
- Memory per GPU scales with full model size
- Simple and efficient, but limited by GPU memory capacity

### Fully Sharded Data Parallel (FSDP)

- Model parameters, gradients, and optimizer states are **sharded across GPUs**
- Each GPU holds only a fraction of the full model state most of the time
- Parameters are **all-gathered just-in-time** for computation and resharded immediately after
- Gradients are synchronized via **reduce-scatter**
- Significantly reduces per-GPU memory usage at the cost of increased communication

FSDP enables training scenarios that are not feasible with DDP due to memory constraints.

---

## Target Configuration

### Hardware (per node)

- 1 × GPU worker node
- 8 × NVIDIA H100 GPUs (80 GB HBM each)
- NVLink interconnect between GPUs
- ≥ 128 vCPUs
- ≥ 200 GB system memory allocated per job

### Scheduler and launch model

- Slurm scheduler
- One process per GPU
- Processes launched via `torchrun` or `accelerate`
- NCCL backend for collective communication

---

## Software Stack

- Python 3.10+
- PyTorch 2.6 (CUDA-enabled)
- NCCL
- Hugging Face Transformers and Datasets
- Hugging Face Trainer + Accelerate
- Shared filesystem mounted at `/mnt/shared`

---

## Repository Layout (Relevant Paths)

```

/mnt/shared/ubuntu/nebius-demoday/
├── training/
│   ├── src/
│   │   ├── train_sft_min.py
│   │   └── train_sft_fsdp_min.py
│   ├── slurm/
│   │   └── step10_fsdp_1node.sbatch
│   ├── accelerate/
│   │   └── fsdp_1node_8gpu.yaml
│   └── env.sh
├── results/
│   └── training/
├── .cache/
│   └── huggingface/
├── .venv/
└── slurm-logs/

````

All training artifacts are stored on the shared filesystem to ensure consistency across login and compute nodes.

---

## FSDP Configuration Strategy

### Sharding strategy

The configuration uses:

- **FULL_SHARD** strategy:
  - parameters, gradients, and optimizer states are sharded
- Mixed precision using **bfloat16**

This provides the maximum memory savings and is representative of production-scale LLM training.

### Auto-wrapping policy

Transformer blocks are wrapped automatically using a transformer-based auto-wrap policy. This ensures that:
- sharding is applied at the transformer layer granularity
- communication and computation are efficiently overlapped

For Mistral-based models, transformer decoder layers are used as wrapping units.

---

## Data Handling and Sharding

- Dataset is stored on disk under the Hugging Face cache
- Data is loaded and tokenized on CPU
- Hugging Face Trainer automatically applies a `DistributedSampler`
- Each rank processes a unique shard of the dataset
- DataLoader workers run on CPU and feed batches to the local GPU
- Batch tensors are transferred from CPU to GPU prior to computation

No explicit distributed sampler is defined in user code; this is handled internally by Trainer and Accelerate.

---

## Execution Model

Training is launched using `accelerate launch` with an FSDP configuration file:

```bash
accelerate launch \
  --config_file training/accelerate/fsdp_1node_8gpu.yaml \
  training/src/train_sft_min.py
````

Slurm is responsible for:

* allocating GPUs and CPUs
* setting the working directory
* providing a shared execution environment

Accelerate manages:

* process group initialization
* device placement
* FSDP wrapping
* synchronization semantics

---

## Expected Runtime Behavior

### GPU memory usage

* Per-GPU memory usage significantly lower than DDP
* Model parameters are not fully resident on any single GPU
* Enables larger models or longer sequences within the same GPU memory budget

### GPU utilization

* Compute utilization may appear more fragmented due to:

  * frequent all-gather and reduce-scatter operations
  * fine-grained synchronization
* Overall utilization improves for larger, steady-state workloads

### CPU utilization

* Increased CPU usage compared to single-GPU runs
* Similar to DDP due to dataloader workers and process overhead

### Communication patterns

* Frequent NCCL collectives:

  * all-gather before layer computation
  * reduce-scatter after backward pass
* Communication occurs primarily over NVLink within a node
* Network traffic increases compared to DDP

---

## Expected Performance Characteristics

For short smoke-test runs:

* Throughput may be similar to or slightly lower than DDP
* Communication overhead dominates for small batch sizes

For longer or memory-constrained runs:

* FSDP enables higher effective batch sizes
* Overall training throughput improves by avoiding OOM constraints
* Scaling efficiency improves as computation amortizes communication costs

---

## Expected Outputs

Upon successful execution, the following artifacts are expected:

* `results/training/<run_name>/run_args.json`
* `results/training/<run_name>/metrics.json`
* Slurm stdout/stderr logs under `slurm-logs/`

Metrics would include:

* wall-clock time
* approximate tokens processed
* tokens per second
* world size
* sharding strategy
* GPU model

---

## Execution Status

At the time of preparation, GPU worker nodes were reported as unavailable in Slurm:

```
ReqNodeNotAvail, UnavailableNodes:fabric3-worker-[0-1]
```

The FSDP configuration, scripts, and launch commands are complete and ready for execution once GPU nodes are resumed.

---

## Next Steps

Once GPU nodes become available:

1. Execute the prepared FSDP job
2. Capture GPU memory and utilization metrics
3. Compare results against DDP and single-GPU baselines
4. Document observed scaling, memory savings, and communication behavior

This document serves as the design and expectation reference for FSDP-based distributed training experiments.

```

