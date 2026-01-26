# DDP 1-Node Smoke Test (8×H100) — Mistral-7B SFT + LoRA

## Goal

Validate that **PyTorch DDP (via `torchrun`) + HuggingFace `Trainer`** can run a short supervised fine-tuning (SFT) loop on a **single node** with **8 GPUs**, using **LoRA** to keep memory and runtime manageable for a Demo-Day style proof-of-concept.

---

## Hardware / Environment

* **Node:** `worker0`
* **GPUs:** 8× **NVIDIA H100 80GB HBM3**
* **World size:** 8 (one process per GPU)
* **Precision:** BF16
* **Distributed backend:** NCCL
* **Shared filesystem:** `/mnt/sharedfs/nebius-demoday-test`
* **HF cache root:** `/mnt/sharedfs/nebius-demoday-test/.cache/huggingface`

> Important note: In this setup, both the **training code** and the **HuggingFace caches** are on the shared filesystem so all ranks see the same paths.

---

## Model / Dataset

### Model

* `mistralai/Mistral-7B-v0.1`

### Dataset

* `yahma/alpaca-cleaned`
* Using `split="train"`
* Subset used for smoke test: `subset_size=2048`

### Prompt format

Each example is converted to a simple instruction-following format:

* If `input` exists:

  * `Instruction`, `Input`, `Response`
* Else:

  * `Instruction`, `Response`

Tokenization:

* `max_length = 512`
* `padding="max_length"`
* Labels are a copy of `input_ids` (standard causal LM SFT)

---

## Job Script (Slurm)

```bash
#!/bin/bash
#SBATCH --job-name=train_ddp_1node
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/sharedfs/nebius-demoday-test
#SBATCH --output=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.err

set -euo pipefail
set -x

mkdir -p /mnt/sharedfs/nebius-demoday-test/slurm-logs

source training/env.sh || true
source /mnt/sharedfs/nebius-demoday-test/.venv/bin/activate

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=^lo,docker0

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

torchrun \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  training/src/train_sft_ddp_min.py \
    --run_name sft_ddp_1node_smoke \
    --max_steps 10 \
    --seq_len 512 \
    --bsz 1 \
    --grad_accum 1
```

---

## Attempt 1 — Dataset cache race / missing Arrow file

### What happened

One of the ranks crashed during:

* `ds = ds.map(tok_fn, remove_columns=ds.column_names)`

Error:

* `FileNotFoundError: ... cache-xxxx.arrow ... No such file or directory`

### Why it happened (root cause)

All 8 ranks were calling `datasets.map()` simultaneously while sharing the same `HF_HOME` on the shared filesystem.
That triggers a **multi-process cache collision / race condition**: one rank expects an Arrow cache shard that another rank hasn’t finalized (or deletes/renames as part of dataset caching).

### Fix

**Do not run `datasets.map()` concurrently across ranks against the same cache path.**
Use one of these safe patterns:

* **Best for your case (simple + robust):**

  * Run preprocessing on **rank 0 only**
  * Save the processed dataset to disk
  * Barrier
  * All ranks load the processed dataset from disk

(You effectively moved toward this direction in the later run where you see “Saving the dataset…” before training begins.)

---

## Attempt 2 — Full fine-tune OOM during optimizer step

### What happened

Training started but crashed with CUDA OOM during AdamW step:

* `torch.OutOfMemoryError ... _multi_tensor_adamw ... torch._foreach_sqrt`

Even though batch size was small, **optimizer state + gradients for full 7B** still causes a huge memory footprint per rank in DDP.

### Why it happened (root cause)

DDP replicates the full model on each GPU. For a 7B model, full fine-tuning includes:

* Trainable weights (BF16)
* Gradients
* AdamW optimizer states (exp_avg / exp_avg_sq) — this is the big one

So the optimizer step is often the first place you hit memory failure.

### Fix

Switch from full fine-tuning → **LoRA** (parameter-efficient fine-tuning):

* Freeze base model weights
* Train only low-rank adapter parameters
* Massive reduction in gradients + optimizer states
* Fits comfortably in memory for a smoke test

---

## Attempt 3 (Final) — LoRA DDP Success

### What worked

* DDP launched correctly (8 ranks)
* NCCL init completed
* Tokenization + dataset materialization completed
* Training loop ran for 10 steps
* Metrics written successfully

You can see NCCL init confirmation:

* `ncclCommInitRank ... nranks 8 ... Init COMPLETE`

And training logs + final report:

```text
{'loss': 1.3668, 'grad_norm': 1.3885, 'learning_rate': 5e-05, 'epoch': 0.0}
...
{'loss': 0.906, 'grad_norm': 1.1261, 'learning_rate': 5e-06, 'epoch': 0.04}
{'train_runtime': 6.2282, 'train_samples_per_second': 12.845, 'train_steps_per_second': 1.606, 'train_loss': 1.1239, 'epoch': 0.04}
```

### Final “PoC” metrics (rank-0 summary)

```json
{
  "wall_time_s": 6.766317844390869,
  "approx_tokens_per_rank": 5120,
  "approx_tokens_per_s_per_rank": 756.6892537045638,
  "world_size": 8,
  "gpu0": "NVIDIA H100 80GB HBM3",
  "torch": "2.5.1+cu121",
  "model": "mistralai/Mistral-7B-v0.1",
  "dataset": "yahma/alpaca-cleaned"
}
```

Interpretation:

* **Per-rank tokens** = `max_steps * bsz * seq_len` = `10 * 1 * 512 = 5120`
* **Per-rank throughput** ≈ **756.7 tok/s**
* This is a **smoke test**, not a tuned benchmark (tiny steps + overhead dominates).

---

## Dashboard evidence (Nebius metrics)

These dashboards show the short burst of activity consistent with a 10-step smoke run.

### GPU dashboard

* File: `/mnt/data/1gpu_gpu_dashboard--ddp-1node.png`
* Notes:

  * GPU utilization spikes during initialization and training steps
  * Frame buffer free/used changes during model load + training
  * PCIe RX/TX spikes are consistent with initial loading + runtime transfers

### CPU dashboard

* File: `/mnt/data/1gpu_cpu_dashboard--ddp-1node.png`
* Notes:

  * CPU spike aligns with dataset prep/tokenization + process startup
  * Disk activity correlates with HF cache reads/writes (model + dataset)

---

## Notes / Answers to your questions

### 1) “Can we use DistributedSampler? Sharding the dataset?”

Yes.

* For **training**, HuggingFace `Trainer` + `accelerate` will automatically use a distributed sampler when launched under DDP (so you usually *don’t* manually plug `DistributedSampler`).
* For **preprocessing/tokenization**, you *should* either:

  * preprocess once (rank 0) → save → all ranks load, **or**
  * shard explicitly per-rank to avoid cache collisions, e.g. `ds = ds.shard(num_shards=world_size, index=rank)` *only for preprocessing work*, then save per-rank outputs (less convenient).

The key is: **avoid multiple ranks writing to the same dataset cache artifacts at the same time**.

### 2) “Did we download code to the shared filesystem before running these models?”

Yes, effectively:

* Your Slurm script sets:

  * `#SBATCH --chdir=/mnt/sharedfs/nebius-demoday-test`
* Your venv is on sharedfs:

  * `/mnt/sharedfs/nebius-demoday-test/.venv`
* HF cache is on sharedfs:

  * `/mnt/sharedfs/nebius-demoday-test/.cache/huggingface`

So both **code execution** and **model/dataset downloads** are happening under the shared filesystem paths you configured.




---


# 2-Node DDP Smoke Test (LoRA SFT) on Nebius H100 — Slurm + torchrun

This note documents a **working 2-node / 16-GPU DDP** run using **PyTorch torchrun + HuggingFace Trainer/Accelerate**, with **LoRA fine-tuning** on **Mistral-7B**. It also records the key **assumptions**, **failure modes encountered**, and the **workarounds** that made the run stable and reproducible.

---

## Goal

Validate that:

- **Slurm → torchrun** multi-node launch is correct
- NCCL connectivity across nodes is stable
- HF Trainer/Accelerate works in multi-node mode (no manual `dist.init_process_group`)
- Dataset prep does not race across ranks
- LoRA training runs end-to-end and produces a baseline throughput metric

---

## Target Architecture

- **Nodes:** 2
- **GPUs:** 8 × H100 per node → **16 GPUs total**
- **Backend:** NCCL
- **Launcher:** `srun` (1 task per node) + `torchrun` (8 procs per node)
- **Model:** `mistralai/Mistral-7B-v0.1`
- **Dataset:** `yahma/alpaca-cleaned` (subset: 2048 examples)
- **Precision:** BF16
- **LoRA:** enabled
- **Gradient checkpointing:** disabled (see issues section)

---

## Working Result (Smoke Metrics)

```json
{
  "wall_time_s": 11.584831714630127,
  "approx_tokens_per_rank": 5120,
  "approx_tokens_per_s_per_rank": 441.95721838014356,
  "world_size": 16,
  "gpu0": "NVIDIA H100 80GB HBM3",
  "torch": "2.5.1+cu121",
  "model": "mistralai/Mistral-7B-v0.1",
  "dataset": "yahma/alpaca-cleaned",
  "use_lora": true,
  "bf16": true,
  "gradient_checkpointing": false
}
````

**Interpretation (for this smoke run):**

* Each rank processed ~`max_steps × bsz × seq_len = 10 × 1 × 512 = 5120 tokens`.
* Per-rank throughput ~ **442 tokens/s**.
* Total effective throughput (very rough) ≈ `442 × 16 ≈ 7072 tokens/s` (ignores overheads, logging, etc.).

---

## Presumptions / Assumptions

### Infrastructure assumptions

* Nodes can reach each other over a real NIC (here assumed **`eth0`**).
* `eth0` has a routable IPv4 address visible to all nodes.
* Slurm allocates exactly **2 nodes**, each with **8 GPUs**, and `CUDA_VISIBLE_DEVICES` is managed per node.

### Storage assumptions

* A **shared filesystem** exists and is accessible from all nodes at:

  `/mnt/sharedfs/nebius-demoday-test`

* HF cache is shared via `HF_HOME` so tokenized dataset artifacts are visible across ranks.

### Software assumptions

* Environment is consistent on both nodes (`.venv` exists and is usable everywhere).
* `torch==2.5.1+cu121`, `transformers`, `datasets`, `accelerate`, `peft` installed.
* HF Trainer/Accelerate controls distributed init (no manual `dist.init_process_group`).

---

## Implementation Overview

### 1) Slurm job responsibilities

* Allocate nodes + GPUs
* Compute rendezvous endpoint
* Export stable NCCL networking variables
* Launch **one** process per node via `srun`
* Inside each node: activate env, then launch `torchrun` with 8 local ranks

### 2) Training script responsibilities

* **Never manually init distributed** (`Trainer/Accelerate` does it)
* Ensure dataset prep is safe:

  * **rank0 preprocesses + writes tokenized dataset to disk**
  * `barrier()`
  * all ranks load from disk
* Safe metrics writing: **rank0 only**

---

## Slurm / torchrun Script (Key Parts)

### Networking and rendezvous

* Force torch/NCCL to use a stable TCP path (avoid IB auto-probing issues):

```bash
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export GLOO_USE_IPV6=0
```

* Derive MASTER address from the **first node’s eth0 IPv4**:

```bash
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE="${NODES[0]}"

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc \
  "ip -4 -o addr show dev eth0 | awk '{print \$4}' | cut -d/ -f1")
MASTER_PORT=29500
RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
export MASTER_ADDR MASTER_PORT RDZV_ENDPOINT
```

### Launch pattern: `srun` + `torchrun`

* `srun`: 1 task per node
* `torchrun`: 8 processes per node
* node rank derived from `SLURM_NODEID`

```bash
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -lc '
  cd /mnt/sharedfs/nebius-demoday-test

  source training/env.sh || true
  source /mnt/sharedfs/nebius-demoday-test/.venv/bin/activate

  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint="$RDZV_ENDPOINT" \
    training/src/train_sft_ddp_lora_multinode.py \
      --run_name sft_ddp_2node_lora_smoke \
      --max_steps 10 \
      --seq_len 512 \
      --bsz 1 \
      --grad_accum 1 \
      --bf16 True \
      --gradient_checkpointing False \
      --use_lora \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05
'
```

### Debug prints for confidence

A tiny Python snippet prints the per-node view of rendezvous + NIC config:

```bash
python - <<'PY'
import os, socket
print("host", socket.gethostname())
print("MASTER_ADDR", os.environ.get("MASTER_ADDR"))
print("MASTER_PORT", os.environ.get("MASTER_PORT"))
print("RDZV_ENDPOINT", os.environ.get("RDZV_ENDPOINT"))
print("NCCL_IB_DISABLE", os.environ.get("NCCL_IB_DISABLE"))
print("NCCL_NET", os.environ.get("NCCL_NET"))
print("NCCL_SOCKET_IFNAME", os.environ.get("NCCL_SOCKET_IFNAME"))
print("GLOO_SOCKET_IFNAME", os.environ.get("GLOO_SOCKET_IFNAME"))
PY
```

---

## Training Script Design (Highlights)

### 1) Distributed initialization philosophy

* **Do not** call `torch.distributed.init_process_group`.
* HuggingFace Trainer/Accelerate handles setup when launched under torchrun.

```python
# IMPORTANT:
# Do NOT call dist.init_process_group here.
# Do NOT move model to device here. Trainer/Accelerate will place it correctly.
```

### 2) LoRA integration (PEFT)

LoRA is applied to attention + MLP projections:

```python
target_modules=[
  "q_proj", "k_proj", "v_proj", "o_proj",
  "gate_proj", "up_proj", "down_proj",
]
```

`print_trainable_params()` logs how many parameters are actually trainable.

### 3) Dataset race avoidance (critical)

Problem: if all ranks tokenize/write to the same cache location simultaneously, you can get:

* file corruption
* partial writes
* lock contention / cache races
* nondeterministic failures

Solution implemented:

* Only rank0 loads, tokenizes, and saves to disk
* Barrier
* Everyone loads tokenized dataset from shared cache

```python
if is_main():
    ds = load_dataset(...).select(...)
    ds = ds.map(tok_fn, ...)
    if os.path.exists(tok_cache_dir):
        shutil.rmtree(tok_cache_dir)
    ds.save_to_disk(tok_cache_dir)

barrier()
ds = load_from_disk(tok_cache_dir)
```

### 4) Metrics written only by rank0

Avoid multiple ranks writing the same file:

```python
if is_main():
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
```

---

## Issues Encountered + Solutions / Workarounds

### Issue A — NCCL multi-node init failures (socket / IB probing)

**Symptom (earlier attempt):**

* NCCL internal errors during DDP verification / process group setup
* Errors resembling socket handshake mismatches (or IB-related warnings)

**Root cause hypothesis:**

* NIC selection ambiguity or NCCL probing incorrect interfaces (or attempting IB)
* Unstable behavior when IB stack is partially present / misdetected

**Workaround that stabilized the cluster immediately:**
Force deterministic TCP sockets and pin to the correct NIC:

```bash
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

Optional socket tuning for high process counts:

```bash
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=2
```

---

### Issue B — DDP crash: “Expected to mark a variable ready only once”

**Symptom:**
DDP fails during the first backward pass with:

> `Expected to mark a variable ready only once...`
> `...lora_B...weight has been marked as ready twice`

**Root cause hypothesis (most likely):**
The combination of:

* **DDP**
* **gradient checkpointing**
* **LoRA / PEFT modules**
  can create **re-entrant backward passes** where the same parameter triggers autograd hooks twice.
  DDP expects each parameter to be “ready” exactly once per iteration.

**Workaround used (successful):**
Disable gradient checkpointing for the smoke run:

* CLI: `--gradient_checkpointing False`
* Script respects this flag, so `model.gradient_checkpointing_enable()` is not called.

This change eliminated the “ready twice” error and allowed stable training.

**Notes / alternatives (not used in this smoke run):**
If checkpointing is required later for memory reasons, typical workarounds include:

* `_set_static_graph()` on the DDP model (if graph is static)
* enabling non-reentrant checkpointing (`use_reentrant=False`) where supported

---

## Files / Where This Lives in Repo

Suggested repo locations under your `training/` area:

* Slurm script:

  * `training/slurm/train_ddp_2node.sbatch`
* Training script:

  * `training/src/train_sft_ddp_lora_multinode.py`
* Logs:

  * `slurm-logs/train_ddp_2node_lora-<jobid>.out`
  * `slurm-logs/train_ddp_2node_lora-<jobid>.err`
* Run outputs:

  * `results/training/sft_ddp_2node_lora_smoke/`

    * `run_args.json`
    * `metrics.json`
    * checkpoint (if saved)

---

## Quick Checklist for Re-running

* [ ] Confirm `eth0` is the correct NIC on both nodes (`ip -br link`)
* [ ] Confirm shared FS is mounted on both nodes (`/mnt/sharedfs/...`)
* [ ] Ensure `.venv` exists and is compatible on all nodes
* [ ] Ensure `HF_HOME` points to a shared location
* [ ] Use `--gradient_checkpointing False` when LoRA + DDP is enabled (unless using advanced workaround)
* [ ] Validate rendezvous endpoint prints the same `MASTER_ADDR:MASTER_PORT` across nodes
* [ ] Keep smoke run small (`max_steps=10`, subset_size=2048) before scaling

---




