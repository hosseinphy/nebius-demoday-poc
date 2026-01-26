
# Single-Node, Single-GPU Training Validation

This document records the successful execution and validation of a
single-node, single-GPU training run on the Nebius GPU cluster using Slurm.
This test establishes a correct and reproducible baseline before scaling to
multi-GPU (DDP) and multi-node training.

---

## Objective

The goals of this test were to verify that:

- Slurm can allocate a GPU and launch a training job
- The Python virtual environment is available on compute nodes
- PyTorch detects and uses CUDA correctly
- A real LLM fine-tuning script runs end-to-end
- Training outputs are written to a writable shared filesystem
- GPU memory, compute, and system behavior are stable

This serves as the **baseline correctness check** for all subsequent
distributed experiments.

---

## Repository Context

- **Repository root**  
  `/mnt/sharedfs/nebius-demoday-test`

- **Training script**  
  `training/src/train_sft_min.py`

- **Slurm submission script**  
  `training/slurm/train_1gpu.sbatch`

- **Python virtual environment**  
  `/mnt/sharedfs/nebius-demoday-test/.venv`

- **Training outputs directory**  
  `$RUNS_ROOT = /mnt/sharedfs/nebius-demoday-test/results/training`

---

## Model and Dataset

- **Model**: `mistralai/Mistral-7B-v0.1`
- **Architecture**: Decoder-only Transformer (7B parameters)
- **Dataset**: `yahma/alpaca-cleaned`
- **Training type**: Supervised Fine-Tuning (SFT)
- **Sequence length**: 512 tokens

---

## Environment Setup

### Python Environment

A shared Python virtual environment was created on the shared filesystem
to ensure consistency across login and compute nodes:

```

/mnt/sharedfs/nebius-demoday-test/.venv

````

Key components:

- Python: 3.12.3
- PyTorch: 2.5.1 + CUDA 12.1
- Accelerate, Transformers, Datasets, and related dependencies

### Environment Variables

The training environment is configured via `training/env.sh`:

```bash
REPO_ROOT=/mnt/sharedfs/nebius-demoday-test
RUNS_ROOT=/mnt/sharedfs/nebius-demoday-test/results/training
HF_HOME=/mnt/sharedfs/nebius-demoday-test/.cache/huggingface
````

These ensure:

* All outputs are written inside the repository
* HuggingFace cache is writable
* No dependency on restricted system paths

---

## Slurm Job Configuration

The training job was launched using Slurm with the following configuration:

* **Nodes**: 1
* **GPUs**: 1
* **CPUs per task**: 8
* **Memory**: 64 GB
* **Time limit**: 30 minutes
* **Partition**: `gpu`

### Slurm Submission Script

File: `training/slurm/train_1gpu.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=train_1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/sharedfs/nebius-demoday-test
#SBATCH --output=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/sharedfs/nebius-demoday-test/slurm-logs/%x-%j.err

set -euo pipefail
set -x

mkdir -p slurm-logs

source training/env.sh
source /mnt/sharedfs/nebius-demoday-test/.venv/bin/activate

python training/src/train_sft_min.py \
  --run_name sft_1gpu_smoke \
  --max_steps 10 \
  --seq_len 512
```

---

## Training Execution

The job was submitted from the login node using:

```bash
sbatch training/slurm/train_1gpu.sbatch
```

The job executed entirely on a single compute node (`worker0`) with one
NVIDIA H100 80GB GPU.

---

## Training Metrics

### Step-Level Metrics (sample)

| Step |   Loss | Grad Norm | Learning Rate | Epoch |
| ---: | -----: | --------: | ------------: | ----: |
|    1 | 1.2753 |     50.50 |       2.0e-05 |  0.00 |
|    2 | 1.8605 |     69.00 |       1.8e-05 |  0.00 |
|    3 | 1.1892 |     29.75 |       1.6e-05 |  0.01 |
|    4 | 1.2061 |     28.13 |       1.4e-05 |  0.01 |
|    5 | 1.0492 |     30.25 |       1.2e-05 |  0.01 |
|    6 | 1.1469 |     52.25 |       1.0e-05 |  0.01 |
|    7 | 1.3285 |     50.00 |       8.0e-06 |  0.01 |
|    8 | 1.2227 |     33.75 |       6.0e-06 |  0.02 |
|    9 | 1.1622 |     29.88 |       4.0e-06 |  0.02 |
|   10 | 0.9635 |     34.75 |       2.0e-06 |  0.02 |

### Aggregate Metrics

```json
{
  "train_runtime_s": 139.23,
  "train_loss": 1.2404,
  "train_samples_per_second": 0.287,
  "train_steps_per_second": 0.072,
  "approx_tokens": 5120,
  "approx_tokens_per_second": 36.74
}
```

---

## Hardware and Software

* **GPU**: NVIDIA H100 80GB HBM3
* **CUDA**: Enabled and detected correctly
* **PyTorch**: 2.5.1+cu121
* **Python**: 3.12.3

---

## Results and Interpretation

* The job completed successfully with exit code `0`
* PyTorch correctly detected and utilized the GPU
* GPU memory usage peaked well below the 80 GB limit
* No CUDA, NCCL, or runtime errors were observed
* Training outputs were written correctly to `$RUNS_ROOT`

This confirms that:

* The environment is correctly configured
* The training pipeline is functional
* The hardware is stable and not resource-constrained

---

## Conclusion

Single-node, single-GPU training has been **successfully validated**.

This establishes a reliable baseline and clears the path for:

1. Single-node, multi-GPU training with DDP
2. Multi-node, multi-GPU training with DDP
3. Fully Sharded Data Parallel (FSDP) experiments

All subsequent distributed training builds on this verified setup.


