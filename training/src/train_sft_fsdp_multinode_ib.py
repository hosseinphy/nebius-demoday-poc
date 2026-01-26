#!/usr/bin/env python3
"""
Minimal, reliable FSDP training script for Slurm/torchrun infra validation.

Usage (single node):
  torchrun --nproc_per_node=8 src/train_sft_fsdp_multinode_ib.py

Usage (multi node):
  torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=worker0:20046 \
    src/train_sft_fsdp_multinode_ib.py

This script:
- initializes torch.distributed (NCCL)
- builds a small TransformerEncoder model (SFT-like sequence modeling)
- wraps with FSDP
- runs a few steps with synthetic token data
"""

import os
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


# -------------------------
# Utils
# -------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def local_rank() -> int:
    # torchrun sets LOCAL_RANK
    return int(os.environ.get("LOCAL_RANK", "0"))


def log0(msg: str) -> None:
    if rank() == 0:
        print(msg, flush=True)


def barrier():
    if is_dist():
        dist.barrier()


# -------------------------
# Data
# -------------------------
class RandomTokenDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int = 1234):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # deterministic-ish per idx to reduce noise
        g = torch.Generator()
        g.manual_seed(self.seed + idx)
        x = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g, dtype=torch.long)
        # next-token prediction: shift left by 1
        y = torch.roll(x, shifts=-1, dims=0)
        return x, y


# -------------------------
# Model (SFT-like)
# -------------------------
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lm = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: [B, T]
        h = self.tok(input_ids)              # [B, T, D]
        h = self.enc(h)                      # [B, T, D]
        logits = self.lm(h)                  # [B, T, V]
        return logits


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    steps: int
    batch_size: int
    seq_len: int
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    lr: float
    num_samples: int
    num_workers: int
    fp16: bool
    bf16: bool
    grad_accum: int
    log_every: int


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_samples", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--log_every", type=int, default=5)
    a = p.parse_args()

    if a.fp16 and a.bf16:
        raise SystemExit("Choose only one: --fp16 or --bf16")

    return TrainConfig(
        steps=a.steps,
        batch_size=a.batch_size,
        seq_len=a.seq_len,
        vocab_size=a.vocab_size,
        d_model=a.d_model,
        n_heads=a.n_heads,
        n_layers=a.n_layers,
        dropout=a.dropout,
        lr=a.lr,
        num_samples=a.num_samples,
        num_workers=a.num_workers,
        fp16=a.fp16,
        bf16=a.bf16,
        grad_accum=max(1, a.grad_accum),
        log_every=max(1, a.log_every),
    )


# -------------------------
# Main
# -------------------------
def main():
    cfg = parse_args()

    # ---- distributed init ----
    # torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank())
    device = torch.device("cuda", local_rank())

    log0(f"dist: world_size={world_size()} ranks, node_local_rank={local_rank()}")
    log0(f"env: MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}")
    barrier()

    # ---- dataset / loader ----
    ds = RandomTokenDataset(num_samples=cfg.num_samples, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
    sampler = DistributedSampler(ds, num_replicas=world_size(), rank=rank(), shuffle=True, drop_last=True)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---- model ----
    model = TinyTransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)

    # FSDP auto-wrap to shard encoder layers
    auto_wrap = size_based_auto_wrap_policy(min_num_params=5_000_000)

    mp = None
    if cfg.bf16:
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    elif cfg.fp16:
        mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        device_id=device,
        sync_module_states=True,   # important for multinode correctness
    )

    # ---- loss/optim ----
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # ---- train loop ----
    model.train()
    t0 = time.time()
    step = 0
    data_iter = iter(dl)

    # For consistent shuffling across epochs
    epoch = 0

    log0("Starting training...")
    barrier()

    while step < cfg.steps:
        if step % len(dl) == 0:
            sampler.set_epoch(epoch)
            epoch += 1
            data_iter = iter(dl)

        optim.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for micro in range(cfg.grad_accum):
            x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)  # [B, T, V]
            loss = loss_fn(logits.view(-1, cfg.vocab_size), y.view(-1))
            (loss / cfg.grad_accum).backward()
            loss_accum += loss.item()

        optim.step()

        if step % cfg.log_every == 0:
            # reduce loss to rank0 for nicer logs
            loss_t = torch.tensor([loss_accum], device=device, dtype=torch.float32)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            loss_mean = (loss_t / world_size()).item()

            if rank() == 0:
                dt = time.time() - t0
                toks = cfg.batch_size * cfg.seq_len * world_size() * cfg.log_every
                tok_s = toks / dt if dt > 0 else 0.0
                print(
                    f"[step {step:04d}] loss={loss_mean:.4f} "
                    f"tok/s≈{tok_s:,.0f} (global) "
                    f"(bs={cfg.batch_size}, seq={cfg.seq_len}, ws={world_size()})",
                    flush=True,
                )
            t0 = time.time()

        step += 1

    barrier()
    log0("Training finished cleanly.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

