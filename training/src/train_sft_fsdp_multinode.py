#!/usr/bin/env python3
"""
Multi-node FSDP SFT smoke test (Slurm + torchrun):

- torchrun initializes distributed
- rank0 tokenizes & save_to_disk; all ranks load tokenized dataset
- DistributedSampler for proper sharding
- FSDP auto-wrap transformer blocks
- bf16 mixed precision for H100
- rank0 writes metrics.json
"""

import os
import time
import json
import argparse
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from functools import partial



# -------------------------
# helpers
# -------------------------
def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def rank() -> int:
    return int(os.environ.get("RANK", "0"))

def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_main() -> bool:
    return rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def build_text(example):
    inst = example.get("instruction", "")
    inp  = example.get("input", "")
    out  = example.get("output", "")
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{inst}\n\n### Response:\n{out}"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--dataset", default="yahma/alpaca-cleaned")
    ap.add_argument("--run_name", default="sft_fsdp_multinode_smoke")

    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset_size", type=int, default=2048)

    ap.add_argument("--bf16", type=str, default="True")  # "True"/"False"
    ap.add_argument("--activation_ckpt", type=str, default="False")  # optional later

    args = ap.parse_args()

    # ---- distributed init ----
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank())
    device = torch.device("cuda", local_rank())

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if is_main():
        print(
            f"TORCHRUN: RANK={rank()} LOCAL_RANK={local_rank()} WORLD_SIZE={world_size()} "
            f"HOST={os.environ.get('HOSTNAME', os.uname().nodename)}",
            flush=True,
        )

    runs_root = os.environ.get("RUNS_ROOT", "results/training")
    out_dir = os.path.join(runs_root, args.run_name)
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- dataset cache: rank0 preprocess -> barrier -> all ranks load ----
    hf_home = os.environ.get("HF_HOME", "/tmp/hf")
    tok_cache_dir = os.path.join(
        hf_home,
        "tokenized",
        f"{args.dataset.replace('/','_')}_L{args.seq_len}_N{args.subset_size}",
    )
    tok_cache_dir = str(Path(tok_cache_dir))

    if is_main():
        ds = load_dataset(args.dataset, split="train").select(range(args.subset_size))

        def tok_fn(ex):
            text = build_text(ex)
            tok = tokenizer(
                text,
                truncation=True,
                max_length=args.seq_len,
                padding="max_length",
            )
            tok["labels"] = tok["input_ids"].copy()
            return tok

        ds = ds.map(tok_fn, remove_columns=ds.column_names, desc="tokenizing")

        Path(tok_cache_dir).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(tok_cache_dir):
            shutil.rmtree(tok_cache_dir)
        ds.save_to_disk(tok_cache_dir)

    barrier()
    ds = load_from_disk(tok_cache_dir)

    sampler = DistributedSampler(
        ds,
        num_replicas=world_size(),
        rank=rank(),
        shuffle=True,
        drop_last=True,
    )

    dl = DataLoader(
        ds,
        batch_size=args.bsz,
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=2,
        pin_memory=True,
    )

    # ---- model ----
    use_bf16 = args.bf16.lower() == "true"
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.train()

    # ---- FSDP wrap ----
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        reduce_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        buffer_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    # Try to auto-detect transformer block class for wrapping
    # Works for Llama/Mistral style HF models.
    block_cls = None
    for name, mod in model.named_modules():
        # Common block names; fallback is ok for smoke test
        if name.endswith("layers.0") or name.endswith("model.layers.0"):
            block_cls = type(mod)
            break

   # if block_cls is None:
   #     # Last resort: don't auto-wrap; still runs but less sharding benefit
   #     auto_wrap_policy = None
   #     if is_main():
   #         print("WARN: could not detect transformer block class; running without auto-wrap policy", flush=True)
   # else:
   #     auto_wrap_policy = transformer_auto_wrap_policy({block_cls})
   #     if is_main():
   #         print(f"FSDP auto-wrap block class: {block_cls}", flush=True)


    if block_cls is None:
        auto_wrap_policy = None
        if is_main():
            print("WARN: could not detect transformer block class; running without auto-wrap policy", flush=True)
    else:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={block_cls},
        )
        if is_main():
            print(f"FSDP auto-wrap block class: {block_cls}", flush=True)
    


    model = FSDP(
        model,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
    )

    # ---- optimizer ----
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # mixed precision scaling (only needed if fp16; bf16 typically doesn't need it, but safe)
    scaler = ShardedGradScaler(enabled=(not use_bf16))

    # ---- train loop ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    step = 0
    optim.zero_grad(set_to_none=True)

    for batch in dl:
        # sampler epoch for determinism across ranks
        if step == 0:
            sampler.set_epoch(0)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
            out = model(**batch)
            loss = out.loss / args.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        step += 1
        if step >= args.max_steps:
            break

    barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    tokens_per_rank = args.max_steps * args.bsz * args.seq_len
    toks_per_s = tokens_per_rank / max(t1 - t0, 1e-9)

    if is_main():
        metrics = {
            "wall_time_s": t1 - t0,
            "approx_tokens_per_rank": tokens_per_rank,
            "approx_tokens_per_s_per_rank": toks_per_s,
            "world_size": world_size(),
            "gpu0": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model": args.model,
            "dataset": args.dataset,
            "fsdp": True,
            "bf16": use_bf16,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("METRICS:", json.dumps(metrics, indent=2), flush=True)

    # clean exit
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

