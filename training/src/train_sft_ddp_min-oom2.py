# training/src/train_sft_ddp_min.py
import os
import time
import json
import argparse

import torch
import torch.distributed as dist

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main() -> bool:
    return (not is_dist()) or int(os.environ.get("RANK", "0")) == 0


def ddp_setup() -> None:
    if is_dist() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier(local_rank: int) -> None:
    # Avoid the "devices used by this process are currently unknown" warning/hang risk
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])


def build_text(example: dict) -> str:
    inst = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{inst}\n\n### Response:\n{out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--dataset", default="yahma/alpaca-cleaned")
    ap.add_argument("--run_name", default="sft_ddp_1node_smoke")
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset_size", type=int, default=2048)
    args = ap.parse_args()

    # DDP
    ddp_setup()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(
        f"DDP init: RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}",
        flush=True,
    )

    # H100-friendly math
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Outputs
    runs_root = os.environ.get("RUNS_ROOT", "results/training")
    out_dir = os.path.join(runs_root, args.run_name)

    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    ddp_barrier(local_rank)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model: load bf16 directly + place on correct GPU
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    # -------------------------------
    # Dataset preprocessing (DDP-safe)
    # -------------------------------
    ds_raw = load_dataset(args.dataset, split="train").select(range(args.subset_size))

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

    hf_home = os.environ.get("HF_HOME", os.path.join(os.getcwd(), ".cache/huggingface"))
    tok_cache_dir = os.path.join(
        hf_home,
        "tokenized",
        f"{args.dataset.replace('/','__')}__seqlen{args.seq_len}__subset{args.subset_size}",
    )

    if is_main():
        os.makedirs(os.path.dirname(tok_cache_dir), exist_ok=True)
        if not os.path.exists(tok_cache_dir):
            print(f"[rank0] Tokenizing and saving dataset to: {tok_cache_dir}", flush=True)
            ds_tok = ds_raw.map(tok_fn, remove_columns=ds_raw.column_names, desc="tokenize")
            ds_tok.save_to_disk(tok_cache_dir)
        else:
            print(f"[rank0] Using existing tokenized dataset at: {tok_cache_dir}", flush=True)

    ddp_barrier(local_rank)

    ds = load_from_disk(tok_cache_dir)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=out_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=1,
        save_steps=args.max_steps,
        save_total_limit=1,
        bf16=True,
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )

    torch.cuda.synchronize()
    t0 = time.time()
    trainer.train()
    torch.cuda.synchronize()
    t1 = time.time()

    # Throughput estimates
    tokens_per_rank = args.max_steps * args.bsz * args.seq_len
    toks_per_s_per_rank = tokens_per_rank / max(t1 - t0, 1e-9)
    toks_per_s_global_est = toks_per_s_per_rank * world_size

    if is_main():
        metrics = {
            "wall_time_s": t1 - t0,
            "approx_tokens_per_rank": tokens_per_rank,
            "approx_tokens_per_s_per_rank": toks_per_s_per_rank,
            "approx_tokens_per_s_global_est": toks_per_s_global_est,
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model": args.model,
            "dataset": args.dataset,
            "bsz": args.bsz,
            "grad_accum": args.grad_accum,
            "seq_len": args.seq_len,
            "max_steps": args.max_steps,
            "subset_size": args.subset_size,
            "hf_home": hf_home,
            "tokenized_dataset_path": tok_cache_dir,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("METRICS:", json.dumps(metrics, indent=2), flush=True)

    ddp_cleanup()


if __name__ == "__main__":
    main()

