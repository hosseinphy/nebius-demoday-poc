"""
Minimal single-GPU SFT training (smoke test).

GOAL (v1):
- single node
- single GPU
- ~10 steps
- write metrics.json
- no DDP / FSDP yet
"""

# -------------------------
# imports (keep minimal)
# -------------------------
import os
import time
import json
import argparse

import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# -------------------------
# helper: prompt formatting
# -------------------------
def format_example(example: dict) -> str:
    """
    Convert one dataset row into a single training string.

    Expected fields for yahma/alpaca-cleaned:
    - instruction
    - input
    - output
    """
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()

    if inp:
        return (
            f"### Instruction:\n{inst}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{out}"
        )
    return (
        f"### Instruction:\n{inst}\n\n"
        f"### Response:\n{out}"
    )


# -------------------------
# main
# -------------------------
def main():
    # ---- args
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, required=True)
    #parser.add_argument("--dataset", type=str, required=True)
    #parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--run_name", default="sft_1gpu_smoke")

    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=2048)
    args = parser.parse_args()

    # ---- paths (DO NOT hardcode)
    # NOTE: RUNS_ROOT comes from env.sh
    runs_root = os.environ["RUNS_ROOT"]
    out_dir = os.path.join(runs_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # save args -> run_args.json
    with open(os.path.join(out_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- sanity
    assert torch.cuda.is_available(), "CUDA must be available"
    device = torch.device("cuda")

    # ---- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # ---- dataset
    ds = load_dataset(args.dataset, split="train")
    if args.subset_size is not None and args.subset_size > 0:
        ds = ds.select(range(min(args.subset_size, len(ds))))

    # ---- tokenization
    def tokenize_fn(example):
        """
        Convert raw example -> tokenized tensors.

        MUST:
        - truncate
        - fixed max_length
        - create labels = input_ids copy
        """
        text = format_example(example)
        tok = tokenizer(
            text,
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

    # ---- collator
    # (labels already exist, but LM collator is safe + standard)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- training args
    train_args = TrainingArguments(
        output_dir=out_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-5,
        warmup_steps=0,
        weight_decay=0.0,
        logging_steps=1,
        save_steps=args.max_steps,
        save_total_limit=1,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    # ---- trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )

    # ---- train + timing
    torch.cuda.synchronize()
    t0 = time.time()

    trainer.train()

    torch.cuda.synchronize()
    t1 = time.time()

    # ---- metrics (very simple)
    # NOTE: this is *approximate* throughput, good enough for PoC
    approx_tokens = args.max_steps * args.bsz * args.seq_len
    metrics = {
        "wall_time_s": t1 - t0,
        "approx_tokens": approx_tokens,
        "approx_tokens_per_s": approx_tokens / max(t1 - t0, 1e-9),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model": args.model,
        "dataset": args.dataset,
    }

    # write metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("METRICS:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

