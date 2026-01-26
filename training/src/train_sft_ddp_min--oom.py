import os, time, json, argparse
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

import os
print(f"DDP init: RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}", flush=True)


def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def is_main():
    return (not is_dist()) or int(os.environ.get("RANK", "0")) == 0

def ddp_setup():
    if is_dist() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

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
    ap.add_argument("--run_name", default="sft_ddp_1node_smoke")
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset_size", type=int, default=2048)
    args = ap.parse_args()

    ddp_setup()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    runs_root = os.environ.get("RUNS_ROOT", "results/training")
    out_dir = os.path.join(runs_root, args.run_name)
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    ds = load_dataset(args.dataset, split="train").select(range(args.subset_size))

    def tok_fn(ex):
        text = build_text(ex)
        tok = tokenizer(text, truncation=True, max_length=args.seq_len, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds = ds.map(tok_fn, remove_columns=ds.column_names)

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
        report_to=[],
        remove_unused_columns=False,

        # critical for DDP
        ddp_backend="nccl",
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

    # Each rank processed its own batches. Rough, but good for PoC.
    tokens = args.max_steps * args.bsz * args.seq_len
    toks_per_s = tokens / max(t1 - t0, 1e-9)

    if is_main():
        metrics = {
            "wall_time_s": t1 - t0,
            "approx_tokens_per_rank": tokens,
            "approx_tokens_per_s_per_rank": toks_per_s,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "gpu0": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model": args.model,
            "dataset": args.dataset,
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("METRICS:", json.dumps(metrics, indent=2))

    ddp_cleanup()

if __name__ == "__main__":
    main()

