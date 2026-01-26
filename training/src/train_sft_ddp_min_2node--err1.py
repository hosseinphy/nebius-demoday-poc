# training/src/train_sft_ddp_min.py
import os, time, json, argparse, shutil
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def rank() -> int:
    return int(os.environ.get("RANK", "0"))

def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_dist() -> bool:
    return world_size() > 1

def is_main() -> bool:
    return (not is_dist()) or rank() == 0

def build_text(example):
    inst = example.get("instruction", "")
    inp  = example.get("input", "")
    out  = example.get("output", "")
    if inp:
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{inst}\n\n### Response:\n{out}"

def print_trainable_params(model):
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(total, 1)
    if is_main():
        print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)", flush=True)

def wait_for_file(path: str, timeout_s: int = 3600, poll_s: float = 1.0):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_s)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--dataset", default="yahma/alpaca-cleaned")
    ap.add_argument("--run_name", default="sft_ddp_smoke")
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset_size", type=int, default=2048)

    # LoRA
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # memory knobs
    ap.add_argument("--gradient_checkpointing", type=str, default="True")
    ap.add_argument("--bf16", type=str, default="True")

    args = ap.parse_args()

    if is_main():
        print(
            f"TORCHRUN: RANK={os.environ.get('RANK')} "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
            f"HOST={os.environ.get('HOSTNAME', os.uname().nodename)}",
            flush=True,
        )

    torch.cuda.set_device(local_rank())
    device = torch.device("cuda", local_rank())

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    runs_root = os.environ.get("RUNS_ROOT", "results/training")
    out_dir = os.path.join(runs_root, args.run_name)

    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = args.bf16.lower() == "true"
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)

    use_gc = args.gradient_checkpointing.lower() == "true"
    if use_gc:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    # LoRA
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)
        print_trainable_params(model)

    # ---------- dataset caching (rank0 builds, others wait) ----------
    hf_home = os.environ.get("HF_HOME", "/tmp/hf")
    tok_cache_dir = os.path.join(
        hf_home, "tokenized",
        f"{args.dataset.replace('/','_')}_L{args.seq_len}_N{args.subset_size}",
    )
    done_flag = os.path.join(tok_cache_dir, ".DONE")

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

        Path(done_flag).touch()

    # everyone waits for rank0 to finish writing
    if is_dist():
        wait_for_file(done_flag, timeout_s=3600, poll_s=1.0)

    ds = load_from_disk(tok_cache_dir)

    # ---------- training ----------
    train_args = TrainingArguments(
        output_dir=out_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=1,
        save_steps=args.max_steps,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=use_gc,
        optim="adamw_torch",
        ddp_backend="nccl",
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=default_data_collator,
    )

    torch.cuda.synchronize()
    t0 = time.time()
    trainer.train()
    torch.cuda.synchronize()
    t1 = time.time()

    tokens = args.max_steps * args.bsz * args.seq_len
    toks_per_s = tokens / max(t1 - t0, 1e-9)

    if is_main():
        metrics = {
            "wall_time_s": t1 - t0,
            "approx_tokens_per_rank": tokens,
            "approx_tokens_per_s_per_rank": toks_per_s,
            "world_size": world_size(),
            "gpu0": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model": args.model,
            "dataset": args.dataset,
            "use_lora": bool(args.use_lora),
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("METRICS:", json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()

