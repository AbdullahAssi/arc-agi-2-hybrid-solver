"""Per-task test-time training: one small LoRA adapter per evaluation task.

Each task's demonstration pairs (never its test outputs) become leave-one-out examples under many
augmentations; the base adapter is trained on them for a few steps and saved as <out>/<task id>.
The model is loaded once and the LoRA weights are reset to the base adapter before every task.
Resumable: tasks whose adapter already exists are skipped.

Usage: python -m llm.ttt --init-adapter models/lora_q06 --out models/ttt_tasks [--augs 16 --epochs 2 --mix 0]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from solver.grid import load_split

from .data import leave_one_out
from .train import answer_loss, answer_tokens, batches, collate, encode


def train_task(model, examples, pad_id, args, rng):
    """A short cosine-decayed run over one task's examples; returns the mean loss per epoch."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    epochs = [batches(examples, args.token_budget, rng) for _ in range(args.epochs)]
    total = sum(len(e) for e in epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    losses = []
    for groups in epochs:
        epoch_loss = []
        for g in groups:
            ids, labels, mask = collate(examples, g, pad_id, model.device)
            loss = answer_loss(model, ids, mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            epoch_loss.append(loss.item())
        losses.append(float(np.mean(epoch_loss)))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--init-adapter", default="models/lora_q06")
    ap.add_argument("--out", default="models/ttt_tasks")
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--augs", type=int, default=16, help="augmented copies per left-out pair")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mix", type=int, default=0, help="generic training rows mixed into each task's examples")
    ap.add_argument("--mix-from", default="data/llm_train.jsonl")
    ap.add_argument("--token-budget", type=int, default=8192, help="padded tokens per optimizer step")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    model.train()
    base = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    generic = [json.loads(line) for line in open(args.mix_from)] if args.mix else []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = dict(list(load_split(args.split).items())[: args.limit])
    t0 = time.perf_counter()
    for k, (tid, task) in enumerate(tasks.items(), 1):
        if (out / tid / "adapter_config.json").exists():
            continue
        rng = np.random.default_rng([args.seed, k])
        rows = leave_one_out(task["train"], rng, args.augs, args.max_len, min_demos=1)
        examples = encode(tok, rows, args.max_len)
        if not examples:
            print(f"[{k}/{len(tasks)}] {tid}: no example fits {args.max_len} tokens, keeps the base adapter", flush=True)
            continue
        if args.mix:  # keeps the output format from drifting while fitting a handful of pairs
            examples += encode(tok, [generic[i] for i in rng.choice(len(generic), args.mix, replace=False)], args.max_len)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in base:
                    p.copy_(base[n])
        start = time.perf_counter()
        losses = train_task(model, examples, tok.pad_token_id, args, random.Random(k))
        model.save_pretrained(out / tid)
        (out / tid / "ttt.json").write_text(json.dumps({"examples": len(examples), "losses": losses,
                                                        "seconds": time.perf_counter() - start}))
        elapsed = time.perf_counter() - t0
        print(f"[{k}/{len(tasks)}] {tid}: {len(examples)} examples, loss "
              f"{' -> '.join(f'{l:.3f}' for l in losses)}  {time.perf_counter() - start:.0f}s  "
              f"(total {elapsed / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
