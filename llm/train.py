"""LoRA fine-tuning on the JSONL built by llm.data (loss on the completion only).

Usage: python -m llm.train --data data/llm_train.jsonl --out models/lora_q06 [--epochs 1]
       python -m llm.train --data data/llm_ttt.jsonl --init-adapter models/lora_q06 --out models/lora_ttt_q06
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode(tok, rows, max_len):
    """Token ids with labels masked (-100) over the prompt; drops over-long examples."""
    out = []
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(p) + len(c) <= max_len:
            out.append((p + c, [-100] * len(p) + c))
    return out


def batches(examples, token_budget, rng):
    """Shuffled batches of similar-length examples, padded size <= token_budget."""
    idx = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    groups, cur, longest = [], [], 0
    for i in idx:
        n = len(examples[i][0])
        if cur and max(longest, n) * (len(cur) + 1) > token_budget:
            groups.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        groups.append(cur)
    rng.shuffle(groups)
    return groups


def collate(examples, group, pad_id, device):
    longest = max(len(examples[i][0]) for i in group)
    ids = torch.full((len(group), longest), pad_id, dtype=torch.long)
    labels = torch.full((len(group), longest), -100, dtype=torch.long)
    mask = torch.zeros((len(group), longest), dtype=torch.long)
    for row, i in enumerate(group):
        x, y = examples[i]
        ids[row, :len(x)] = torch.tensor(x)
        labels[row, :len(y)] = torch.tensor(y)
        mask[row, :len(x)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def answer_loss(model, ids, mask, labels):
    """Mean cross-entropy over answer tokens, computing vocabulary logits only at those positions.

    Full logits (positions x 151k vocab, fp32) would dominate memory; the prompt positions
    carry no loss, so they never need logits.
    """
    causal_lm = model.get_base_model()  # LoRA layers are injected in place, so this still trains them
    hidden = causal_lm.model(input_ids=ids, attention_mask=mask).last_hidden_state
    keep = labels[:, 1:] != -100  # position t predicts token t+1
    logits = causal_lm.lm_head(hidden[:, :-1][keep]).float()
    return torch.nn.functional.cross_entropy(logits, labels[:, 1:][keep])


def answer_tokens(examples, group):
    return sum(sum(t != -100 for t in examples[i][1]) for i in group)


@torch.no_grad()
def evaluate(model, examples, token_budget, pad_id):
    model.eval()
    total, count = 0.0, 0
    for group in batches(examples, token_budget, random.Random(0)):
        ids, labels, mask = collate(examples, group, pad_id, model.device)
        n = answer_tokens(examples, group)
        total += answer_loss(model, ids, mask, labels).item() * n
        count += n
    model.train()
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default="data/llm_train.jsonl")
    ap.add_argument("--out", default="models/lora_q06")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--init-adapter", help="continue training this adapter instead of a fresh one")
    ap.add_argument("--token-budget", type=int, default=8192, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, help="stop early (smoke tests)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in open(args.data)]
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = encode(tok, rows[:n_val], args.max_len), encode(tok, rows[n_val:], args.max_len)
    print(f"train {len(train)}  val {len(val)}  "
          f"train tokens {sum(len(x) for x, _ in train) / 1e6:.1f}M", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(batches(train, args.token_budget, rng)) / args.accum)
    total_steps = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warmup = max(1, total_steps // 20)

    def lr_scale(s):
        """Linear warm-up over the first 5% of steps, then cosine decay to zero."""
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    print(f"{steps_per_epoch} optimizer steps/epoch, running {total_steps}", flush=True)

    model.train()
    step, t0, seen_tokens, loss_sum, loss_n = 0, time.perf_counter(), 0, 0.0, 0
    print(f"step 0 val_loss {evaluate(model, val, args.token_budget, tok.pad_token_id):.4f}", flush=True)
    while step < total_steps:
        groups = batches(train, args.token_budget, rng)
        for start in range(0, len(groups), args.accum):
            micro = groups[start:start + args.accum]
            n_answer = sum(answer_tokens(train, g) for g in micro)
            for g in micro:
                ids, labels, mask = collate(train, g, tok.pad_token_id, model.device)
                # Weight each micro-batch by its share of answer tokens so the step is a true mean
                loss = answer_loss(model, ids, mask, labels)
                (loss * answer_tokens(train, g) / n_answer).backward()
                loss_sum += loss.item()
                loss_n += 1
                seen_tokens += int(mask.sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                el = time.perf_counter() - t0
                print(f"step {step}/{total_steps} loss {loss_sum / loss_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{seen_tokens / el:.0f} tok/s  eta {el / step * (total_steps - step) / 60:.0f} min  "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
                loss_sum, loss_n = 0.0, 0
            if step % args.eval_every == 0 or step == total_steps:
                print(f"step {step} val_loss {evaluate(model, val, args.token_budget, tok.pad_token_id):.4f}", flush=True)
                model.save_pretrained(args.out)
            if step >= total_steps:
                break

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out}  ({(time.perf_counter() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
