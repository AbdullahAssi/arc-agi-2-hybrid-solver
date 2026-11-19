"""Run a (optionally LoRA-adapted) causal LM on ARC tasks with augmentation voting.

For every test input, the task is presented under several augmentations; each completion is
parsed, mapped back through the inverse augmentation, and the two most-voted grids become
attempts 1 and 2.

Usage: python -m llm.infer --model Qwen/Qwen3-0.6B --split evaluation --augs 8 [--adapter DIR]
                           [--adapter-dir DIR] [--engine vllm|hf] --tag NAME
--adapter-dir holds one adapter per task (DIR/<task id>, from llm.ttt); tasks without one use --adapter.
Writes results/llm_<split>_<tag>.json. The vllm engine runs in .venv-vllm, hf in .venv-llm.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from solver.grid import load_split

from .format import IDENTITY, END, make_prompt, make_target, n_tokens_estimate, random_aug, text_to_grid

RESULTS = Path(__file__).resolve().parents[1] / "results"
MAX_NEW_TOKENS = 31 * 30 + 4  # a 30x30 grid with newlines, plus END


class VllmEngine:
    """vLLM with one LoRA directory (or None) per prompt; batches and schedules memory itself."""

    def __init__(self, model_name, adapters, max_len):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        paths = sorted({a for a in adapters if a})
        # A smaller prefill chunk bounds the prompt-logprob tensor (chunk x 151k vocab) when rescoring
        self.llm = LLM(model=model_name, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.8,
                       enable_lora=bool(paths), max_lora_rank=64, enable_prefix_caching=True,
                       max_num_batched_tokens=4096,
                       **({"max_loras": min(16, len(paths)), "max_cpu_loras": len(paths)} if paths else {}))
        self.requests = {p: LoRARequest(Path(p).name, i + 1, p) for i, p in enumerate(paths)}

    def _lora(self, adapters):
        return [self.requests[a] if a else None for a in adapters] if self.requests else None

    def generate(self, prompts, adapters, n=1, temperature=0.0, seed=0):
        """n completions per prompt (greedy when temperature is 0), stopping at END."""
        from vllm import SamplingParams

        params = SamplingParams(n=n, temperature=temperature, seed=seed if temperature else None,
                                max_tokens=MAX_NEW_TOKENS, stop=[END])
        outputs = self.llm.generate(prompts, params, lora_request=self._lora(adapters))
        if n == 1:
            return [o.outputs[0].text for o in outputs]
        return [[c.text for c in o.outputs] for o in outputs]

    def logprob(self, prompts, completions, adapters):
        """Total log-probability of each completion given its prompt (tokenised separately, as in training)."""
        from vllm import SamplingParams

        tok = self.llm.get_tokenizer()
        inputs, starts = [], []
        for p, c in zip(prompts, completions):
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            inputs.append({"prompt_token_ids": p_ids + tok(c, add_special_tokens=False)["input_ids"]})
            starts.append(len(p_ids))
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outputs = self.llm.generate(inputs, params, lora_request=self._lora(adapters))
        return [sum(o.prompt_logprobs[i][o.prompt_token_ids[i]].logprob for i in range(start, len(o.prompt_token_ids)))
                for o, start in zip(outputs, starts)]


def rescore(engine, jobs, adapters, votes, top_k):
    """Rank each test's top_k voted grids by their summed log-probability over all of its views.

    The model's own likelihood of a candidate, under every augmentation, is a finer signal than
    how many greedy decodes happened to land on it.
    """
    views = {}
    for job, adapter in zip(jobs, adapters):
        views.setdefault((job["task"], job["test"]), []).append((job, adapter))
    prompts, targets, owners, keys = [], [], [], []
    for k, counter in votes.items():
        for cand, _ in counter.most_common(top_k):
            shape, raw = cand
            grid = np.frombuffer(raw, dtype=np.int8).reshape(shape)
            for job, adapter in views[k]:
                prompts.append(job["prompt"])
                targets.append(make_target(job["aug"].grid(grid)))
                owners.append(adapter)
                keys.append((k, cand))
    print(f"rescoring {len(prompts)} candidate views")
    per_view = {}
    for key, lp in zip(keys, engine.logprob(prompts, targets, owners)):
        per_view.setdefault(key, []).append(lp)
    order = {}
    for (k, cand), lps in per_view.items():
        order.setdefault(k, []).append((sum(lps), cand))
    return {k: [c for _, c in sorted(v, key=lambda t: -t[0])] for k, v in order.items()}, per_view


def dump_pool(path, tasks, votes, per_view):
    """Every rescored candidate with its votes and per-view log-probabilities, for offline ranking."""
    rows = []
    for (tid, ti), counter in votes.items():
        y = tasks[tid]["test"][ti][1]
        cands = []
        for cand, n in counter.most_common():
            if (k := ((tid, ti), cand)) not in per_view:
                continue
            shape, raw = cand
            grid = np.frombuffer(raw, dtype=np.int8).reshape(shape)
            cands.append({"grid": grid.tolist(), "votes": n, "logprobs": per_view[k],
                          "correct": bool(y is not None and grid.shape == y.shape and np.array_equal(grid, y))})
        rows.append({"task": tid, "test": ti, "candidates": cands})
    path.write_text(json.dumps(rows))
    print(f"wrote {path} ({sum(len(r['candidates']) for r in rows)} candidates)")


def load_model(name, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def build_jobs(tasks, n_augs, max_tokens, seed):
    """One job per (task, test index, augmentation)."""
    rng = np.random.default_rng(seed)
    jobs = []
    for tid, task in tasks.items():
        for ti, (query, _) in enumerate(task["test"]):
            augs = [IDENTITY] + [random_aug(rng, len(task["train"])) for _ in range(n_augs - 1)]
            for aug in augs:
                prompt = make_prompt(aug.pairs(task["train"]), aug.grid(query))
                if n_tokens_estimate(task["train"], query) > max_tokens:
                    continue
                jobs.append({"task": tid, "test": ti, "aug": aug, "prompt": prompt})
    return jobs


def token_batches(prompts, max_batch, token_budget, max_new_tokens, attn_budget=1.6e8):
    """Group prompts (longest first) so each batch stays within two memory budgets.

    token_budget bounds padded length x batch (activations, KV cache); attn_budget bounds
    length^2 x batch (attention scores, materialised when padding masks are present).
    Prompts are mostly digits and newlines, so characters approximate tokens well enough.
    """
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    batch, longest = [], 0
    for i in order:
        longest_if_added = max(longest, len(prompts[i]) + max_new_tokens)
        n = len(batch) + 1
        if batch and (len(batch) == max_batch or longest_if_added * n > token_budget
                      or longest_if_added ** 2 * n > attn_budget):
            yield batch
            batch, longest_if_added = [], len(prompts[i]) + max_new_tokens
        batch.append(i)
        longest = longest_if_added
    if batch:
        yield batch


def generate(tok, model, prompts, max_batch, max_new_tokens, token_budget):
    import torch

    out, done = [None] * len(prompts), 0
    for idx in token_batches(prompts, max_batch, token_budget, max_new_tokens):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 stop_strings=[END], tokenizer=tok, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = t
        done += len(idx)
        print(f"  generated {done}/{len(prompts)} (batch {len(idx)})", flush=True)
    return out


def vote(jobs, completions):
    """Group parsed, de-augmented grids by (task, test) and rank them by vote count."""
    votes, valid = {}, Counter()
    for job, text in zip(jobs, completions):
        g = text_to_grid(text)
        k = (job["task"], job["test"])
        votes.setdefault(k, Counter())
        if g is None:
            continue
        valid[k] += 1
        g = job["aug"].invert(g)
        votes[k][(g.shape, g.tobytes())] += 1
    return votes, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", help="LoRA adapter directory")
    ap.add_argument("--adapter-dir", help="per-task adapters, DIR/<task id>; overrides --adapter where present")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--augs", type=int, default=8, help="augmented views per test input (first is identity)")
    ap.add_argument("--max-tokens", type=int, default=8192, help="skip tasks with longer prompts")
    ap.add_argument("--batch-size", type=int, default=32, help="max sequences per batch")
    ap.add_argument("--token-budget", type=int, default=60_000, help="max padded tokens per batch")
    ap.add_argument("--rescore", type=int, default=0, metavar="K",
                    help="vllm only: rank each test's top K voted grids by log-probability across its views")
    ap.add_argument("--samples", type=int, default=0,
                    help="vllm only: extra sampled completions per view, added to the candidate pool")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--pool", action="store_true",
                    help="with --rescore: also write every rescored candidate to results/llm_pool_<split>_<tag>.json")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = dict(list(load_split(args.split).items())[: args.limit])
    jobs = build_jobs(tasks, args.augs, args.max_tokens, args.seed)
    prompts = [j["prompt"] for j in jobs]
    print(f"{len(tasks)} tasks, {len(jobs)} generations")

    t0 = time.perf_counter()
    if args.engine == "vllm":
        adapters = [args.adapter] * len(jobs)
        if args.adapter_dir:
            own = {j["task"] for j in jobs if (Path(args.adapter_dir) / j["task"] / "adapter_config.json").exists()}
            adapters = [str(Path(args.adapter_dir) / j["task"]) if j["task"] in own else args.adapter for j in jobs]
            print(f"per-task adapters for {len(own)}/{len(tasks)} tasks")
        engine = VllmEngine(args.model, adapters, args.max_tokens + MAX_NEW_TOKENS + 512)
        completions = engine.generate(prompts, adapters)
    else:
        tok, model = load_model(args.model, args.adapter)
        completions = generate(tok, model, prompts, args.batch_size, MAX_NEW_TOKENS, args.token_budget)
    votes, valid = vote(jobs, completions)
    if args.engine == "vllm" and args.samples:
        # Sampled answers widen the candidate pool; valid-grid counts stay greedy-only for comparability
        sampled = engine.generate(prompts, adapters, n=args.samples, temperature=args.temperature, seed=args.seed)
        flat_jobs = [j for j, texts in zip(jobs, sampled) for _ in texts]
        extra, _ = vote(flat_jobs, [t for texts in sampled for t in texts])
        for k, counter in extra.items():
            votes.setdefault(k, Counter()).update(counter)
        print(f"sampled {len(flat_jobs)} extra completions (temperature {args.temperature})")

    order, vote_total = None, None
    if args.rescore:
        order, per_view = rescore(engine, jobs, adapters, votes, args.rescore)
        if args.pool:
            RESULTS.mkdir(exist_ok=True)
            dump_pool(RESULTS / f"llm_pool_{args.split}_{args.tag}.json", tasks, votes, per_view)
        vote_total = summarize(tasks, jobs, votes, valid)[1]
        print(f"by votes alone: {vote_total:.2f}/{len(tasks)}")
    rows, total = summarize(tasks, jobs, votes, valid, order)
    elapsed = time.perf_counter() - t0
    n_valid = sum(r["valid"] for r in rows)
    pool = sum(t["in_pool"] for r in rows for t in r["tests"])
    print(f"correct grid among the candidates for {pool}/{sum(len(r['tests']) for r in rows)} test inputs")
    print(f"{args.split} [{args.tag}]: score {total:.2f}/{len(tasks)} ({total / len(tasks):.1%})  "
          f"valid grids {n_valid}/{len(jobs)}  time {elapsed:.0f}s")
    for r in rows:
        if r["score"] > 0:
            print(f"  solved {r['task']} votes={r['top_votes']}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"llm_{args.split}_{args.tag}.json"
    out.write_text(json.dumps({"args": vars(args), "score": total, "vote_score": vote_total, "seconds": elapsed,
                               "tasks": rows}, indent=1))
    print(f"wrote {out}")


def summarize(tasks, jobs, votes, valid, order=None):
    """Per-task scores plus per-test detail (votes, near-miss accuracy, attempt grids).

    order: optional ranking of candidates per (task, test), replacing vote order (see rescore).
    """
    views = Counter((j["task"], j["test"]) for j in jobs)
    rows, total = [], 0.0
    for tid, task in tasks.items():
        hits, n_valid, top_votes, tests = [], 0, [], []
        for ti, (_, y) in enumerate(task["test"]):
            counter = votes.get((tid, ti), Counter())
            if order and (tid, ti) in order:
                ranked = [(c, counter[c]) for c in order[(tid, ti)][:2]]
            else:
                ranked = counter.most_common(2)
            attempts = [np.frombuffer(b, dtype=np.int8).reshape(s) for (s, b), _ in ranked]
            hit = any(a.shape == y.shape and np.array_equal(a, y) for a in attempts)
            hits.append(hit)
            n_valid += valid[(tid, ti)]
            top_votes.append([c for _, c in ranked])
            top = attempts[0] if attempts else None
            shape_ok = top is not None and top.shape == y.shape
            in_pool = (y.shape, y.astype(np.int8).tobytes()) in counter
            # Per-test detail for analysis: near misses, vote agreement, the grids themselves
            tests.append({
                "correct": hit,
                "views": views[(tid, ti)],
                "valid": valid[(tid, ti)],
                "votes": [c for _, c in ranked],
                "shape_ok": shape_ok,
                "in_pool": in_pool,
                "cell_acc": float((top == y).mean()) if shape_ok else None,
                "attempts": [a.tolist() for a in attempts],
            })
        score = sum(hits) / len(hits)
        total += score
        rows.append({"task": tid, "score": score, "valid": n_valid, "top_votes": top_votes, "tests": tests})
    return rows, total


if __name__ == "__main__":
    main()
