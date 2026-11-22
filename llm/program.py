"""Program synthesis with a local LLM: ask for a Python transform(grid), keep it only if it reproduces
every demonstration exactly, then apply it to the test inputs.

Unlike the fine-tuned grid model, a candidate here is checked from outside the model: a program
that gets any demonstration wrong is discarded, however confident the model was. A failed program
is sent back once per round with the first mismatch, so the model can revise it.

Talks to Ollama (http://localhost:11434). Generated code runs in a separate process with a timeout
and a memory cap, but is not otherwise sandboxed: only use on a machine you control.

Usage: python -m llm.program --model gpt-oss:20b --split evaluation --sample 20 --attempts 2 --rounds 2 --tag NAME
Appends one JSON line per task to results/prog_<split>_<tag>.jsonl (resumable), then prints a summary.
"""
import argparse
import http.client
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from solver.grid import load_split

from .format import grid_to_text

RESULTS = Path(__file__).resolve().parents[1] / "results"

SYSTEM = """You solve ARC puzzles by writing Python.
Each puzzle gives example pairs of input and output grids. Grids are rectangles of integers 0-9 (colours),
written one row per line. Find the single rule that turns every example input into its output, and
write it as a function:

def transform(grid: list[list[int]]) -> list[list[int]]:

The function must work for all the examples and for the new test inputs, which may differ in size.
You may import numpy (as np) and the Python standard library. Answer with one ```python code block."""

# Runs untrusted generated code: reads {"code", "inputs"} on stdin, prints outputs or the error per input.
RUNNER = r"""
import json, sys, traceback
job = json.load(sys.stdin)
ns = {}
try:
    exec(job["code"], ns)
    fn = ns["transform"]
except Exception:
    print(json.dumps({"error": traceback.format_exc(limit=2)[-800:]}))
    sys.exit()
outs = []
for g in job["inputs"]:
    try:
        o = fn([row[:] for row in g])
        o = [[int(v) for v in row] for row in (o.tolist() if hasattr(o, "tolist") else o)]
        outs.append({"grid": o})
    except Exception:
        outs.append({"error": traceback.format_exc(limit=2)[-800:]})
print(json.dumps({"outputs": outs}))
"""


def _limit_memory():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))


def run_program(code, inputs, timeout=10.0):
    """Outputs of transform() on each input (a grid, or an error string), or one error for the whole program."""
    job = json.dumps({"code": code, "inputs": [g.tolist() for g in inputs]})
    # One BLAS thread: on a many-core machine OpenBLAS reserves memory per core, and under the memory
    # cap `import numpy` then hangs until the timeout
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=job, capture_output=True, text=True,
                              timeout=timeout, env=env,
                              preexec_fn=_limit_memory if sys.platform != "win32" else None)
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout:.0f}s"
    try:
        res = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return f"crashed: {proc.stderr[-500:]}"
    if "error" in res:
        return res["error"]
    return [_as_grid(o["grid"]) if "grid" in o else o["error"] for o in res["outputs"]]


def _as_grid(rows):
    if not rows or not all(isinstance(r, list) and len(r) == len(rows[0]) for r in rows) or not rows[0]:
        return "returned a non-rectangular or empty grid"
    g = np.array(rows)
    if g.min() < 0 or g.max() > 9 or max(g.shape) > 30:
        return f"returned values outside 0-9 or a grid larger than 30x30 (shape {g.shape})"
    return g.astype(np.int8)


def task_prompt(task):
    parts = []
    for i, (x, y) in enumerate(task["train"], 1):
        parts.append(f"Example {i} input ({x.shape[0]}x{x.shape[1]}):\n{grid_to_text(x)}\n"
                     f"Example {i} output ({y.shape[0]}x{y.shape[1]}):\n{grid_to_text(y)}\n")
    for i, (x, _) in enumerate(task["test"], 1):
        parts.append(f"Test input {i} ({x.shape[0]}x{x.shape[1]}):\n{grid_to_text(x)}\n")
    parts.append("Write transform() for this puzzle.")
    return "\n".join(parts)


def feedback(task, outs):
    """Describe the first demonstration the program gets wrong."""
    if isinstance(outs, str):
        return f"Your program failed to run:\n{outs}\nFix it."
    for i, ((x, y), o) in enumerate(zip(task["train"], outs), 1):
        if isinstance(o, str):
            return f"On example {i} your program raised:\n{o}\nFix it."
        if o.shape != y.shape:
            return (f"On example {i} your program returned a {o.shape[0]}x{o.shape[1]} grid, but the expected "
                    f"output is {y.shape[0]}x{y.shape[1]}:\n{grid_to_text(y)}\nRevise the rule.")
        if not np.array_equal(o, y):
            wrong = int((o != y).sum())
            return (f"On example {i} your program got {wrong} of {y.size} cells wrong. It returned:\n"
                    f"{grid_to_text(o)}\nExpected:\n{grid_to_text(y)}\nRevise the rule.")
    return None


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else text
    return code if "def transform" in code else None


def chat(model, messages, think, num_ctx, temperature, host, max_tokens):
    # num_predict caps reasoning plus answer: without it a long think can fill the context and leave no code
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature, "num_predict": max_tokens}}
    if think:
        body["think"] = think if think in ("low", "medium", "high") else True
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # The server may restart mid-run: retry connection failures for up to ~15 minutes before giving up
    for wait in [10, 20, 30, 60, 120, 240, 420, 0]:
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                d = json.loads(resp.read())
            return d["message"].get("content", ""), d.get("eval_count", 0), d.get("prompt_eval_count", 0)
        except (ConnectionError, urllib.error.URLError, http.client.HTTPException) as e:
            if not wait:
                raise
            print(f"  server unreachable ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def solve_task(tid, task, args):
    """Up to `attempts` fresh conversations, each revised up to `rounds - 1` times; stop at the first pass.

    Without a pass, the program that got the most demonstrations right (then the fewest cells wrong)
    supplies fallback predictions, kept apart from the score.
    """
    train_in = [x for x, _ in task["train"]]
    test_in = [x for x, _ in task["test"]]
    log, t0, best = [], time.perf_counter(), None
    for attempt in range(args.attempts):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_prompt(task)}]
        for rnd in range(args.rounds):
            t = time.perf_counter()
            try:
                text, n_out, n_in = chat(args.model, messages, args.think, args.num_ctx, args.temperature, args.host,
                                         args.max_tokens)
            except Exception as e:  # server error or timeout: record it and move on to the next attempt
                log.append({"attempt": attempt, "round": rnd, "status": "llm_error", "error": str(e)[:300]})
                break
            entry = {"attempt": attempt, "round": rnd, "seconds": round(time.perf_counter() - t, 1),
                     "tokens_out": n_out, "tokens_in": n_in}
            code = extract_code(text)
            if code is None:
                entry["status"] = "no_code"
                log.append(entry)
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "Answer with a ```python block defining transform(grid)."}]
                continue
            outs = run_program(code, train_in + test_in)
            train_outs = outs if isinstance(outs, str) else outs[: len(train_in)]
            msg = feedback(task, train_outs)
            entry["code"] = code
            if isinstance(outs, list):
                entry["train_correct"] = sum(not isinstance(o, str) and o.shape == y.shape and np.array_equal(o, y)
                                             for o, (_, y) in zip(outs, task["train"]))
                wrong = sum(int((o != y).sum()) if not isinstance(o, str) and o.shape == y.shape else y.size
                            for o, (_, y) in zip(outs, task["train"]))
                rank = (entry["train_correct"], -wrong)
                if all(isinstance(p, np.ndarray) for p in outs[len(train_in):]) and (best is None or rank > best[0]):
                    best = (rank, outs[len(train_in):])
            if msg is None:
                preds = outs[len(train_in):]
                entry["status"] = "pass"
                log.append(entry)
                return _row(tid, task, log, preds, t0)
            entry["status"] = "fail"
            log.append(entry)
            messages += [{"role": "assistant", "content": f"```python\n{code}```"}, {"role": "user", "content": msg}]
    return _row(tid, task, log, None, t0, fallback=best[1] if best else None)


def _hits(task, preds):
    return [y is not None and preds is not None and isinstance(preds[i], np.ndarray)
            and preds[i].shape == y.shape and bool(np.array_equal(preds[i], y))
            for i, (_, y) in enumerate(task["test"])]


def _grids(preds):
    return [p.tolist() if isinstance(p, np.ndarray) else None for p in preds] if preds else []


def _row(tid, task, log, preds, t0, fallback=None):
    hits, fb_hits = _hits(task, preds), _hits(task, fallback)
    return {"task": tid, "fit_train": preds is not None, "score": sum(hits) / len(hits),
            "fallback_score": sum(fb_hits) / len(fb_hits),
            "predictions": _grids(preds), "fallback": _grids(fallback),
            "seconds": round(time.perf_counter() - t0, 1), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--sample", type=int, help="random subset of N tasks (fixed by --seed)")
    ap.add_argument("--tasks", nargs="*", help="explicit task ids")
    ap.add_argument("--attempts", type=int, default=2, help="fresh conversations per task")
    ap.add_argument("--rounds", type=int, default=2, help="LLM calls per conversation (1 = no revision)")
    ap.add_argument("--think", default="medium", help="gpt-oss: low|medium|high; qwen3: on; '' to disable")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=16384, help="cap on reasoning plus answer tokens per call")
    ap.add_argument("--max-prompt-chars", type=int, default=40_000, help="skip tasks with longer prompts")
    ap.add_argument("--workers", type=int, default=1, help="concurrent tasks (match Ollama's OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    tasks = load_split(args.split)
    ids = args.tasks or sorted(tasks)
    if args.sample:
        ids = sorted(random.Random(args.seed).sample(ids, args.sample))
    out = RESULTS / f"prog_{args.split}_{args.tag}.jsonl"
    RESULTS.mkdir(exist_ok=True)
    if out.exists():  # resume; unsolved tasks cut short by an unreachable server are run again in full
        kept = [line for line in out.open()
                if (r := json.loads(line))["fit_train"] or all(e["status"] != "llm_error" for e in r["log"])]
        out.write_text("".join(kept))
    done = {json.loads(line)["task"] for line in out.open()} if out.exists() else set()
    todo = [t for t in ids if t not in done and len(task_prompt(tasks[t])) <= args.max_prompt_chars]
    print(f"{len(ids)} tasks, {len(done & set(ids))} already done, {len(todo)} to run "
          f"({len(ids) - len(done & set(ids)) - len(todo)} skipped as too long)", flush=True)

    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as f:
        for row in pool.map(lambda t: solve_task(t, tasks[t], args), todo):
            f.write(json.dumps(row) + "\n")
            f.flush()
            last = row["log"][-1]["status"] if row["log"] else "-"
            print(f"  {row['task']}: score {row['score']:.2f}  fit {row['fit_train']}  calls {len(row['log'])}  "
                  f"last {last}  {row['seconds']:.0f}s", flush=True)

    rows = [r for r in map(json.loads, out.open()) if r["task"] in set(ids)]
    calls = [e for r in rows for e in r["log"] if "seconds" in e]
    print(f"{args.split} [{args.tag}]: score {sum(r['score'] for r in rows):.2f}/{len(rows)}  "
          f"fit-train {sum(r['fit_train'] for r in rows)}  "
          f"false positives {sum(r['fit_train'] and r['score'] < 1 for r in rows)}  "
          f"fallback {sum(r.get('fallback_score', 0) for r in rows):.2f}  "
          f"calls {len(calls)}  mean {np.mean([e['seconds'] for e in calls]):.0f}s/call  "
          f"{np.mean([e['tokens_out'] for e in calls]):.0f} tokens/call")


if __name__ == "__main__":
    main()
