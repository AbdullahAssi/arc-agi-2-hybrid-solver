"""Compare ways of choosing 2 attempts from a saved candidate pool (llm.infer --pool), offline.

The run's own ranking sums each candidate's log-probability over its views. A sum over tokens
favours small grids, since every extra cell adds a negative term; the rankers below test
whether normalising by size, or filtering by the size the demonstrations imply, picks better.

Usage: python -m llm.rank results/llm_pool_evaluation_<tag>.json [--split evaluation]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from solver.grid import load_split


def n_tokens(shape):
    """Answer tokens for a grid: one per cell, a newline per row, then END."""
    h, w = shape
    return h * (w + 1) + 1


def implied_shape(task, ti):
    """Output shape the demonstrations pin down for test input ti, or None if they don't."""
    pairs = task["train"]
    if all(x.shape == y.shape for x, y in pairs):
        return task["test"][ti][0].shape
    shapes = {y.shape for _, y in pairs}
    if len(shapes) == 1:
        return shapes.pop()
    return None


def _total(c):
    return sum(c["logprobs"])


def _per_token(c):
    return np.mean(c["logprobs"]) / n_tokens(np.array(c["grid"]).shape)


def _worst_view(c):
    return min(c["logprobs"]) / n_tokens(np.array(c["grid"]).shape)


RANKERS = {
    "votes": lambda c: (c["votes"], _total(c)),
    "logprob sum (current)": _total,
    "logprob per token": _per_token,
    "worst view per token": _worst_view,
    "votes, then per token": lambda c: (c["votes"], _per_token(c)),
}


def rank(cands, key, shape=None):
    """All candidates, best first, by key; with shape, candidates of the implied shape go first."""
    def full(c):
        k = key(c)
        k = k if isinstance(k, tuple) else (k,)
        return (shape is None or np.array(c["grid"]).shape == tuple(shape), *k)
    return sorted(cands, key=full, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--split", default="evaluation")
    args = ap.parse_args()

    tasks = load_split(args.split)
    pool = json.loads(open(args.pool).read())
    n_tasks = len({r["task"] for r in pool})
    tests_per_task = defaultdict(int)
    for r in pool:
        tests_per_task[r["task"]] += 1

    print(f"{len(pool)} test inputs over {n_tasks} tasks, "
          f"{sum(len(r['candidates']) for r in pool)} candidates")
    oracle = sum(any(c["correct"] for c in r["candidates"]) / tests_per_task[r["task"]] for r in pool)
    print(f"{'oracle (correct anywhere in pool)':40s} {oracle:5.2f}/{n_tasks}")

    in_pool = [r for r in pool if any(c["correct"] for c in r["candidates"])]
    for use_shape in (False, True):
        for name, key in RANKERS.items():
            score, ranks = 0.0, []
            for r in pool:
                shape = implied_shape(tasks[r["task"]], r["test"]) if use_shape else None
                order = rank(r["candidates"], key, shape)
                score += any(c["correct"] for c in order[:2]) / tests_per_task[r["task"]]
                if r in in_pool:
                    ranks.append(next(i for i, c in enumerate(order, 1) if c["correct"]))
            label = name + (" + implied shape" if use_shape else "")
            print(f"{label:40s} {score:5.2f}/{n_tasks}   rank of correct: {ranks}")

    for r in in_pool:
        print(f"\n{r['task']} test {r['test']}: implied shape {implied_shape(tasks[r['task']], r['test'])}")
        for c in sorted(r["candidates"], key=_total, reverse=True)[:8]:
            g = np.array(c["grid"])
            print(f"  {'*' if c['correct'] else ' '} {str(g.shape):9s} votes {c['votes']:2d}  "
                  f"sum {_total(c):8.2f}  per-token {_per_token(c):7.4f}")


if __name__ == "__main__":
    main()
