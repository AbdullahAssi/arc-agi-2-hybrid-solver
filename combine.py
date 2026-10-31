"""Combine the DSL search and the LLM into one solver and write a Kaggle-format submission.

Rule: when the DSL found a program that fits every train pair (right 93 times out of 107 on the
training split), its predictions come first; the LLM's top-voted grids fill the remaining
attempts; with neither, the test input is echoed. DSL predictions are recomputed only for the
tasks the saved DSL run fitted, so this takes seconds.

Usage: python combine.py --split evaluation --dsl results/evaluation_v2_uniform.json
                         --llm results/llm_evaluation_ttt_task_q06.json --out results/submission_evaluation.json
"""
import argparse
import json
from collections import Counter

import numpy as np

from solver.grid import load_split
from solver.search import solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--dsl", default="results/evaluation_v2_uniform.json")
    ap.add_argument("--llm", default="results/llm_evaluation_ttt_task_q06.json")
    ap.add_argument("--out", default="results/submission_evaluation.json")
    args = ap.parse_args()

    tasks = load_split(args.split)
    dsl = json.load(open(args.dsl))
    llm = {t["task"]: t["tests"] for t in json.load(open(args.llm))["tasks"]}
    search = {k: dsl["args"][k] for k in ("max_depth", "max_nodes", "time_limit")}

    submission, sources, score = {}, Counter(), 0.0
    for row in dsl["tasks"]:
        tid, task = row["task"], tasks[row["task"]]
        dsl_attempts = solve(task, **search).attempts if row["fit_train"] else [[] for _ in task["test"]]
        entries, hits = [], []
        for ti, (x, y) in enumerate(task["test"]):
            llm_attempts = [np.array(a) for a in llm[tid][ti]["attempts"]] if tid in llm else []
            ranked = [("dsl", a) for a in dsl_attempts[ti]] + [("llm", a) for a in llm_attempts] + [("copy", x)]
            chosen = []
            for src, a in ranked:  # first two distinct grids
                if not any(a.shape == b.shape and np.array_equal(a, b) for _, b in chosen):
                    chosen.append((src, a))
            chosen = chosen[:2] if len(chosen) > 1 else chosen * 2
            sources[chosen[0][0]] += 1
            entries.append({f"attempt_{i + 1}": a.tolist() for i, (_, a) in enumerate(chosen)})
            if y is not None:
                hit = next((src for src, a in chosen if a.shape == y.shape and np.array_equal(a, y)), None)
                hits.append(hit)
        submission[tid] = entries
        if hits:
            score += sum(h is not None for h in hits) / len(hits)
            for h in hits:
                if h:
                    print(f"  solved {tid} via {h}")

    with open(args.out, "w") as f:
        json.dump(submission, f)
    print(f"{args.split}: combined score {score:.2f}/{len(submission)}  "
          f"(attempt 1 from: {dict(sources)})  -> {args.out}")


if __name__ == "__main__":
    main()
