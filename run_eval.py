"""Evaluate the solver on a public ARC-AGI-2 split.

Usage: python run_eval.py --split evaluation [--prior models/prior.joblib] [--max-nodes 20000]
                           [--holdout] [--limit N] [--tag NAME]
Writes results/<split>_<tag>.json with per-task scores, programs and search statistics.
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np

from solver.grid import load_split
from solver.prior import is_holdout, primitive_costs
from solver.search import solve

RESULTS = Path(__file__).parent / "results"
_model = None


def _init_worker(prior_path):
    global _model
    # joblib unpickles: only load priors trained locally by solver.prior, never downloaded files
    _model = joblib.load(prior_path) if prior_path else None


def score(task, attempts):
    """ARC scoring: fraction of test inputs where either attempt matches exactly."""
    hits = [any(a.shape == y.shape and np.array_equal(a, y) for a in tries)
            for (_, y), tries in zip(task["test"], attempts)]
    return sum(hits) / len(hits)


def run_one(args):
    task_id, task, opts = args
    t0 = time.perf_counter()
    costs = primitive_costs(_model, task) if _model else None
    res = solve(task, costs=costs, max_depth=opts["max_depth"], max_nodes=opts["max_nodes"],
                time_limit=opts["time_limit"])
    return {
        "task": task_id,
        "score": score(task, res.attempts),
        "fit_train": bool(res.programs),
        "programs": [" > ".join(p) or "identity" for p in res.programs],
        "nodes": res.nodes,
        "first_hit": res.first_hit,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=20_000)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--prior", help="learned prior (joblib); omit for uniform breadth-first search")
    ap.add_argument("--holdout", action="store_true", help="training split: only tasks the prior never saw")
    ap.add_argument("--limit", type=int, help="only the first N tasks")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    tasks = [(tid, t) for tid, t in load_split(args.split).items() if not args.holdout or is_holdout(tid)]
    tasks = tasks[: args.limit]
    opts = {"max_depth": args.max_depth, "max_nodes": args.max_nodes, "time_limit": args.time_limit}
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.prior,)) as pool:
        rows = pool.map(run_one, [(tid, t, opts) for tid, t in tasks], chunksize=1)

    solved = [r for r in rows if r["score"] > 0]
    fit = [r for r in rows if r["fit_train"]]
    total = sum(r["score"] for r in rows)
    print(f"{args.split} [{args.tag}]: score {total:.2f}/{len(rows)} ({total / len(rows):.1%})  "
          f"fit-train {len(fit)}  false-positives {len(fit) - len(solved)}  "
          f"time {time.perf_counter() - t0:.0f}s")
    for r in solved:
        print(f"  {r['task']}  {r['score']:.2f}  nodes={r['first_hit']:<6} {r['programs'][0]}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.split}_{args.tag}.json"
    out.write_text(json.dumps({"args": vars(args), "score": total, "tasks": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
