"""Compare search arms: tasks solved within node budgets, nodes-to-solution, false positives.

Usage: python analyze.py results/training_holdout_uniform.json results/training_holdout_dream.json ...
"""
import json
import sys
from pathlib import Path

import numpy as np

BUDGETS = [100, 500, 2_000, 5_000, 20_000]


def summarize(path):
    rows = json.loads(Path(path).read_text())["tasks"]
    solved = [r for r in rows if r["score"] > 0]
    # A task counts as solved within budget B if its first fitting program appeared by node B
    within = {b: sum(r["first_hit"] <= b for r in solved) for b in BUDGETS}
    return {
        # e.g. training_hold_d4_dream -> "hold_d4_dream" (keeps depth so d3/d4 arms stay distinct)
        "arm": Path(path).stem.split("_", 1)[-1],
        "tasks": len(rows),
        "score": sum(r["score"] for r in rows),
        **{f"@{b}": within[b] for b in BUDGETS},
        "median_nodes": int(np.median([r["first_hit"] for r in solved])) if solved else None,
        "false_pos": sum(r["fit_train"] and r["score"] == 0 for r in rows),
    }, {r["task"]: r for r in solved}


def main(paths):
    summaries, solved_sets = zip(*(summarize(p) for p in paths))
    cols = list(summaries[0])
    width = {c: max(12, *(len(str(s[c])) for s in summaries)) for c in cols}
    print(" | ".join(f"{c:>{width[c]}}" for c in cols))
    for s in summaries:
        print(" | ".join(f"{str(s[c]):>{width[c]}}" for c in cols))

    # Per-task speed-up on tasks every arm solved
    common = set.intersection(*(set(s) for s in solved_sets))
    if len(paths) > 1 and common:
        base = solved_sets[0]
        print(f"\nTasks solved by all arms: {len(common)}")
        for s, name in zip(solved_sets[1:], [x["arm"] for x in summaries[1:]]):
            ratio = [base[t]["first_hit"] / s[t]["first_hit"] for t in common]
            print(f"  {name}: geometric-mean node speed-up vs {summaries[0]['arm']} = "
                  f"{np.exp(np.mean(np.log(ratio))):.1f}x")
        print("\nPer task (nodes to first solution):")
        for t in sorted(set().union(*solved_sets)):
            cells = [str(s[t]["first_hit"]) if t in s else "-" for s in solved_sets]
            prog = next(s[t]["programs"][0] for s in solved_sets if t in s)
            print(f"  {t}  " + "  ".join(f"{c:>6}" for c in cells) + f"   {prog}")


if __name__ == "__main__":
    main(sys.argv[1:])
