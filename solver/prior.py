"""Learned prior over DSL primitives: P(primitive appears in the program | task features).

Trained only on the training split:
  * "dream" data: random DSL programs run on real training-split input grids, labelled with
    the primitives they used (no human annotation needed);
  * "wake" data (optional): programs the search actually found on real training tasks.
Tasks in HOLDOUT (1 in 5 training tasks) are never used, so they can serve for ablations.

Usage: python -m solver.prior --n 30000 [--wake results/training_baseline.json] --out models/prior.joblib
"""
import argparse
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dsl import MARK, PRIMITIVES
from .features import task_features
from .grid import load_split

NAMES = list(PRIMITIVES)
MIN_COST = 0.1


def is_holdout(task_id):
    return zlib.crc32(task_id.encode()) % 5 == 0


def dream_task(rng, inputs, max_depth=3):
    """Run a random program on a task's inputs; return (pairs, primitive names) or None."""
    program = rng.choice(NAMES, size=rng.integers(1, max_depth + 1))
    outs = []
    for g in inputs:
        for name in program:
            g = PRIMITIVES[name](g)
            if g is None:
                return None
        outs.append(np.ascontiguousarray(g))
    if all(o.shape == i.shape and np.array_equal(o, i) for o, i in zip(outs, inputs)):
        return None
    mark_color = rng.integers(1, 10)
    outs = [np.where(o == MARK, mark_color, o).astype(np.int8) for o in outs]
    return list(zip(inputs, outs)), set(program)


def _dream_batch(args):
    seed, n, sources = args
    rng = np.random.default_rng(seed)
    X, Y = [], []
    while len(X) < n:
        inputs = sources[rng.integers(len(sources))]
        sample = dream_task(rng, inputs)
        if sample is None:
            continue
        pairs, used = sample
        X.append(task_features(pairs))
        Y.append([name in used for name in NAMES])
    return X, Y


def wake_data(results_path, tasks):
    """Features/labels from programs the search found on (non-holdout) training tasks."""
    X, Y = [], []
    for row in json.loads(Path(results_path).read_text())["tasks"]:
        if row["score"] == 0 or is_holdout(row["task"]) or row["programs"][0] == "identity":
            continue
        used = set(row["programs"][0].split(" > "))
        X.append(task_features(tasks[row["task"]]["train"]))
        Y.append([name in used for name in NAMES])
    return X, Y


def train(n, wake=None, wake_weight=20, seed=0):
    tasks = load_split("training")
    sources = [[x for x, _ in t["train"]] + [x for x, _ in t["test"]]
               for tid, t in tasks.items() if not is_holdout(tid)]
    n_jobs = 8
    with Pool(n_jobs) as pool:
        parts = pool.map(_dream_batch, [(seed + i, n // n_jobs, sources) for i in range(n_jobs)])
    X = [x for p in parts for x in p[0]]
    Y = [y for p in parts for y in p[1]]
    print(f"dream samples: {len(X)}")
    if wake:
        Xw, Yw = wake_data(wake, tasks)
        print(f"wake samples: {len(Xw)} (x{wake_weight})")
        X += Xw * wake_weight
        Y += Yw * wake_weight

    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60,
                                                          early_stopping=True, random_state=seed))
    model.fit(np.nan_to_num(np.array(X)), np.array(Y, dtype=int))
    print(f"validation score (exact label-set match): {model[-1].best_validation_score_:.3f}")
    return model


def primitive_costs(model, task):
    """Search cost per primitive for this task: -log P(primitive used)."""
    p = model.predict_proba(np.nan_to_num(task_features(task["train"]))[None])[0]
    return {name: max(MIN_COST, -np.log(max(pi, 1e-3))) for name, pi in zip(NAMES, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--wake", help="results JSON from a training-split search run")
    ap.add_argument("--out", default="models/prior.joblib")
    args = ap.parse_args()
    model = train(args.n, args.wake)
    Path(args.out).parent.mkdir(exist_ok=True)
    joblib.dump(model, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
