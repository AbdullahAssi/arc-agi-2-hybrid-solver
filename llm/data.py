"""Build the fine-tuning dataset (JSONL of prompt/completion) from the training split only.

* Real tasks: every pair (train and test) takes a turn as the query, the remaining pairs are
  the demonstrations ("leave-one-out"), each under several random augmentations.
* Synthetic tasks: random DSL programs executed on training-split grids (the same generator
  that trains the search prior), giving the model many more distinct rules to learn from.
* RE-ARC (--re-arc DIR): pre-generated examples from hand-written generators of the 400 original
  ARC training tasks; each row is a fresh pseudo-task of 2-4 demonstrations plus a query.
The evaluation split is never touched, except by --ttt: test-time training data built from the
evaluation tasks' demonstration pairs only (their test outputs are never used).

Usage: python -m llm.data --augs 2 --synthetic 5000 --out data/llm_train.jsonl
       python -m llm.data --ttt --augs 16 --out data/llm_ttt.jsonl
       python -m llm.data --re-arc ~/ext/re_arc_data/re_arc/tasks --re-arc-per-task 40 --out data/llm_train_v2.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np

from solver.grid import load_split

from .format import make_prompt, make_target, n_tokens_estimate, random_aug


def leave_one_out(pairs, rng, n_augs, max_tokens, min_demos=2):
    """Examples where each pair in turn is the query and the others are demonstrations."""
    out = []
    for qi in range(len(pairs)):
        demos = pairs[:qi] + pairs[qi + 1:]
        query_in, query_out = pairs[qi]
        if len(demos) < min_demos or n_tokens_estimate(demos, query_in) + query_out.size * 1.1 > max_tokens:
            continue
        for _ in range(n_augs):
            aug = random_aug(rng, len(demos))
            out.append({"prompt": make_prompt(aug.pairs(demos), aug.grid(query_in)),
                        "completion": make_target(aug.grid(query_out))})
    return out


def synthetic_tasks(tasks, n, rng):
    """Random DSL programs run on a training task's inputs; 3-5 pairs each."""
    from solver.prior import dream_task  # imported here: the prior needs joblib, absent from .venv-llm

    sources = [[x for x, _ in t["train"]] + [x for x, _ in t["test"]] for t in tasks.values()]
    made = []
    while len(made) < n:
        inputs = sources[rng.integers(len(sources))]
        if len(inputs) < 3:
            continue
        sample = dream_task(rng, inputs)
        if sample is not None:
            made.append(sample[0][:5])
    return made


def re_arc_rows(folder, per_task, rng, max_tokens):
    """One row per pseudo-task sampled from each RE-ARC generator's examples.

    Guards against evaluation leakage twice: the task ids must not be evaluation ids, and any
    pseudo-task containing an exact evaluation (input, output) pair is dropped and counted.
    """
    evaluation = load_split("evaluation")
    eval_pairs = {(x.shape, x.tobytes(), y.shape, y.tobytes())
                  for t in evaluation.values() for x, y in t["train"] + t["test"] if y is not None}
    paths = sorted(Path(folder).expanduser().glob("*.json"))
    assert not {p.stem for p in paths} & set(evaluation), "RE-ARC task id in the evaluation split"
    rows, dropped = [], 0
    for path in paths:
        examples = [(np.array(e["input"], dtype=np.int8), np.array(e["output"], dtype=np.int8))
                    for e in json.loads(path.read_text())]
        made = 0
        for _ in range(per_task * 3):  # some draws are too long for max_tokens
            if made == per_task:
                break
            pick = rng.choice(len(examples), size=int(rng.integers(3, 6)), replace=False)
            pairs = [examples[i] for i in pick]
            if any((x.shape, x.tobytes(), y.shape, y.tobytes()) in eval_pairs for x, y in pairs):
                dropped += 1
                continue
            demos, (query_in, query_out) = pairs[:-1], pairs[-1]
            if n_tokens_estimate(demos, query_in) + query_out.size * 1.1 > max_tokens:
                continue
            aug = random_aug(rng, len(demos))
            rows.append({"prompt": make_prompt(aug.pairs(demos), aug.grid(query_in)),
                         "completion": make_target(aug.grid(query_out))})
            made += 1
    print(f"re-arc rows {len(rows)} from {len(paths)} generators; {dropped} draws dropped for evaluation overlap")
    return rows


def ttt_rows(rng, n_augs, max_tokens, mix, mix_rows):
    """Leave-one-out over each evaluation task's demonstrations, plus a share of generic rows."""
    rows, empty = [], []
    for tid, t in load_split("evaluation").items():
        made = leave_one_out(t["train"], rng, n_augs, max_tokens, min_demos=1)
        hidden = {make_target(y) for _, y in t["test"] if y is not None}
        assert not any(r["completion"] in hidden for r in made), tid  # cheap guard against leakage
        if not made:
            empty.append(tid)
        rows += made
    print(f"ttt rows {len(rows)}; {len(empty)} tasks too long for any example: {' '.join(empty)}")
    if mix:
        rows += [mix_rows[i] for i in rng.choice(len(mix_rows), size=int(len(rows) * mix), replace=False)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augs", type=int, default=2, help="augmented copies per real example")
    ap.add_argument("--synthetic", type=int, default=5000, help="number of synthetic tasks")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default="data/llm_train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--re-arc", help="folder of RE-ARC task JSONs (training-set generators)")
    ap.add_argument("--re-arc-per-task", type=int, default=40, help="pseudo-tasks per RE-ARC generator")
    ap.add_argument("--ttt", action="store_true", help="test-time data from evaluation demonstrations")
    ap.add_argument("--mix", type=float, default=0.15, help="--ttt: share of generic rows mixed in")
    ap.add_argument("--mix-from", default="data/llm_train.jsonl")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.ttt:
        generic = [json.loads(line) for line in open(args.mix_from)] if args.mix else []
        rows = ttt_rows(rng, args.augs, args.max_tokens, args.mix, generic)
        rng.shuffle(rows)
        write(rows, args.out)
        return
    tasks = load_split("training")
    real = []
    for t in tasks.values():
        pairs = t["train"] + [(x, y) for x, y in t["test"] if y is not None]
        real += leave_one_out(pairs, rng, args.augs, args.max_tokens)
    synth = []
    for pairs in synthetic_tasks(tasks, args.synthetic, rng):
        synth += leave_one_out(pairs, rng, 1, args.max_tokens)[:1]  # one example per synthetic task

    extra = re_arc_rows(args.re_arc, args.re_arc_per_task, rng, args.max_tokens) if args.re_arc else []
    rows = real + synth + extra
    rng.shuffle(rows)
    print(f"real {len(real)}  synthetic {len(synth)}  re-arc {len(extra)}")
    write(rows, args.out)


def write(rows, out):
    Path(out).parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tokens = sum(len(r["prompt"]) + len(r["completion"]) for r in rows)
    print(f"total {len(rows)}  ~{tokens / 1e6:.1f}M tokens -> {out}")


if __name__ == "__main__":
    main()
