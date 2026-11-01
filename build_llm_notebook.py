"""Builds llm_analysis.ipynb from results/. Usage: python build_llm_notebook.py llm_analysis.ipynb [findings.md]"""
import sys
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# ARC-AGI-2 — LLM pipeline analysis

A small language model (Qwen3-0.6B, later Qwen3-1.7B) reads each task as text and writes the test output grid.
Every test input is shown under 8 views (the original plus 7 random rotations/reflections,
colour permutations and example orders); answers are mapped back to the original orientation and
the two most-voted grids become attempts 1 and 2.

| Run | Model |
|---|---|
| `zero-shot` | Qwen3-0.6B as released |
| `fine-tuned` | + LoRA adapter trained on the training split (real tasks, leave-one-out, augmented) and synthetic DSL tasks |
| `test-time trained` | + one more epoch on the evaluation tasks' own demonstration pairs (leave-one-out, 16 augmentations each, 15% generic rows mixed in); their test outputs are never used |
| `per-task TTT` | fine-tuned adapter + a separate adapter per task, trained only on that task's demonstrations (leave-one-out, 16 augmentations, 2 epochs) |
| `per-task TTT, rescored` | the same answers, with the top 8 candidates per test re-ranked by summed log-probability across the 8 views (findings table only) |
| `1.7B fine-tuned` | Qwen3-1.7B + LoRA trained on the training split plus 10,000 RE-ARC pseudo-tasks (generated examples of the 400 original ARC training tasks) |
| `1.7B per-task TTT` | the 1.7B adapter + a separate adapter per task (as `per-task TTT`), rescored |
| `1.7B sampled` | as `1.7B per-task TTT`, plus 4 sampled answers per view (temperature 0.7); the top 12 candidates are rescored |

All scores are on the 120-task public **evaluation** split. No model saw its test outputs; only the
test-time-trained adapter saw its demonstration pairs, which every solver is given anyway.
""")

md("## 1. Setup")
code("""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from solver.grid import load_split
from taskstats import category

RES = Path("results")
# Evaluation runs to compare, in order (add test-time-training runs here later)
# (shared TTT, the rescored-only 0.6B run and per-task v2 are in the findings table; six runs keep the plots readable)
RUNS = {"zero-shot": "llm_evaluation_zeroshot_q06", "fine-tuned": "llm_evaluation_lora_q06",
        "per-task TTT": "llm_evaluation_ttt_task_q06",
        "1.7B fine-tuned": "llm_evaluation_lora_q17", "1.7B per-task TTT": "llm_evaluation_ttt_task_q17",
        "1.7B sampled": "llm_evaluation_ttt_task_sample_q17"}
TRAIN_LOGS = {"0.6B fine-tuning": RES / "llm_train_q06.log", "1.7B fine-tuning (+RE-ARC)": RES / "llm_train_q17.log"}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7d5bd6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
})
ARC_CMAP = ListedColormap(["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
                           "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"])

eval_tasks = load_split("evaluation")
cat_eval = pd.Series({t: category(v) for t, v in eval_tasks.items()})
runs = {name: json.loads((RES / f"{f}.json").read_text()) for name, f in RUNS.items() if (RES / f"{f}.json").exists()}
COLOR = dict(zip(runs, SERIES))


def tests_frame(run):
    \"\"\"One row per test input, with task-level descriptors attached.\"\"\"
    rows = []
    for t in run["tasks"]:
        for i, d in enumerate(t.get("tests", [])):
            y = eval_tasks[t["task"]]["test"][i][1]
            rows.append({"task": t["task"], "test": i, **{k: v for k, v in d.items() if k != "attempts"},
                         "top_votes": d["votes"][0] if d["votes"] else 0,
                         "out_cells": y.size, "category": cat_eval[t["task"]]})
    return pd.DataFrame(rows)


frames = {name: tests_frame(r) for name, r in runs.items()}
print("runs loaded:", {n: f"{len(f)} test inputs" for n, f in frames.items()})
""")

md("## 2. Training curves")
code("""
def read_log(path):
    train_pts, val_pts = [], []
    for line in path.read_text().splitlines() if path.exists() else []:
        if m := re.match(r"step (\\d+)/\\d+ loss ([\\d.]+)", line):
            train_pts.append((int(m[1]), float(m[2])))
        elif m := re.match(r"step (\\d+) val_loss ([\\d.]+)", line):
            val_pts.append((int(m[1]), float(m[2])))
    return train_pts, val_pts

logs = {name: read_log(path) for name, path in TRAIN_LOGS.items()}
fig, axes = plt.subplots(1, len(logs), figsize=(12, 4), squeeze=False)
for ax, (name, (train_pts, val_pts)) in zip(axes[0], logs.items()):
    if train_pts:
        s, l = zip(*train_pts)
        ax.plot(s, l, color=SERIES[0], alpha=0.85, label="training loss (mean of last 10 steps)")
    if val_pts:
        s, l = zip(*val_pts)
        ax.plot(s, l, color=SERIES[1], marker="o", markersize=8, label="validation loss")
        for x, y in val_pts:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", color=INK2, fontsize=9)
    ax.set(xlabel="optimizer step", ylabel="cross-entropy on answer tokens")
    ax.set_title(f"{name}: loss on the answer grid", loc="left")
    ax.legend(frameon=False)
    if val_pts:
        print(f"{name}: validation loss {val_pts[0][1]:.3f} -> {val_pts[-1][1]:.3f}")
plt.tight_layout(); plt.show()
print("Test-time validation rows come from the same evaluation tasks as its training rows, "
      "so that curve is not a held-out signal.")
""")

md("""
## 3. Before vs after

* **valid** — the model wrote a well-formed grid (rectangular, digits only)
* **right size** — the top-voted grid has the expected height and width
* **cell accuracy** — share of cells correct, on test inputs where the size is right
* **solved** — attempt 1 or 2 is exactly right (the ARC metric)
""")
code("""
def headline(name):
    f, r = frames[name], runs[name]
    return {"solved (of 120)": r["score"], "valid grids %": 100 * f["valid"].sum() / f["views"].sum(),
            "right size %": 100 * f["shape_ok"].mean(),
            "cell accuracy % (right size)": 100 * f.loc[f["shape_ok"], "cell_acc"].mean() if f["shape_ok"].any() else 0,
            "minutes": r["seconds"] / 60}

head = pd.DataFrame({n: headline(n) for n in frames}).T.round(1)
display(head)

metrics = ["valid grids %", "right size %", "cell accuracy % (right size)"]
fig, ax = plt.subplots(figsize=(10, 3.8))
x = np.arange(len(metrics))
w = 0.8 / len(head)
for k, name in enumerate(head.index):
    vals = head.loc[name, metrics].to_numpy(float)
    bars = ax.bar(x + (k - (len(head) - 1) / 2) * w, vals, w * 0.92, color=COLOR[name], label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", color=INK2, fontsize=9)
ax.set_xticks(x, metrics); ax.set_ylim(0, 125); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("%")
ax.set_title("What the model gets right, before and after fine-tuning", loc="left")
ax.legend(frameon=False, loc="upper left", ncol=len(head))
plt.tight_layout(); plt.show()
""")

md("## 4. Near misses: how close are the wrong answers?")
code("""
fig, ax = plt.subplots(figsize=(10, 3.8))
bins = np.linspace(0, 1, 21)
for name, f in frames.items():
    acc = f.loc[f["shape_ok"], "cell_acc"]
    ax.hist(acc, bins=bins, histtype="step", linewidth=2, color=COLOR[name], label=f"{name} (n={len(acc)})")
ax.set(xlabel="share of cells correct (test inputs where the grid size is right)", ylabel="test inputs")
ax.set_title("Cell accuracy of the top-voted answer", loc="left")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout(); plt.show()
last = list(frames)[-1]
f = frames[last]
print(f"{last}: {(f['cell_acc'] >= 0.9).sum()} test inputs with >=90% of cells right, "
      f"{(f['cell_acc'] >= 0.99).sum()} with >=99%, {int(f['correct'].sum())} exactly right")
""")

md("## 5. Where does it do better or worse?")
code("""
f = frames[last].copy()
by_cat = f.groupby("category").agg(tests=("task", "size"), right_size=("shape_ok", "mean"),
                                   cell_acc=("cell_acc", "mean"), solved=("correct", "sum"))
by_cat = by_cat.sort_values("right_size")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
axes[0].barh(by_cat.index, 100 * by_cat["right_size"], color=SERIES[0], height=0.6)
axes[0].set(xlabel="% right size", xlim=(0, 115), xticks=range(0, 101, 20))
axes[0].set_title(f"{last}: right-size rate by task type", loc="left")
axes[1].barh(by_cat.index, 100 * by_cat["cell_acc"].fillna(0), color=SERIES[0], height=0.6)
axes[1].set(xlabel="% cells correct (right size only)", xlim=(0, 115), xticks=range(0, 101, 20))
axes[1].set_title("cell accuracy by task type", loc="left")
for ax, col in zip(axes, ["right_size", "cell_acc"]):
    for i, (v, n) in enumerate(zip(by_cat[col].fillna(0), by_cat["tests"])):
        ax.text(100 * v + 1, i, f"n={n}", va="center", color=INK2, fontsize=8)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(10, 3.8))
ok = f[f["shape_ok"]]
ax.scatter(ok["out_cells"], ok["cell_acc"], s=50, color=SERIES[0], edgecolor="#fcfcfb", linewidth=1.5, zorder=2)
hit = ok[ok["correct"]]
ax.scatter(hit["out_cells"], hit["cell_acc"], s=90, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=3, label="solved")
ax.set(xscale="log", xlabel="cells in the expected output (log scale)", ylabel="share of cells correct")
ax.set_title(f"{last}: accuracy falls with output size", loc="left")
if len(hit):
    ax.legend(frameon=False)
plt.tight_layout(); plt.show()
""")

md("## 6. Does agreement between views predict correctness?")
code("""
fig, ax = plt.subplots(figsize=(10, 3.6))
for name, f in frames.items():
    g = f[f["shape_ok"]].groupby("top_votes")["cell_acc"].mean()
    ax.plot(g.index, g.values, marker="o", markersize=8, color=COLOR[name], label=name)
ax.set(xlabel="votes for the top answer (out of 8 views)", ylabel="mean cell accuracy", ylim=(0, 1.05))
ax.set_title("When more views agree, is the answer better?", loc="left")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
pd.crosstab(frames[last]["top_votes"], frames[last]["correct"], margins=True)
""")

md("## 7. Examples")
code('''
def show(task_id, test_idx, run_name, title):
    task = eval_tasks[task_id]
    tests = next(t for t in runs[run_name]["tasks"] if t["task"] == task_id)["tests"]
    pred = np.array(tests[test_idx]["attempts"][0]) if tests[test_idx]["attempts"] else None
    panels = [(task["train"][0][0], "train 1 in"), (task["train"][0][1], "train 1 out"),
              (task["test"][test_idx][0], "test in"), (task["test"][test_idx][1], "expected")]
    if pred is not None:
        panels.append((pred, "model (attempt 1)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.9))
    for ax, (g, label) in zip(axes, panels):
        ax.imshow(g, cmap=ARC_CMAP, vmin=0, vmax=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(label, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left")
    plt.tight_layout(); plt.show()


f = frames[last]
for _, r in f[f["correct"]].head(3).iterrows():
    show(r["task"], r["test"], last, f"Solved {r['task']} ({r['category']}), {r['top_votes']}/8 views agree")
near = f[f["shape_ok"] & ~f["correct"]].sort_values("cell_acc", ascending=False).head(3)
for _, r in near.iterrows():
    show(r["task"], r["test"], last, f"Near miss {r['task']}: {r['cell_acc']:.0%} of cells right ({r['category']})")
''')

findings = Path(sys.argv[2]).read_text() if len(sys.argv) > 2 else "## 8. Findings\n\n_Written once the results are in._"
md(findings)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1])
