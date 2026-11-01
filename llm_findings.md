## 8. Findings

| | zero-shot | fine-tuned | + shared TTT | + per-task TTT | per-task, rescored | per-task v2 (rescored) | 1.7B fine-tuned (+RE-ARC) | 1.7B per-task (rescored) | 1.7B + sampling |
|---|---|---|---|---|---|---|---|---|---|
| Solved (of 120) | 0 | 0 | 0 | **0.5** | 0 | 0 | 0 | 0 | 0 |
| Valid grids | 57% | 77% | 84% | 80% | 80% | 84% | **89%** | **89%** | 88% |
| Right-size answers (of 167 test inputs) | 32% | 61% | 66% | 60% | 64% | 65% | 71% | **74%** | **74%** |
| Cell accuracy, same-size tests sized right (copying: 80.3%) | 80.8% | 82.2% | 80.9% | 83.9% | 84.6% | 84.3% | 80.6% | **85.0%** | 84.5% |
| Answers with >= 90% of cells right | – | 39 | 43 | 36 | 48 | 47 | 46 | **62** | 61 |
| Answers with >= 99% of cells right | 0 | 0 | 0 | 2 | 2 | 1 | 0 | **3** | 2 |
| Training time | – | 89 min | 44 min | 75 min | (same adapters) | 125 min | 244 min | 130 min | (same adapters) |
| Evaluation time (vLLM) | 10 min | 13 min | 12 min | 14 min | 27 min | 28 min | 15 min | 47 min | 103 min |

The shared TTT validation loss (0.144 → 0.111) comes from rows of the same tasks it trained on, so it
is not a held-out number.

**1. Fine-tuning taught the format and the output size, not the rule.** The share of well-formed
grids rose from 57% to 77% and right-size answers nearly doubled (32% → 61%). Most of the 64% drop
in validation loss is explained by this: getting dimensions, row lengths and the "copy most of the
grid" structure right is worth a lot of tokens.

**2. Until per-task training, the cell-level gain was almost entirely copying.** A trivial answer —
the test input, unchanged — already gets 80.3% of cells right on same-size tasks. The fine-tuned
model gets 82.2% on the same-size tests it sizes correctly (zero-shot: 80.8%).

**3. One shared test-time adapter helped the format again, but not the rules.** Training one adapter
on all 120 tasks' demonstration pairs (never their test outputs) raised valid grids to 84% and
right-size answers to 66%, and fixed the output size on some tasks (`8f215267`: wrong size → 98.7%
of cells right). Its same-size cell accuracy, 80.9%, is level with copying.

**4. Per-task test-time training is the first step that learns something beyond copying.** A
separate adapter per task, trained only on that task's demonstrations, reaches 83.9% same-size cell
accuracy (3.6 points above copying) and produces the first exactly correct answer (`38007db0`,
test 2) plus a second answer with 4 of 621 cells wrong (`8f215267`). Each adapter sees a single rule,
so its steps are not spread across 120 different ones.

**5. The cost is format robustness and agreement.** With only 2–4 demonstrations per task, the
per-task adapters overfit: valid grids drop from 84% to 80%, right-size answers from 66% to 60%, and
the views agree less. The one solve won with a single vote (1 of 8 views; the two attempts tied at one
vote each), so it is fragile: the right answer was found, but voting barely kept it.

**6. Re-ranking by likelihood picks closer answers, not exact ones.** Scoring each test's top 8
candidates by their summed log-probability across all 8 views (instead of vote count) raises
right-size answers from 60% to 64% and answers with >= 90% of cells right from 36 to 48, but it ranks
two near misses above the correct grid on `38007db0`, losing the one solve. The model's likelihood
is a good guide to "close", not yet to "exactly right".

**7. More robust per-task training restored the format but not the solve.** With 24 augmentations
and 8 generic rows mixed into each task, valid grids return to 84% and right-size answers to 65%, and
cell accuracy holds at 84.3%. But nothing is solved by either ranking: the earlier solve was at the
edge of what this model finds, and a different random draw of augmentations loses it.

**8. Scale and more data improve everything except the solve count.** Qwen3-1.7B, fine-tuned on the
training split plus 10,000 RE-ARC pseudo-tasks (generated examples of the 400 original ARC training
tasks; none are evaluation tasks, and no generated pair matches an evaluation pair), writes valid
grids 89% of the time and right-size answers 71% of the time. On its own its cells are still at the
copying level (80.6%), like the 0.6B model before test-time training. With per-task adapters it is the
best run on every near-miss measure: 85.0% cell accuracy on same-size tests, 62 answers with at least
90% of cells right, and three within 1–2 cells (`53fb4810`: 2 of 450 wrong; `38007db0`: 1 of 175).
Still nothing is exactly right. The model and the data changed together, so this gain cannot be
credited to either alone. Its validation loss (0.375 → 0.098) shares generators with the training
rows, so it is not a held-out number.

**9. The near misses are still copy tasks.** `e376de54`, `4c3d4a41` and `8e5c0c38` remain 3–17
cells away in every run; the remaining cells are exactly the ones the rule changes.

**10. Sampling finds a new answer but cannot rank it.** Adding 4 sampled answers per view
(temperature 0.7) to the 1.7B per-task run, 5,312 extra answers in all, and rescoring the top 12
candidates by likelihood puts the exactly correct grid among the candidates for 1 of 167 test
inputs (`2d0172a1`, a task no earlier run came close on). It was not ranked into the two attempts:
the top-ranked answer there is not even the right size. The other measures barely move (84.5% cell
accuracy, 61 answers with at least 90% of cells right), and `38007db0`'s 1-cell miss falls out of the
top two. Sampling adds some diversity, but likelihood ranking cannot tell the right answer from the near
misses. One found answer is too few to call ranking the main bottleneck, but it is the first place
where generation succeeded and selection failed.

### What this suggests next
Every step has moved the answers closer, but none has crossed into exact answers: ARC scores only
grids with every cell right, and the last 1–10 cells are exactly the ones the rule changes.
- **Better ranking, not more samples:** sampling found a correct grid that likelihood ranking then
  discarded. Checking candidates against the demonstrations (does the same transformation reproduce
  the demo outputs?) or scoring each candidate under every augmented view separately are the
  natural next filters.
- **Larger models and far more data:** published LLM approaches use 4–8B models trained on
  hundreds of thousands of RE-ARC-style examples, against 1.7B and 20k here.
- **Combine with the DSL:** the DSL is exact when one of its programs fits; the LLM's strength so far
  is producing a well-sized, near-correct canvas.
