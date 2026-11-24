## 10. Findings

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

**11. The ranking failure is not a length bias; the model prefers the wrong answers.** Finding 10
suggested a simple cause: a summed log-probability adds one negative term per cell, so smaller grids
score higher, and on `2d0172a1` the top answers (9x11, 9x9) were smaller than the correct 9x13. To
test it, the same run was repeated with every candidate rescored (up to 40 per test instead of the
top 12; 39,120 candidate views, 3.1 h) and the whole pool saved with per-view log-probabilities
(`infer --pool`, `results/llm_pool_evaluation_ttt_task_pool_q17.json`). Rankers were then compared
offline (`python -m llm.rank`). The pool held the correct grid for 2 of 167 test inputs (`2d0172a1`
again, and `8f215267`), so a perfect ranker would score 1.5/120. Every ranker tried scores 0:

| Ranker | Score | Rank of correct (`2d0172a1`, `8f215267`) |
|---|---|---|
| votes | 0 | 5, 15 |
| summed log-prob (the run's own) | 0 | 4, 14 |
| log-prob per token | 0 | 4, 14 |
| worst view, per token | 0 | 6, 22 |
| any of the above, implied-shape candidates first | 0 | same |
| oracle (correct anywhere in the pool) | 1.5 | – |

Normalising by size changes nothing: on `2d0172a1` the correct grid is also the least likely per
token of the top four (-0.052 against -0.021 for the 9x11 answer), and its output size is not
fixed by the demonstrations (11x12, 5x5, 7x5, 9x11), so no shape filter applies. On `8f215267` all
32 candidates are the right size and the top 13 are 2–8 cells from the answer; the correct grid
trails the best by 14 nats summed over 8 views, under 2 per view. So the model is not mis-scored
by a fixable artefact: its per-task adapter genuinely finds the near misses more likely, and among
candidates a few cells apart its likelihood carries almost no signal. Better ranking alone would
reach at most 1.5/120 here; generation, which got the right answer into the pool for 2 of 167
inputs, is the larger bottleneck.

**12. A 20B model writing programs solves half the training tasks but no evaluation tasks.** A
different generator: `gpt-oss:20b`, run locally with Ollama, writes a Python `transform(grid)`; the
program is kept only if it reproduces every demonstration exactly, and a failed one goes back with
its first mismatch for revision (2 conversations x 3 rounds per task; `llm/program.py`). Running
the code on the demonstrations is the check from outside the model that finding 11 found missing.
Pilots on 20 random tasks per split (section 9):

| Pilot | Solved | DSL, same tasks | Passed all demos | Calls without code | Time per call |
|---|---|---|---|---|---|
| training, think medium | **10/20** | 2/20 | 11 (1 wrong on test) | 56% | 140 s |
| evaluation, think low | 0/20 | 0 | 0 | 38% | 6 s |
| evaluation, think medium + forcing | 0/20 | 0 | 1 (wrong on test) | 2.5% | 117 s |

On training tasks it is the strongest solver in the project: 5x the DSL on the same tasks, solving
both of the DSL's tasks, four of them only after feedback rounds, and it reached that score while
over half its calls produced nothing (medium reasoning ran past the 16k-token cap). On evaluation
it solves nothing. With low reasoning the model gives up within ~500 tokens (on `dfadab01`: "Hard.
Given limited time, maybe identity?") and returns the input unchanged. With medium reasoning plus
budget forcing (a call that hits the cap is handed back the tail of its reasoning and must write
code), almost every call produces a program, but no program reproduces even one demonstration of
19 of the 20 tasks. The best program for 12 of 20 tasks beats copying the input, mostly by a few
points: the programs get the easy cells and miss the rule. The one program that passed (`45a5af55`)
fitted only 2 demonstrations and returned a 23x23 grid for a 26x26 answer, showing how few
demonstrations limit this check.

Two infrastructure bugs cost runs and are worth recording: under the runner's memory cap, `import
numpy` hung on the 112-core server (OpenBLAS reserves memory per core) until the timeout, silently
failing every program that used numpy; and the Ollama container was stopped mid-run, which now
triggers retries instead of empty results.

### What this suggests next
Every step has moved the answers closer, but none has crossed into exact answers: ARC scores only
grids with every cell right, and the last 1–10 cells are exactly the ones the rule changes.
- **Program synthesis is the approach to scale:** checking candidates against the demonstrations
  works (11 passes on training, 1 wrong), but the generator has to find the rule first. That
  needs a stronger reasoning model than a local 20B one, or many more samples per task with
  parallel requests; the training pilot should also be rerun with budget forcing for a fair number.
- **Ranking needs a signal other than the model's own likelihood:** size normalisation and shape
  filters were tried and do not help (finding 11). Even a perfect ranker reaches only 1.5/120
  here, so this is worth doing only alongside better generation.
- **Larger models and far more data:** published LLM approaches use 4–8B models trained on
  hundreds of thousands of RE-ARC-style examples, against 1.7B and 20k here.
- **Combine with the DSL:** the DSL is exact when one of its programs fits; the LLM's strength so far
  is producing a well-sized, near-correct canvas.
