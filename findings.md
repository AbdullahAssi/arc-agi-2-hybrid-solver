## 8. Findings

**1. The learned prior makes search ~6x more efficient, with zero task-specific supervision.**
On the 184 held-out training tasks, the median solution is found after **3 nodes instead of 34**
(geometric-mean speed-up 5.9x for the dream prior, 6.6x with wake data). The prior was trained only on
random DSL programs executed on *other* tasks' input grids — it never saw a held-out task or a
human-written solution. Inside the prior, the primitive that actually solves a task has median rank
**2 of 55** on held-out tasks, and is in the top 5 for 81% of them (random ranking: median ~28).

**2. ...but it solves exactly the same tasks. The bottleneck is the DSL, not the search.**
Every arm solves the same 15/184 held-out tasks. With 55 primitives, every program the DSL can express
that fits a held-out task is found within ~550 nodes even by blind breadth-first search, far below the
5,000-node budget. A better search order cannot find programs that do not exist.

**3. Going deeper does not help either.** Allowing 4-step programs found no new solutions. What the
prior *does* buy at depth 4 is wall-clock time: median 32 s vs 59 s per task, because guided search
reaches its solutions before the frontier explodes. This is where a prior would matter if the DSL
had solutions at depth 4+.

**4. Growing the DSL is what moved the score.** Adding 9 primitives (drawing, cell summaries, odd-one-out)
and object-level recolouring took the training split from **74.5 to 93 / 1000** (+19 tasks, 1 lost to the
larger branching factor). Gains are uneven: "smaller output" (16.5%) and "larger output" (18.5%) tasks
are solved 3x as often as same-size edit tasks (~5–8%), which need *local, object-conditional* rules
that whole-grid primitives cannot express. 15 of 55 primitives are never used in any solution.

**5. Coincidental fits are a real failure mode.** 14 training tasks have a program that fits every train
pair yet fails the test (5 of them in the held-out set). Most are "smaller output" tasks where the output
is tiny (often 1x1 or a small crop), so many unrelated programs can match a handful of examples by
chance. Shortest-program-first (Occam) ordering does not prevent this.

**6. ARC-AGI-2 evaluation: 0 / 120 for every configuration.** Evaluation grids are 3x larger (median 299
vs 100 cells), use more colours (5 vs 3), and are dominated by mixed same-size edits (41%) — the category
this DSL handles worst. This matches ARC-AGI-2's design goal of resisting brute-force program search.

### What this suggests next
- **Expressiveness over search:** primitives that are *parameterised by the examples* (like the
  object-recolour fit) generalise better than more fixed transformations. Object-wise "apply f to each
  object" combinators are the obvious next step.
- **Verification beyond the train pairs:** penalise programs whose output is trivially small or that
  ignore most of the input, to reduce coincidental fits.
- **A learned model for what the DSL cannot express:** LLM test-time training on the GPU, combined with
  the DSL (DSL answers when a program fits; LLM otherwise).
