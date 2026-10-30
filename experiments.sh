#!/usr/bin/env bash
# Full experiment pipeline: baseline coverage, prior training, ablation on held-out tasks.
# Node budgets (not wall-clock) are the controlled variable, so time limits are generous.
# Resumable: a step is skipped when its output file already exists (delete it to redo).
# Usage: WORKERS=100 bash experiments.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=../.venv/Scripts/python; fi
WORKERS=${WORKERS:-8}
NODES=5000
TL=300

# run_eval <split> <tag> [extra args...]  -> results/<split>_<tag>.json
run_eval() {
    local split=$1 tag=$2; shift 2
    if [ -f "results/${split}_${tag}.json" ]; then
        echo "skip ${split}_${tag} (exists)"
        return
    fi
    $PY run_eval.py --split "$split" --max-nodes $NODES --time-limit $TL --workers "$WORKERS" --tag "$tag" "$@"
}

# train_prior <out> [extra args...]
train_prior() {
    local out=$1; shift
    if [ -f "$out" ]; then
        echo "skip $out (exists)"
        return
    fi
    $PY -m solver.prior --n 30000 --out "$out" "$@"
}

# 1. Uniform search over the whole training split (coverage + wake data) and the public eval split
run_eval training   v2_uniform
run_eval evaluation v2_uniform

# 2. Priors: synthetic programs only, and synthetic + programs found on non-holdout training tasks
train_prior models/prior_dream_v2.joblib
train_prior models/prior_wake_v2.joblib --wake results/training_v2_uniform.json

# 3. Ablation on the held-out fifth of the training split (never seen by either prior)
run_eval training hold_d3_uniform --holdout
run_eval training hold_d3_dream   --holdout --prior models/prior_dream_v2.joblib
run_eval training hold_d3_wake    --holdout --prior models/prior_wake_v2.joblib
run_eval training hold_d4_uniform --holdout --max-depth 4
run_eval training hold_d4_dream   --holdout --max-depth 4 --prior models/prior_dream_v2.joblib
run_eval evaluation v2_d4_dream   --max-depth 4 --prior models/prior_dream_v2.joblib

$PY analyze.py results/training_hold_d3_uniform.json results/training_hold_d3_dream.json \
    results/training_hold_d3_wake.json results/training_hold_d4_uniform.json results/training_hold_d4_dream.json
