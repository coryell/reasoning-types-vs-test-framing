#!/usr/bin/env bash
# F4 controls, chained behind the random-direction arms (2026-09-04 17:45): build the shuffled-label
# vectors, run the shuffled arms, then every readout and the analysis on the whole F4 directory.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ~/.config/d10/env; set +a
LOG=logs/run_followup.log
step() { echo "[$(date '+%F %T')] ==> $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep -v "Loading weights" | tee -a "$LOG"; }
until grep -q "F4C_DONE" "$LOG"; do sleep 60; done
R=results/qwen3_4b; D=$R/followups/f4_reverse
step "shuffled-label vectors"
run uv run python scripts/build_vectors.py --stages extract --shuffle-labels || exit 1
step "F4 shuffled-label arms"
run uv run python scripts/run_followup.py f4_reverse --coefficient 0.25 --arms shuffled_bt shuffled_unc || exit 1
step "F4 readouts"
for S in annotate awareness prompt_state project_incontext probe; do run uv run python scripts/readouts.py --stage "$S" --steering-dir "$D" || true; done
step "F4 analysis"
run uv run python scripts/analyze_tier_m.py --steering-dir "$D" --out-prefix f4_reverse --reference "$R/steering/baseline.json" "$R/steering/baseline_seed1.json" || true
echo "F4S_DONE" >> "$LOG"
