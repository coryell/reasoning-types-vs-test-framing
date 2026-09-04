#!/usr/bin/env bash
# GPU readout batch (2026-09-04 13:00): representation-level readouts added after the main run —
# in-context projections, last-prompt-token states and the probe score — on F2, F1 and the main run,
# then the analyses. One model load per stage per directory; sequential. Run inside tmux.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ~/.config/d10/env; set +a
LOG=logs/readouts_batch.log
step() { echo "[$(date '+%F %T')] ==> $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep -v "Loading weights" | tee -a "$LOG"; }
R=results/qwen3_4b
for D in "$R/followups/f2_uncertainty" "$R/followups/f1_prompt" "$R/steering"; do
  for S in project_incontext prompt_state probe; do
    step "$S on $D"; run uv run python scripts/readouts.py --stage "$S" --steering-dir "$D" || true
  done
done
step "analyses"
run uv run python scripts/analyze_tier_m.py --steering-dir "$R/followups/f2_uncertainty" --out-prefix f2_uncertainty || true
run uv run python scripts/analyze_tier_m.py --steering-dir "$R/followups/f1_prompt" --out-prefix f1_prompt --reference "$R/steering/baseline.json" "$R/steering/baseline_seed1.json" "$R/steering/aware_strong.json" || true
run uv run python scripts/analyze_tier_m.py || true
step "readouts batch done"
