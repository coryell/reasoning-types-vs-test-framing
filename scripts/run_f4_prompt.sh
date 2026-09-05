#!/usr/bin/env bash
# f4_prompt, chained behind the F4 random arms (reordered 18:05 at the user's priority): the four "just ask" arms (positives first), then
# every readout and the analysis against the main-run baselines.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ~/.config/d10/env; set +a
LOG=logs/run_followup.log
step() { echo "[$(date '+%F %T')] ==> $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep -v "Loading weights" | tee -a "$LOG"; }
until grep -q "F4C_DONE" "$LOG"; do sleep 60; done
R=results/qwen3_4b; D=$R/followups/f4_prompt
step "f4_prompt arms"
run uv run python scripts/run_followup.py f4_prompt --arms ask_unc_pos ask_bt_pos ask_unc_neg ask_bt_neg || exit 1
step "f4_prompt readouts"
for S in annotate awareness prompt_state project_incontext probe; do run uv run python scripts/readouts.py --stage "$S" --steering-dir "$D" || true; done
step "f4_prompt analysis"
run uv run python scripts/analyze_tier_m.py --steering-dir "$D" --out-prefix f4_prompt --reference "$R/steering/baseline.json" "$R/steering/baseline_seed1.json" || true
echo "F4P_DONE" >> "$LOG"
