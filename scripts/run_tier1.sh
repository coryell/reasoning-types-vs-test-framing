#!/usr/bin/env bash
# Full Tier 1 annotation run plus the M0 gate, in the plan's priority order (EXECUTION_PLAN_2 §1, §M0).
# Run inside tmux:  tmux new -s d10 'bash scripts/run_tier1.sh'
# Reruns are safe: cached results are skipped, failed calls are retried. A failing step is logged and
# the schedule continues; the final retry pass picks up whatever it left.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_tier1.log
mkdir -p logs
echo "[$(date '+%F %T')] run_tier1 start" | tee -a "$LOG"

step() { echo "[$(date '+%F %T')] ==> $*" | tee -a "$LOG"; }
run() {
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then step "FAILED (exit $rc): $*"; fi
  return 0
}

step "1/5 actions, all models"
run uv run python scripts/annotate_shipped.py --families actions --concurrency 12

step "2/5 M0 gate: judge the 1.5B pilot traces"
run uv run python scripts/judge_pilot.py --concurrency 12

step "3/5 actions_sampling (Distill-Qwen noise floor)"
run uv run python scripts/annotate_shipped.py --families actions_sampling --concurrency 12

step "4/5 triggers, QwQ first"
run uv run python scripts/annotate_shipped.py --models qwq --families triggers --concurrency 12
run uv run python scripts/annotate_shipped.py --models qwen3 deepseek_qwen --families triggers --concurrency 12

step "5/5 harmbench"
run uv run python scripts/annotate_shipped.py --families harmbench --concurrency 12

step "retry pass (errors only; successes and truncations are cached)"
run uv run python scripts/annotate_shipped.py --concurrency 8

echo "[$(date '+%F %T')] run_tier1 done" | tee -a "$LOG"
