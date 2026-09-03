#!/usr/bin/env bash
# Full Tier 1 annotation run plus the M0 gate, in the plan's priority order (EXECUTION_PLAN_2 §1, §M0).
# Run inside tmux:  tmux new -s d10 'bash scripts/run_tier1.sh'
# Reruns are safe: cached results are skipped, failed calls are retried.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_tier1.log
mkdir -p logs
echo "[$(date '+%F %T')] run_tier1 start" | tee -a "$LOG"

step() { echo "[$(date '+%F %T')] ==> $*" | tee -a "$LOG"; }

step "1/5 actions, all models"
uv run python scripts/annotate_shipped.py --families actions --concurrency 12 2>&1 | tee -a "$LOG"

step "2/5 M0 gate: judge the 1.5B pilot traces"
uv run python scripts/judge_pilot.py --concurrency 12 2>&1 | tee -a "$LOG"

step "3/5 actions_sampling (Distill-Qwen noise floor)"
uv run python scripts/annotate_shipped.py --families actions_sampling --concurrency 12 2>&1 | tee -a "$LOG"

step "4/5 triggers, QwQ first"
uv run python scripts/annotate_shipped.py --models qwq --families triggers --concurrency 12 2>&1 | tee -a "$LOG"
uv run python scripts/annotate_shipped.py --models qwen3 deepseek_qwen --families triggers --concurrency 12 2>&1 | tee -a "$LOG"

step "5/5 harmbench"
uv run python scripts/annotate_shipped.py --families harmbench --concurrency 12 2>&1 | tee -a "$LOG"

step "retry pass (errors only; everything else is cached)"
uv run python scripts/annotate_shipped.py --concurrency 8 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] run_tier1 done" | tee -a "$LOG"
