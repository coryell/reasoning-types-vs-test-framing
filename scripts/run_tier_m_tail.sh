#!/usr/bin/env bash
# Resume the mechanistic arm after the main run was stopped between arms (2026-09-03 23:30, see the
# LLM log): remaining arms in a new order — noise-floor baseline first, then the probe edit at a
# larger coefficient, then the remaining controls — followed by stages 7/8 of run_tier_m.sh and the
# probe-pooling ablations. Existing arm files are skipped by run_steering.py (an arm killed mid-generation
# writes nothing, so resumption is clean either way). Run inside tmux:
#   tmux new -s tierm2 'bash scripts/run_tier_m_tail.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_tier_m.log
ROOT=results/qwen3_4b
step() { echo "[$(date '+%F %T')] ==> $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}; if [ "$rc" -ne 0 ]; then step "FAILED (exit $rc): $*"; return 1; fi; return 0; }
AA=$(uv run python -c "import json; print(json.load(open('results/qwen3_4b/sweep/chosen.json'))['chosen']['aware']['alpha'])")
AU=$(uv run python -c "import json; print(json.load(open('results/qwen3_4b/sweep/chosen.json'))['chosen']['unaware']['alpha'])")
STRONG=${STRONG:-0.2}

step "6/8 main steering run, resumed: remaining arms reordered, strong alpha $STRONG"
run uv run python scripts/run_steering.py main --alpha-aware "$AA" --alpha-unaware "$AU" --strong-alpha "$STRONG" \
  --arms baseline_seed1 aware_strong unaware_strong random_aware shuffled_aware random_samerows_aware shuffled_samerows_aware || exit 1

step "7/8 readouts"
run uv run python scripts/readouts.py --stage annotate || true
run uv run python scripts/readouts.py --stage awareness || true
run uv run python scripts/readouts.py --stage project || true

step "8/8 analysis"
run uv run python scripts/analyze_tier_m.py || true

step "probe pooling ablations (whole / last)"
for P in whole last; do
  run uv run python scripts/build_probe.py --generations "$ROOT/triggers_probe.json" --judgments "$ROOT/triggers_probe.json.awareness.jsonl" --pooling "$P" --out-dir "$ROOT/probe_$P" || true
done
step "run_tier_m done"
