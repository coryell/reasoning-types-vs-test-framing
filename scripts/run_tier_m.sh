#!/usr/bin/env bash
# Mechanistic arm on Qwen3-4B (EXECUTION_PLAN_2 §2b), end to end, with the gate. Run inside tmux:
#   tmux new -s tierm 'bash scripts/run_tier_m.sh'
# Every stage is resumable; a failed stage is logged and the schedule stops at the next dependency.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/run_tier_m.log
ROOT=results/qwen3_4b
mkdir -p logs "$ROOT"
step() { echo "[$(date '+%F %T')] ==> $*" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}; if [ "$rc" -ne 0 ]; then step "FAILED (exit $rc): $*"; return 1; fi; return 0; }
verdict() { uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" "$1"; }

step "smoke: 3 pairs end to end (catches OOM before any spend)"
run uv run python scripts/gen_triggers.py --n-pairs 3 --seed 0 --out "$ROOT/smoke_triggers.json" || exit 1
run uv run python scripts/judge_awareness.py --input "$ROOT/smoke_triggers.json" --model-name qwen3_4b --family smoke --gate 0 || exit 1

step "0/8 gate: 200 trigger pairs"
run uv run python scripts/gen_triggers.py --n-pairs 200 --seed 0 --out "$ROOT/triggers_gate.json" || exit 1
run uv run python scripts/judge_awareness.py --input "$ROOT/triggers_gate.json" --model-name qwen3_4b --family triggers_gate || exit 1
V=$(verdict "$ROOT/triggers_gate.json.awareness.stats.json")
step "gate verdict: $V"
if [ "$V" != "PASS" ]; then step "gate failed; stopping the mechanistic arm"; exit 0; fi

step "1/8 probe set: 400 more pairs (offset 200)"
run uv run python scripts/gen_triggers.py --n-pairs 400 --offset 200 --seed 0 --out "$ROOT/triggers_probe_extra.json" || exit 1
run uv run python -c "
import json
a=json.load(open('results/qwen3_4b/triggers_gate.json')); b=json.load(open('results/qwen3_4b/triggers_probe_extra.json'))
json.dump(a+b, open('results/qwen3_4b/triggers_probe.json','w'), indent=1); print('merged', len(a)+len(b), 'pairs')
" || exit 1
run uv run python scripts/judge_awareness.py --input "$ROOT/triggers_probe.json" --model-name qwen3_4b --family triggers_probe || exit 1

step "2/8 probe"
run uv run python scripts/build_probe.py --generations "$ROOT/triggers_probe.json" --judgments "$ROOT/triggers_probe.json.awareness.jsonl" || exit 1

step "3/8 Venhoff vectors: generate, annotate, extract, layers"
run uv run python scripts/build_vectors.py || exit 1

step "4/8 geometry"
run uv run python scripts/geometry.py || true

step "5/8 alpha sweep"
run uv run python scripts/run_steering.py sweep --alphas 0.02 0.05 0.1 0.2 --n-items 40 || exit 1
for f in "$ROOT"/sweep/*.json; do
  case "$f" in *.gen.json|*.stats.json) continue;; esac
  run uv run python scripts/judge_awareness.py --input "$f" --model-name qwen3_4b --family sweep --arm "$(basename "$f" .json)" || true
done
run uv run python scripts/pick_alpha.py || exit 1
AA=$(uv run python -c "import json; print(json.load(open('results/qwen3_4b/sweep/chosen.json'))['chosen']['aware']['alpha'])")
AU=$(uv run python -c "import json; print(json.load(open('results/qwen3_4b/sweep/chosen.json'))['chosen']['unaware']['alpha'])")
FLAG=$(uv run python -c "import json; c=json.load(open('results/qwen3_4b/sweep/chosen.json'))['chosen']; print(int(c['aware']['flagged_no_qualifying_arm'] or c['unaware']['flagged_no_qualifying_arm']))")
step "chosen alpha: aware $AA, unaware $AU (flagged=$FLAG)"
if [ "$FLAG" != "0" ]; then step "no sweep arm met the closure/repetition guard; stopping before the main run"; exit 0; fi

step "6/8 main steering run"
run uv run python scripts/run_steering.py main --alpha-aware "$AA" --alpha-unaware "$AU" || exit 1

step "7/8 readouts"
run uv run python scripts/readouts.py --stage annotate || true
run uv run python scripts/readouts.py --stage awareness || true
run uv run python scripts/readouts.py --stage project || true

step "8/8 analysis"
run uv run python scripts/analyze_tier_m.py || true
step "run_tier_m done"
