#!/usr/bin/env bash
# After the seed-1 Qwen3-32B generation ends (GEN_DONE in its log), annotate and awareness-judge both
# arms, then run the seed comparison. API only; no GPU.
set -a; . ~/.config/d10/env; set +a
cd "$(dirname "$0")/.."
PY=~/d10/.venv/bin/python
LOG=logs/api_seed1_readouts.log
D=results/api_prompt/qwen3_32b_api_seed1
until grep -q GEN_DONE logs/run_api_prompt_seed1.log 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] generation done; readouts" >> $LOG
$PY scripts/readouts.py --stage annotate  --steering-dir $D --model-name qwen3_32b_api_seed1 --log $LOG || true
$PY scripts/readouts.py --stage awareness --steering-dir $D --model-name qwen3_32b_api_seed1 --log $LOG || true
$PY scripts/analyze_api_seeds.py >> $LOG 2>&1 || true
echo "[$(date '+%F %T')] ALL_DONE" >> $LOG
