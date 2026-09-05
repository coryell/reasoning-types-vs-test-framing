# d10

Which parts of a reasoning model's chain-of-thought does steering its **test-awareness** direction
actually move?

Abdelnabi & Salem ([2505.14617](https://arxiv.org/abs/2505.14617)) showed that steering a linear
"test awareness" direction changes how often a reasoning model complies with harmful requests, and
asked in their §4.7 whether awareness also changes reasoning behaviours such as backtracking.
Venhoff et al. ([2506.18167](https://arxiv.org/abs/2506.18167)) released steerable directions for six
such behaviours. This project puts the two together and measures which behaviours move, which stay
flat, and whether any of them mediate the compliance effect.

Work in progress. See `CLAUDE.md` for layout, environment, and the working rules.

```bash
uv sync
git clone https://github.com/microsoft/Test_Awareness_Steering assets/test_awareness_steering
git clone https://github.com/cvenhoff/steering-thinking-llms assets/steering_thinking_llms
```

## What is here, in the order it was run

Every stage writes its outputs under `results/` (small JSON/CSV/markdown committed; tensors
regenerated) and a progress log under `logs/`. `SUMMARY*.md` files are script-generated tables,
not write-ups.

| Stage | Script(s) | Output |
|---|---|---|
| Audit of the shipped probe-training evidence quotes | `scripts/audit_evidence.py` | `results/evidence_audit/` |
| Tier 1: Venhoff-style annotation of the shipped steered 32B outputs, paired analysis | `scripts/run_tier1.sh` → `annotate_shipped.py`, `analyze_tier1.py`; `reannotate_sample.py` (judge stability) | `results/annotations/`, `results/tier1/` |
| Mechanistic arm on Qwen3-4B: gate, probe set, probe, rebuilt Venhoff directions, geometry, α sweep, main steering run, readouts, analysis | `scripts/run_tier_m.sh` (+ `run_tier_m_tail.sh` for the resumed second half) → `gen_triggers.py`, `judge_awareness.py`, `build_probe.py`, `build_vectors.py`, `geometry.py`, `run_steering.py`, `pick_alpha.py`, `readouts.py`, `analyze_tier_m.py` | `results/qwen3_4b/` (`SUMMARY_M.md`) |
| F1 prompt-only baseline, F2 uncertainty-direction steering, F4 reverse arrow (+ coefficient calibration) | `scripts/run_followup.py`, then `readouts.py --steering-dir …` and `analyze_tier_m.py --steering-dir … --out-prefix …` | `results/qwen3_4b/followups/`, `results/qwen3_4b/SUMMARY_f*.md` |
| Representation-level readouts (in-context projections, last-prompt-token states, probe scores) | `scripts/run_readouts_batch.sh` → `readouts.py --stage project_incontext / prompt_state / probe` | alongside each arm file |
| Their §4.5 prompt comparison on Qwen3-32B through OpenRouter | `scripts/run_api_prompt.py` | `results/api_prompt/` |
| F5: geometry of the shipped 32B probes | `scripts/probe_geometry_shipped.py` | `results/probe_geometry_shipped/` |

Shared code lives in `d10/`: judge client and caching (`judge.py`), the two papers' prompts
(`prompts.py`), span parsing and metrics incl. the self-referential test lexicon (`parse.py`),
loaders for the shipped and generated files (`shipped.py`), statistics and tables (`analysis.py`),
generation (`generate.py`), the weight edit (`steer.py`), residual-stream tools (`activations.py`),
the probe recipe (`probe.py`), the Venhoff recipe (`venhoff.py`), awareness-judge parsing
(`awareness.py`), and the serialisable steering `Condition` (`conditions.py`). Tests: `uv run pytest`.
API keys are read from `~/.config/d10/env`.
