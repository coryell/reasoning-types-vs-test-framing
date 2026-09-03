# CLAUDE.md — d10

Execution repo for a short research project: **which parts of a reasoning model's chain-of-thought
does steering its test-awareness direction actually move?** It builds on two released artifact sets:

- **Abdelnabi & Salem 2025**, *Linear Control of Test Awareness Reveals Differential Compliance in
  Reasoning Models* ([arXiv 2505.14617](https://arxiv.org/abs/2505.14617)). Per-layer awareness
  probes for three 32B reasoning models, a weight-edit steering method, labelled trigger datasets with
  judge evidence quotes, and saved steered outputs.
- **Venhoff et al. 2025**, *Understanding Reasoning in Thinking Language Models via Steering Vectors*
  ([arXiv 2506.18167](https://arxiv.org/abs/2506.18167)). Six reasoning-behaviour directions
  (backtracking, deduction, adding-knowledge, example-testing, initializing, uncertainty-estimation)
  as layer-indexed mean vectors for three R1-Distill models, plus 500 annotated traces per model.

## Layout

| Path | What |
|---|---|
| `assets/test_awareness_steering/` | Clone of `microsoft/Test_Awareness_Steering` at `eb78ea5`. Gitignored; re-clone to reproduce |
| `assets/steering_thinking_llms/` | Clone of `cvenhoff/steering-thinking-llms` at `93259bc`. Gitignored; re-clone to reproduce |
| `logs/` | One progress log per long-running job, a line per stage |
| `results/` | Outputs. Small JSON/CSV/PNG are committed; `.npy`/`.pt`/`.pth` are gitignored and regenerated from the scripts |

## Environment

`uv sync` builds `.venv` from `pyproject.toml`; torch comes from the CUDA 13 index. Run scripts as
`uv run python …`. The interpreter is a **uv-managed CPython 3.12** (`.python-version`), not the system
python: torch 2.14 routes some ops through Triton, which compiles a helper against `Python.h` at
runtime, and the system interpreter has no headers.

API keys are loaded from `~/.config/d10/env` (a dotenv file **outside** the repo), never from a file
in this directory. Required: `HF_TOKEN` (gated model access), `OPENROUTER_API_KEY` (judge calls).

This machine is WSL2 with an RTX 4070 Ti (12 GB). Never install NVIDIA packages inside WSL; the
Windows driver provides CUDA. Load anything over ~1B parameters in bf16.

## Rules

**1. Nothing private lives here.** No keys, no personal notes, no symlinks into other directories.
Treat every file under this directory as public whether or not it is gitignored.

**2. Verification standards.**
- Verbatim quotes with a location, or the claim does not count. Never present a paraphrase as a
  quotation. If a source could not be accessed, say so.
- Enumerate artifact inventories before forming a view: file trees and field names, not landing pages.
- Read the source code, not the paper's description of it. Every discrepancy found so far came from
  reading a function rather than a summary.
- Recompute load-bearing numbers rather than inheriting them, and check the population each
  statistic was computed over.
- Use Python string counts rather than `grep` to establish absence; `grep` has produced false
  negatives in this environment.

**3. Write here, not to the session scratchpad.** The scratchpad has deleted files mid-task.

**4. Do not use the local GPU without asking.** It is the user's workstation card. One job at a time;
concurrent jobs OOM each other at 12 GB.

**5. Emit progress for any long-running job.** A line per stage to a file under `logs/`. Long jobs run
in tmux so a dropped SSH connection does not kill them.

**6. Labels are not shared vocabulary.** If you introduce shorthand (phase names, Q1/Q2, section
numbers), say it is yours and define it. Do not name directories with abbreviations.

**7. Report faithfully.** Negative results are reported as negative. Every number in a write-up must
reproduce from an artifact on disk.
