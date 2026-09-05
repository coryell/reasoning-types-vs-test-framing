#!/usr/bin/env python
"""Readouts for the steering arms (EXECUTION_PLAN_2 §M3): morphology annotation (Venhoff's prompt),
awareness verbalization (Abdelnabi's judge), and representation-level morphology — the mean
projection of each generated reasoning trace onto the rebuilt backtracking and uncertainty-estimation
directions at their chosen layers, computed on the steered model (the arm's weight edit re-applied).

    uv run python scripts/readouts.py --stage annotate      # Venhoff annotations per arm
    uv run python scripts/readouts.py --stage awareness     # awareness judgments per arm
    uv run python scripts/readouts.py --stage project       # projections (GPU)
    uv run python scripts/readouts.py --stage probe         # projection onto the awareness probe direction (GPU)
    uv run python scripts/readouts.py --stage prompt_state  # last-prompt-token state, before any generation (GPU)
    uv run python scripts/readouts.py --stage project_incontext  # projections over the reasoning tokens with the prompt in context (GPU)

The ``probe`` stage scores each trace's reasoning on the *unedited* model in the probe's own input
regime (``build_probe.py --pooling whole``: chat template + lowercased task + lowercased reasoning,
mean over the reasoning tokens at the probe's hidden-state index), projected onto the probe's
positive-class row (unit norm). It asks whether a trace reads as "aware" to the probe regardless of how it was
produced — e.g. whether told-awareness (the F1 prompt) and self-inferred awareness (baseline traces
the judge scores positive) share the direction.

The ``prompt_state`` stage reads the residual stream at the *last prompt token* (the position that
generates the first reasoning token), under the arm's own condition (edit applied, or the arm's
system prompt), and projects it onto the rebuilt behaviour directions at their layers and onto the
probe direction at its index. With Qwen3's thinking template the last prompt token is the newline
after ``assistant``, the position that generates ``<think>``. It measures the state an intervention
induces before the model has written anything, independent of the sampled trace.

The ``project_incontext`` stage is the ``project`` readout done properly for prompt conditions: the
forward pass runs on the arm's chat-templated prompt followed by the trace's own reasoning, under the
arm's condition, and the mean projection is taken over the reasoning tokens only. Where ``project``
reads the written text in isolation (so a system-prompt cue can act only through what it changed in
the writing), this measures the representation the model had while reasoning under the cue.
Directions: the rebuilt behaviour directions at their layers and the probe direction at its index.

Residual-add arms, indexing: transformers 5 collects ``output_hidden_states`` through its own
forward hooks, so whether ``hidden_states[layer + 1]`` of a residual-add arm includes the addition
depends on hook order; ``ResidualAdd`` now prepends its hook, so the recorded state at ``layer + 1``
is the post-addition residual stream (the projection onto the steered direction there therefore
shifts by ``coefficient × ‖vector‖`` mechanically; downstream indices show the model's response).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from d10.activations import abdelnabi_probe_text, char_spans_to_token_spans, mean_projection, pooled_states
from d10.conditions import applied, condition_from_sidecar
from d10.generate import format_prompt, load_model
from d10.judge import Job, Judge, run_jobs
from d10.prompts import abdelnabi_judge_prompt, load_abdelnabi_template, venhoff_annotation_prompt
from d10.shipped import abdelnabi_split, load_generations
from d10.venhoff import feature_vectors

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_steering import AGENT_PROMPT, action_prompt  # noqa: E402


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


@torch.inference_mode()
def last_token_projections(model, tok, prompt: str, dirs: list[tuple[str, int, torch.Tensor]]) -> dict[str, float]:
    """``h · v̂`` at the final position of the chat-templated ``prompt`` (special tokens kept: the
    template already contains them), one forward pass, for every ``(name, hs_index, direction)``."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    hs = model(**enc, output_hidden_states=True).hidden_states
    out = {}
    for name, hs_index, d in dirs:
        h = hs[hs_index][0, -1].float()
        v = d.to(h.device).float()
        out[name] = float(h @ (v / v.norm()))
    return out


@torch.inference_mode()
def in_context_projections(model, tok, prompt: str, reasoning: str, dirs: list[tuple[str, int, torch.Tensor]], max_len: int = 4096) -> dict[str, float]:
    """Mean ``h · v̂`` over the reasoning tokens of ``prompt + reasoning`` (the generation-time context),
    for every ``(name, hs_index, direction)``; NaN if the reasoning falls beyond ``max_len``."""
    text = prompt + reasoning
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=max_len)
    offsets = enc.pop("offset_mapping")[0].tolist()
    start = next((i for i, (a, b) in enumerate(offsets) if b > len(prompt)), None)  # first token that reaches into the reasoning
    if start is None:
        return {name: float("nan") for name, _, _ in dirs}
    hs = model(**enc.to(model.device), output_hidden_states=True).hidden_states
    out = {}
    for name, hs_index, d in dirs:
        h = hs[hs_index][0, start:].float()
        v = d.to(h.device).float()
        out[name] = float((h @ (v / v.norm())).mean()) if h.shape[0] else float("nan")
    return out


@torch.inference_mode()
def whole_reasoning_probe_score(model, tok, task: str, raw: str, hs_index: int, direction: torch.Tensor, max_len: int = 3072) -> float:
    """The probe's input regime (``build_probe.py --pooling whole``): mean hidden state over the
    reasoning tokens of chat-template(lowercased task) + lowercased reasoning, dotted with ``v̂``."""
    text = abdelnabi_probe_text(tok, task, raw)
    reasoning_lower = raw.rsplit("</think>", 1)[0].lower()
    prefix_len = len(text) - len(reasoning_lower + "</think>")
    spans = char_spans_to_token_spans(tok, text, [(prefix_len, prefix_len + len(reasoning_lower))], add_special_tokens=True)
    if not spans or spans[0][1] <= spans[0][0]:
        return float("nan")
    pooled = pooled_states(model, tok, text, [spans[0]], add_special_tokens=True, max_len=max_len)[0][hs_index]
    if torch.isnan(pooled).any():
        return float("nan")
    v = direction.to(pooled.device).float()
    return float(pooled.float() @ (v / v.norm()))


def arm_files(d: Path) -> list[Path]:
    """Generation files only (``<arm>.json``); every sidecar has a second dot in its name."""
    return [p for p in sorted(d.glob("*.json")) if p.name.count(".") == 1]


def load_arm(p: Path, model_name: str, family: str | None):
    """Traces plus the condition the arm was generated under (legacy sidecars are mapped by arm name;
    the family label comes from the sidecar unless overridden)."""
    side = json.loads(p.with_suffix(p.suffix + ".gen.json").read_text())
    cond = condition_from_sidecar(p, side)
    alpha, aware = cond.signed
    return load_generations(p, model_name, family or side.get("family") or "steer_actions", p.stem, alpha, aware), cond


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["annotate", "awareness", "project", "probe", "prompt_state", "project_incontext"], required=True)
    ap.add_argument("--steering-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "steering")
    ap.add_argument("--vectors-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "vectors")
    ap.add_argument("--probe", type=Path, default=REPO / "results" / "qwen3_4b" / "probe" / "probe_best.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-name", default="qwen3_4b")
    ap.add_argument("--family", default=None, help="trace family label (part of every trace id); default: the sidecar's `family`, else steer_actions")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "readouts.log")
    args = ap.parse_args()
    log = make_logger(args.log)

    arms = arm_files(args.steering_dir)
    log(f"arms: {[p.stem for p in arms]}")

    if args.stage == "annotate":
        judge = Judge(concurrency=args.concurrency, provider=args.provider)
        for p in arms:
            traces, _ = load_arm(p, args.model_name, args.family)
            jobs = [Job(id=t.id, prompt=venhoff_annotation_prompt(t.reasoning), meta=t.meta()) for t in traces if t.reasoning.strip()]
            log(json.dumps(run_jobs(judge, jobs, p.with_suffix(p.suffix + ".annotations.jsonl"), log=log)))

    elif args.stage == "awareness":
        template = load_abdelnabi_template()
        judge = Judge(concurrency=args.concurrency, max_tokens=2048, provider=args.provider)
        for p in arms:
            traces, _ = load_arm(p, args.model_name, args.family)
            jobs = []
            for t in traces:
                # unclosed traces are judged too (whole text as reasoning), the Tier M convention set in
                # judge_awareness.py, so the sweep, the probe set and the main run share one population
                r, a = abdelnabi_split(t.raw) if t.has_think_close else (t.raw, "")
                jobs.append(Job(id=t.id, prompt=abdelnabi_judge_prompt(template, task=t.task, reasoning=r, answer=a), meta=t.meta(), json_mode=True))
            log(json.dumps(run_jobs(judge, jobs, p.with_suffix(p.suffix + ".awareness.jsonl"), log=log)))

    elif args.stage == "prompt_state":
        model, tok = load_model(args.model)
        fv = feature_vectors(torch.load(args.vectors_dir / "mean_vectors.pt"))
        layers = json.loads((args.vectors_dir / "layers.json").read_text())["chosen"]
        probe = json.loads(args.probe.read_text())
        # (name, HF hidden-state index, direction): behaviour directions at decoder layer l = hidden_states[l + 1]
        dirs = [(b, layers[b]["layer"] + 1, fv[b][layers[b]["layer"]]) for b in layers] + [("probe", int(probe["layer"]), torch.tensor(probe["direction"], dtype=torch.float32))]
        for p in arms:
            out = p.with_suffix(p.suffix + ".prompt_state.json")
            if out.exists():
                log(f"{p.stem}: prompt states exist, skipping")
                continue
            traces, cond = load_arm(p, args.model_name, args.family)
            side = json.loads(p.with_suffix(p.suffix + ".gen.json").read_text())
            enable_thinking = bool(side.get("config", {}).get("enable_thinking", True))
            records = json.loads(p.read_text())
            if not records:
                continue
            is_actions = "data_item" in records[0]
            system = cond.system_prompt(AGENT_PROMPT.read_text() if is_actions else None)
            rows = []
            with applied(model, cond, probe=probe, vectors_dir=args.vectors_dir):
                log(f"{p.stem}: condition {cond.kind} {json.dumps(cond.info)}; system prompt {'present' if system else 'none'}")
                for t in traces:
                    rec = records[t.index]
                    user = action_prompt(rec["data_item"], t.framing) if is_actions else rec[t.framing]
                    prompt = format_prompt(tok, user, system=system, enable_thinking=enable_thinking)
                    row = {"id": t.id, "index": t.index, "framing": t.framing}
                    row.update({f"proj_lastprompt_{k}": v for k, v in last_token_projections(model, tok, prompt, dirs).items()})
                    rows.append(row)
                    if len(rows) % 100 == 0:
                        log(f"  {p.stem}: {len(rows)} prompt states")
            out.write_text(json.dumps({"arm": p.stem, "condition": cond.to_dict(), "directions": {n: {"hs_index": h} for n, h, _ in dirs}, "system_prompt": system, "rows": rows}, indent=1))
            log(f"{p.stem}: wrote {len(rows)} prompt states")

    elif args.stage == "project_incontext":
        model, tok = load_model(args.model)
        fv = feature_vectors(torch.load(args.vectors_dir / "mean_vectors.pt"))
        layers = json.loads((args.vectors_dir / "layers.json").read_text())["chosen"]
        probe = json.loads(args.probe.read_text())
        dirs = [(b, layers[b]["layer"] + 1, fv[b][layers[b]["layer"]]) for b in layers] + [("probe", int(probe["layer"]), torch.tensor(probe["direction"], dtype=torch.float32))]
        for p in arms:
            out = p.with_suffix(p.suffix + ".projections_incontext.json")
            if out.exists():
                log(f"{p.stem}: in-context projections exist, skipping")
                continue
            traces, cond = load_arm(p, args.model_name, args.family)
            side = json.loads(p.with_suffix(p.suffix + ".gen.json").read_text())
            enable_thinking = bool(side.get("config", {}).get("enable_thinking", True))
            records = json.loads(p.read_text())
            if not records:
                continue
            is_actions = "data_item" in records[0]
            system = cond.system_prompt(AGENT_PROMPT.read_text() if is_actions else None)
            rows = []
            with applied(model, cond, probe=probe, vectors_dir=args.vectors_dir):
                log(f"{p.stem}: condition {cond.kind} {json.dumps(cond.info)}; system prompt {'present' if system else 'none'}")
                for t in traces:
                    if not t.reasoning.strip():
                        continue
                    rec = records[t.index]
                    user = action_prompt(rec["data_item"], t.framing) if is_actions else rec[t.framing]
                    prompt = format_prompt(tok, user, system=system, enable_thinking=enable_thinking)
                    think = t.raw.split("</think>")[0]  # the model's own reasoning, from <think>
                    row = {"id": t.id, "index": t.index, "framing": t.framing}
                    row.update({f"proj_incontext_{k}": v for k, v in in_context_projections(model, tok, prompt, think, dirs).items()})
                    rows.append(row)
                    if len(rows) % 50 == 0:
                        log(f"  {p.stem}: {len(rows)} in-context projections")
            out.write_text(json.dumps({"arm": p.stem, "condition": cond.to_dict(), "directions": {n: {"hs_index": h} for n, h, _ in dirs}, "system_prompt": system, "rows": rows}, indent=1))
            log(f"{p.stem}: wrote {len(rows)} in-context projections")

    elif args.stage == "probe":
        model, tok = load_model(args.model)
        probe = json.loads(args.probe.read_text())
        hs_index, v_pos = int(probe["layer"]), torch.tensor(probe["direction"], dtype=torch.float32)
        for p in arms:
            out = p.with_suffix(p.suffix + ".probe_projections.json")
            if out.exists():
                log(f"{p.stem}: probe projections exist, skipping")
                continue
            traces, cond = load_arm(p, args.model_name, args.family)
            rows = []
            for t in traces:  # no condition applied: the unedited model reads every arm's text
                if not t.reasoning.strip():
                    continue
                rows.append({"id": t.id, "index": t.index, "framing": t.framing, "proj_probe": whole_reasoning_probe_score(model, tok, t.task, t.raw, hs_index, v_pos)})
                if len(rows) % 100 == 0:
                    log(f"  {p.stem}: {len(rows)} probe projections")
            out.write_text(json.dumps({"arm": p.stem, "condition": cond.to_dict(), "probe_hs_index": hs_index, "model_edited": False, "pooling": "whole reasoning, probe input regime", "rows": rows}, indent=1))
            log(f"{p.stem}: wrote {len(rows)} probe projections")

    else:  # project
        model, tok = load_model(args.model)
        fv = feature_vectors(torch.load(args.vectors_dir / "mean_vectors.pt"))
        layers = json.loads((args.vectors_dir / "layers.json").read_text())["chosen"]
        probe = json.loads(args.probe.read_text())
        dirs = {b: (layers[b]["layer"], fv[b][layers[b]["layer"]]) for b in layers}
        for p in arms:
            out = p.with_suffix(p.suffix + ".projections.json")
            if out.exists():
                log(f"{p.stem}: projections exist, skipping")
                continue
            traces, cond = load_arm(p, args.model_name, args.family)
            alpha, aware = cond.signed
            rows = []
            # the arm's own manipulation is re-applied (weight edit or residual add), so the projections
            # are of the steered model's activations; prompt conditions leave the model untouched
            with applied(model, cond, probe=probe, vectors_dir=args.vectors_dir):
                log(f"{p.stem}: condition {cond.kind} {json.dumps(cond.info)}")
                for t in traces:
                    if not t.reasoning.strip():
                        continue
                    row = {"id": t.id, "index": t.index, "framing": t.framing, "executed": t.executed}
                    think = t.raw.split("</think>")[0]  # begins with <think>, the convention the vectors were built under
                    for b, (layer, d) in dirs.items():
                        # Venhoff indexing: decoder layer `layer` output = HF hidden_states[layer + 1];
                        # window (1, ∞) drops the attention-sink position 0
                        row[f"proj_{b}"] = mean_projection(model, tok, think, layer + 1, d, token_window=(1, 10**9))
                    rows.append(row)
                    if len(rows) % 50 == 0:
                        log(f"  {p.stem}: {len(rows)} projections")
            out.write_text(json.dumps({"arm": p.stem, "alpha": alpha, "aware": aware, "condition": cond.to_dict(), "directions": {b: {"layer": l} for b, (l, _) in dirs.items()}, "rows": rows}, indent=1))
            log(f"{p.stem}: wrote {len(rows)} projections")


if __name__ == "__main__":
    main()
