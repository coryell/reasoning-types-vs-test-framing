#!/usr/bin/env python
"""Readouts for the steering arms (EXECUTION_PLAN_2 §M3): morphology annotation (Venhoff's prompt),
awareness verbalization (Abdelnabi's judge), and representation-level morphology — the mean
projection of each generated reasoning trace onto the rebuilt backtracking and uncertainty-estimation
directions at their chosen layers, computed on the steered model (the arm's weight edit re-applied).

    uv run python scripts/readouts.py --stage annotate      # Venhoff annotations per arm
    uv run python scripts/readouts.py --stage awareness     # awareness judgments per arm
    uv run python scripts/readouts.py --stage project       # projections (GPU)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from d10.activations import mean_projection
from d10.conditions import applied, condition_from_sidecar
from d10.generate import load_model
from d10.judge import Job, Judge, run_jobs
from d10.prompts import abdelnabi_judge_prompt, load_abdelnabi_template, venhoff_annotation_prompt
from d10.shipped import abdelnabi_split, load_generations
from d10.venhoff import feature_vectors

REPO = Path(__file__).resolve().parents[1]


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


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
    ap.add_argument("--stage", choices=["annotate", "awareness", "project"], required=True)
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
