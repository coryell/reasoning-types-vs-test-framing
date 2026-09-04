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
from d10.generate import load_model
from d10.judge import Job, Judge, run_jobs
from d10.prompts import abdelnabi_judge_prompt, load_abdelnabi_template, venhoff_annotation_prompt
from d10.shipped import abdelnabi_split, load_generations
from d10.steer import apply_edit, random_direction_like, rank_rows, undo_edit
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["annotate", "awareness", "project"], required=True)
    ap.add_argument("--steering-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "steering")
    ap.add_argument("--vectors-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "vectors")
    ap.add_argument("--probe", type=Path, default=REPO / "results" / "qwen3_4b" / "probe" / "probe_best.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-name", default="qwen3_4b")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "readouts.log")
    args = ap.parse_args()
    log = make_logger(args.log)

    arms = arm_files(args.steering_dir)
    log(f"arms: {[p.stem for p in arms]}")

    if args.stage == "annotate":
        judge = Judge(concurrency=args.concurrency, provider=args.provider)
        for p in arms:
            side = json.loads(p.with_suffix(p.suffix + ".gen.json").read_text())
            traces = load_generations(p, args.model_name, "steer_actions", p.stem, side.get("alpha", 0.0), bool(side.get("aware")))
            jobs = [Job(id=t.id, prompt=venhoff_annotation_prompt(t.reasoning), meta=t.meta()) for t in traces if t.reasoning.strip()]
            log(json.dumps(run_jobs(judge, jobs, p.with_suffix(p.suffix + ".annotations.jsonl"), log=log)))

    elif args.stage == "awareness":
        template = load_abdelnabi_template()
        judge = Judge(concurrency=args.concurrency, max_tokens=2048, provider=args.provider)
        for p in arms:
            side = json.loads(p.with_suffix(p.suffix + ".gen.json").read_text())
            traces = load_generations(p, args.model_name, "steer_actions", p.stem, side.get("alpha", 0.0), bool(side.get("aware")))
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
        v_pos = torch.tensor(probe["direction"], dtype=torch.float32)
        v_shuf = torch.tensor(probe["shuffled_direction"], dtype=torch.float32)
        dirs = {b: (layers[b]["layer"], fv[b][layers[b]["layer"]]) for b in layers}
        for p in arms:
            out = p.with_suffix(p.suffix + ".projections.json")
            if out.exists():
                log(f"{p.stem}: projections exist, skipping")
                continue
            side = json.loads(p.with_suffix(p.suffix + ".gen.json").read_text())
            alpha, aware = side.get("alpha", 0.0), bool(side.get("aware"))
            gen_seed = int(side.get("config", {}).get("seed", args.seed))
            v_shuf_scaled = v_shuf / v_shuf.norm() * v_pos.norm()
            rnd = random_direction_like(v_pos, seed=gen_seed)
            vec = {"aware": v_pos, "unaware": v_pos, "random_aware": rnd, "random_samerows_aware": rnd, "shuffled_aware": v_shuf_scaled, "shuffled_samerows_aware": v_shuf_scaled}.get(p.stem)
            fixed_rows = rank_rows(model, v_pos) if p.stem.endswith("_samerows_aware") else None
            edit = apply_edit(model, vec, alpha, aware, rows=fixed_rows) if vec is not None and alpha else None
            rows = []
            try:
                for t in load_generations(p, args.model_name, "steer_actions", p.stem, alpha, aware):
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
            finally:
                if edit is not None:
                    undo_edit(model, edit)
            out.write_text(json.dumps({"arm": p.stem, "alpha": alpha, "aware": aware, "directions": {b: {"layer": l} for b, (l, _) in dirs.items()}, "rows": rows}, indent=1))
            log(f"{p.stem}: wrote {len(rows)} projections")


if __name__ == "__main__":
    main()
