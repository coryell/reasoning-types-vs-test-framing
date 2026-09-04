#!/usr/bin/env python
"""Rebuild Venhoff et al.'s six behaviour directions on a local model, their recipe, then pick the
steering layer for backtracking and uncertainty-estimation by steering (EXECUTION_PLAN_2 §2b).

Stages (each resumable, each writes under ``--out-dir``):
  1. generate   500 traces on their ``messages.py`` tasks → ``responses.json`` (their layout)
  2. annotate   Venhoff's prompt via the judge → ``annotated_thinking`` filled in
  3. extract    per-span means with their windows → ``mean_vectors.pt`` (+ ``mean_vectors_nontest.pt``
                built from spans without self-referential test language, for the geometry step)
  4. layers     steer ± each behaviour at candidate layers on 20 eval tasks, annotate, pick the layer
                with the largest positive-minus-negative token fraction → ``layers.json``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

from d10 import parse
from d10.activations import ResidualAdd, pooled_states
from d10.generate import GenConfig, format_prompt, generate, load_model, write_sidecar
from d10.judge import Job, Judge, load_results, record_ok, run_jobs
from d10.prompts import venhoff_annotation_prompt
from d10.shipped import ASSETS
from d10.venhoff import BEHAVIOURS, MeanVectors, accumulate_trace, feature_vectors, label_positions

REPO = Path(__file__).resolve().parents[1]
VENHOFF = ASSETS / "steering_thinking_llms"
STEER_BEHAVIOURS = ["backtracking", "uncertainty-estimation"]


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


def extract_thinking(response: str) -> str:
    """Their ``extract_thinking_process``: between ``<think>`` and ``</think>`` (or to the end)."""
    if "<think>" in response:
        response = response.split("<think>", 1)[1]
    return response.split("</think>", 1)[0].strip()


def load_messages():
    sys.path.insert(0, str(VENHOFF))
    from messages.messages import eval_messages, messages  # noqa: E402

    return messages, eval_messages


def token_fractions(annotated: str, full_response: str, tok) -> dict[str, float]:
    """Their ``get_label_counts``: tokens per label ÷ tokens over the steered labels."""
    lp = label_positions(annotated, full_response, tok)
    counts = {b: sum(e - s for s, e in lp.get(b, [])) for b in STEER_BEHAVIOURS + ["example-testing", "adding-knowledge"]}
    tot = sum(counts.values())
    return {b: (c / tot if tot else 0.0) for b, c in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "vectors")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=1000, help="their setting")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--candidate-layers", type=int, nargs="+", default=None, help="default: 0.35/0.5/0.65 of depth")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42, help="their seed")
    ap.add_argument("--stages", nargs="+", default=["generate", "annotate", "extract", "layers"])
    ap.add_argument("--coefficient", type=float, default=1.0, help="layers: |coefficient| of the residual add (their setting is 1; +1 saturates Qwen3-4B). Values other than 1 write layers_c<coef>.json etc. beside the originals")
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "build_vectors.log")
    args = ap.parse_args()
    log = make_logger(args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = args.out_dir / "responses.json"
    messages, eval_messages = load_messages()
    model = tok = None

    def ensure_model():
        nonlocal model, tok
        if model is None:
            log(f"loading {args.model}")
            model, tok = load_model(args.model)
        return model, tok

    # ---- 1. generate (their generate_responses.py: shuffle with seed, take n, chat template, decode)
    if "generate" in args.stages and not responses_path.exists():
        import random

        model, tok = ensure_model()
        sel = list(messages)
        random.Random(args.seed).shuffle(sel)
        sel = sel[: args.n_samples]
        cfg = GenConfig(model_name=args.model, max_new_tokens=args.max_tokens, batch_size=args.batch_size, seed=args.seed, notes={"stage": "venhoff responses"})
        prompts = [format_prompt(tok, m["content"]) for m in sel]
        log(f"generating {len(prompts)} Venhoff-task traces")
        gens = generate(model, tok, prompts, cfg, log=log)
        data = [{"original_message": m, "full_response": g.text, "thinking_process": extract_thinking(g.text), "annotated_thinking": "", "finished": g.finished} for m, g in zip(sel, gens)]
        responses_path.write_text(json.dumps(data, indent=1))
        write_sidecar(responses_path, cfg, {"n": len(data)})
        log(f"wrote {responses_path}; {sum(d['finished'] for d in data)} finished, {sum('</think>' in d['full_response'] for d in data)} closed </think>")

    # ---- 2. annotate
    if "annotate" in args.stages:
        data = json.loads(responses_path.read_text())
        jobs = [Job(id=f"venhoff/{i}", prompt=venhoff_annotation_prompt(d["thinking_process"]), meta={"i": i}) for i, d in enumerate(data) if d["thinking_process"].strip()]
        judge = Judge(concurrency=args.concurrency, provider=args.provider)
        log(json.dumps(run_jobs(judge, jobs, args.out_dir / "annotations.jsonl", log=log)))
        recs = load_results(args.out_dir / "annotations.jsonl")
        n = 0
        for i, d in enumerate(data):
            r = recs.get(f"venhoff/{i}")
            if r and record_ok(r):
                d["annotated_thinking"] = r["text"]
                n += 1
        responses_path.write_text(json.dumps(data, indent=1))
        log(f"annotated {n}/{len(data)}")

    # ---- 3. extract (their train_vectors.py, plus a non-test-language variant)
    if "extract" in args.stages:
        model, tok = ensure_model()
        data = [d for d in json.loads(responses_path.read_text()) if d.get("annotated_thinking")]
        mv_all, mv_nontest, mv_test = MeanVectors(), MeanVectors(), MeanVectors()
        n_spans = 0
        for i, d in enumerate(data):
            full = d["full_response"]

            def pooled(windows, full=full):
                p = pooled_states(model, tok, full, windows, add_special_tokens=False)
                return p[:, 1:, :]  # decoder-layer outputs only, their indexing

            n_spans += accumulate_trace(mv_all, pooled, d["annotated_thinking"], full, tok)
            accumulate_trace(mv_nontest, pooled, d["annotated_thinking"], full, tok, span_filter=lambda lab, txt: not parse.is_test_span(txt))
            accumulate_trace(mv_test, pooled, d["annotated_thinking"], full, tok, span_filter=lambda lab, txt: parse.is_test_span(txt))
            if (i + 1) % 50 == 0:
                log(f"  extracted {i + 1}/{len(data)} traces, {n_spans} spans")
        for name, mv in (("mean_vectors", mv_all), ("mean_vectors_nontest", mv_nontest), ("mean_vectors_testlex", mv_test)):
            torch.save(mv.as_dict(), args.out_dir / f"{name}.pt")
        counts = {k: v["count"] for k, v in mv_all.as_dict().items()}
        (args.out_dir / "counts.json").write_text(json.dumps({"all": counts, "nontest": {k: v["count"] for k, v in mv_nontest.as_dict().items()}, "testlex": {k: v["count"] for k, v in mv_test.as_dict().items()}}, indent=1))
        log(f"mean vectors: {counts}")

    # ---- 4. layer selection by steering (their evaluate_steering.py, restricted)
    if "layers" in args.stages:
        model, tok = ensure_model()
        mv = torch.load(args.out_dir / "mean_vectors.pt")
        fv = feature_vectors(mv)
        n_layers = len(model.model.layers)
        cands = args.candidate_layers or [round(n_layers * f) for f in (0.35, 0.5, 0.65)]
        import random

        ev = list(eval_messages)
        random.Random(args.seed).shuffle(ev)
        ev = ev[: args.n_eval]
        cfg = GenConfig(model_name=args.model, max_new_tokens=args.max_tokens, batch_size=args.batch_size, seed=args.seed)
        prompts = [format_prompt(tok, m["content"]) for m in ev]
        sfx = "" if args.coefficient == 1.0 else f"_c{args.coefficient:g}"
        runs: dict[str, list[str]] = {}
        gens = generate(model, tok, prompts, cfg, log=log)
        runs["original"] = [g.text for g in gens]
        for b in STEER_BEHAVIOURS:
            for layer in cands:
                vec = (fv[b][layer] * abs(args.coefficient)).to(model.device).to(torch.bfloat16)
                for sign, name in ((1, "pos"), (-1, "neg")):
                    with ResidualAdd(model, layer, sign * vec):
                        gens = generate(model, tok, prompts, cfg, log=log)
                    runs[f"{b}/{layer}/{name}"] = [g.text for g in gens]
                    log(f"steered {b} layer {layer} {name} (coefficient {args.coefficient})")
        (args.out_dir / f"layer_runs{sfx}.json").write_text(json.dumps(runs, indent=1))
        jobs = [Job(id=f"{k}/{i}", prompt=venhoff_annotation_prompt(extract_thinking(t)), meta={"run": k, "i": i}) for k, texts in runs.items() for i, t in enumerate(texts) if extract_thinking(t).strip()]
        judge = Judge(concurrency=args.concurrency, provider=args.provider)
        log(json.dumps(run_jobs(judge, jobs, args.out_dir / f"layer_annotations{sfx}.jsonl", log=log)))
        recs = load_results(args.out_dir / f"layer_annotations{sfx}.jsonl")
        frac: dict[str, dict[str, float]] = {}
        for k, texts in runs.items():
            vals = []
            for i, t in enumerate(texts):
                r = recs.get(f"{k}/{i}")
                if r and record_ok(r):
                    vals.append(token_fractions(r["text"], t, tok))
            frac[k] = {b: (sum(v[b] for v in vals) / len(vals) if vals else float("nan")) for b in STEER_BEHAVIOURS}
            # coherence of each steered run, so a layer is not chosen on saturated output
            frac[k]["closure"] = sum("</think>" in t for t in texts) / len(texts)
            frac[k]["repetition"] = sum(parse.repetition_rate(extract_thinking(t)) for t in texts) / len(texts)
        chosen = {}
        for b in STEER_BEHAVIOURS:
            best, score = None, -1.0
            for layer in cands:
                s = frac[f"{b}/{layer}/pos"][b] - frac[f"{b}/{layer}/neg"][b]
                if s > score:
                    best, score = layer, s
            chosen[b] = {"layer": best, "pos_minus_neg": score, "original": frac["original"][b], "pos": frac[f"{b}/{best}/pos"][b], "neg": frac[f"{b}/{best}/neg"][b]}
        (args.out_dir / f"layers{sfx}.json").write_text(json.dumps({"coefficient": args.coefficient, "candidates": cands, "n_eval": len(ev), "fractions": frac, "chosen": chosen}, indent=1))
        log(f"chosen layers: {json.dumps(chosen)}")


if __name__ == "__main__":
    main()
