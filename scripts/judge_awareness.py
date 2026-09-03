#!/usr/bin/env python
"""Label any generation file with Abdelnabi & Salem's awareness judge (their prompt, filled as their
batch script fills it) and summarise under the paper's class rule.

    uv run python scripts/judge_awareness.py --input results/qwen3_4b/triggers_gate.json \\
        --model-name qwen3_4b --family triggers_gate --provider openai

Writes ``<out>.jsonl`` (judge records), ``<summary>.md`` and ``<summary>.stats.json`` (the gate
verdict, positives, rates).

Sides that never closed ``</think>`` (the generation hit ``max_new_tokens``) are judged too by
default, with the whole text as the reasoning and an empty answer — a deviation from their pipeline,
which drops them. Awareness verbalization appears early in a trace, so a truncated trace still
carries it; the ``task_performance`` dimension is meaningless for these and is not used. The summary
reports positives among closed and unclosed sides separately. ``--closed-only`` restores their rule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from d10.awareness import summarise
from d10.judge import Job, Judge, load_results, run_jobs
from d10.prompts import abdelnabi_judge_prompt, load_abdelnabi_template
from d10.shipped import abdelnabi_split, load_generations

REPO = Path(__file__).resolve().parents[1]


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model-name", required=True, help="short id used in trace ids, e.g. qwen3_4b")
    ap.add_argument("--family", required=True)
    ap.add_argument("--arm", default="alpha0.0")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--aware", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="default: <input>.awareness.jsonl")
    ap.add_argument("--summary", type=Path, default=None, help="default: <input>.awareness.md")
    ap.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--rpm", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "judge_awareness.log")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--closed-only", action="store_true", help="their rule: skip sides without </think>")
    args = ap.parse_args()
    log = make_logger(args.log)
    out = args.out or args.input.with_suffix(args.input.suffix + ".awareness.jsonl")
    summary = args.summary or args.input.with_suffix(args.input.suffix + ".awareness.md")

    template = load_abdelnabi_template()
    all_traces = load_generations(args.input, args.model_name, args.family, args.arm, args.alpha, args.aware)
    traces = [t for t in all_traces if t.has_think_close or not args.closed_only]
    if args.limit:
        traces = traces[: args.limit]
    log(f"{args.input}: {len(all_traces)} sides, {len(traces)} to judge ({sum(t.has_think_close for t in traces)} closed)")
    jobs = []
    for t in traces:
        reasoning, answer = abdelnabi_split(t.raw) if t.has_think_close else (t.raw, "")
        jobs.append(Job(id=t.id, prompt=abdelnabi_judge_prompt(template, task=t.task, reasoning=reasoning, answer=answer), meta=t.meta(), json_mode=True))
    if not args.summary_only:
        judge = Judge(concurrency=args.concurrency, max_tokens=2048, rpm=args.rpm or None, provider=args.provider)
        log(json.dumps(run_jobs(judge, jobs, out, log=log)))
    records = {k: v for k, v in load_results(out).items() if k in {j.id for j in jobs}}
    text, stats = summarise(records, n_sides_total=len(all_traces), title=f"{args.model_name} / {args.family} / {args.arm} — awareness judgments", gate=args.gate)
    summary.write_text(text)
    summary.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    log(f"summary → {summary}; verdict {stats['verdict']} ({stats['positives']} positives of {stats['generated']} generated)")
    print(text)


if __name__ == "__main__":
    main()
