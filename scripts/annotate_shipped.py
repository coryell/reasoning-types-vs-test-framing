#!/usr/bin/env python
"""Tier 1 runner: annotate the authors' shipped steered traces with Venhoff's six-behaviour prompt.

Writes one JSONL per (model, family, arm) under ``results/annotations/`` with the judge output and all
trace metadata (including the reasoning text, so downstream analysis is self-contained). Reruns are
free: cached results are skipped. Progress goes to ``logs/annotate_shipped.log``.

Examples::

    uv run python scripts/annotate_shipped.py --dry-run
    uv run python scripts/annotate_shipped.py --models qwq --families actions --limit 3   # smoke test
    uv run python scripts/annotate_shipped.py --families actions actions_sampling         # the core run
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from d10.judge import DEFAULT_MAX_TOKENS, Job, Judge, run_jobs
from d10.prompts import venhoff_annotation_prompt
from d10.shipped import FAMILIES, MODELS, iter_traces, list_families

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "annotations"
DEFAULT_LOG = REPO / "logs" / "annotate_shipped.log"

#: Rough per-trace token estimate for --dry-run: prompt ≈ template (~230 tokens) + reasoning; output ≈
#: reasoning plus label markup. Words → tokens at ~1.4.
TEMPLATE_TOKENS = 230


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


def build_jobs(model: str, family: str, limit: int | None, log) -> dict[str, list[Job]]:
    """Group jobs by arm. Traces with empty reasoning are skipped and counted."""
    by_arm: dict[str, list[Job]] = defaultdict(list)
    skipped = 0
    for tr in iter_traces(model, family):
        if not tr.reasoning.strip():
            skipped += 1
            continue
        if limit is not None and len(by_arm[tr.arm]) >= limit:
            continue
        by_arm[tr.arm].append(Job(id=tr.id, prompt=venhoff_annotation_prompt(tr.reasoning), meta=tr.meta()))
    if skipped:
        log(f"{model}/{family}: skipped {skipped} traces with empty reasoning")
    return by_arm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--limit", type=int, default=None, help="max traces per arm (smoke tests)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rpm", type=int, default=20, help="client-side requests-per-minute ceiling (0 = none)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--dry-run", action="store_true", help="count jobs and estimate cost; no API calls")
    ap.add_argument("--retry-truncated", action="store_true", help="resend prompts whose completion was truncated")
    args = ap.parse_args()

    log = make_logger(args.log)
    log(f"start: models={args.models} families={args.families} limit={args.limit} dry_run={args.dry_run}")

    judge = None if args.dry_run else Judge(concurrency=args.concurrency, max_tokens=args.max_tokens, rpm=args.rpm or None)
    summaries = []
    total_jobs = 0
    est_tokens = 0
    for model in args.models:
        available = list_families(model)
        for family in args.families:
            if family not in available:
                log(f"{model}/{family}: no files, skipping")
                continue
            by_arm = build_jobs(model, family, args.limit, log)
            for arm, jobs in sorted(by_arm.items()):
                total_jobs += len(jobs)
                if args.dry_run:
                    words = sum(j.meta["reasoning_words"] for j in jobs)
                    est_tokens += int(2 * 1.4 * words + TEMPLATE_TOKENS * len(jobs))
                    log(f"{model}/{family}/{arm}: {len(jobs)} jobs, {words:,} reasoning words")
                    continue
                out_path = args.out / model / family / f"{arm}.jsonl"
                s = run_jobs(judge, jobs, out_path, log=log, retry_truncated=args.retry_truncated)
                summaries.append(s)
                log(f"{model}/{family}/{arm}: {json.dumps(s)}")

    if args.dry_run:
        # Cost estimate assumes output tokens ≈ input tokens; the output price dominates.
        approx_cost = est_tokens / 2 * (2.5e-6 + 10e-6)
        log(f"dry run: {total_jobs:,} jobs, ~{est_tokens:,} tokens, ~${approx_cost:.0f}")
        return
    run = sum(s["run"] for s in summaries)
    errors = sum(s["errors"] for s in summaries)
    cost = sum(s["est_cost_usd"] for s in summaries)
    log(f"done: {total_jobs:,} jobs, {run:,} run now, {errors} errors, ~${cost:.2f} this invocation")
    if errors:
        log("rerun the same command to retry the errors; cached successes are skipped")


if __name__ == "__main__":
    main()
