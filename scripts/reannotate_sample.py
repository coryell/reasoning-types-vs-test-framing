#!/usr/bin/env python
"""Judge-stability check (EXECUTION_PLAN_2 §2.4): re-annotate a random sample of already-annotated
traces with the identical prompt and compare the two annotations.

Writes the second annotations to ``results/annotations_rerun/<model>/<family>/<arm>.jsonl`` (ids get a
``#rerun`` suffix so the cache treats them as new jobs) and a comparison table to
``results/tier1/judge_stability.csv`` with, per trace: word-level span-label agreement, absolute
density differences per behaviour, and the correlation of densities across the sample.

Agreement is measured at the word level: each word of the reasoning is assigned the label of the
counted span that covers it (or "none"); agreement is the fraction of words with the same label in
both annotations. Per-behaviour Jaccard is |A∩B| / |A∪B| over the word sets carrying that label.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from d10 import parse
from d10.judge import Job, Judge, load_results, record_ok, run_jobs
from d10.parse import LABELS
from d10.prompts import venhoff_annotation_prompt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ANN = REPO / "results" / "annotations"
DEFAULT_RERUN = REPO / "results" / "annotations_rerun"
DEFAULT_OUT = REPO / "results" / "tier1" / "judge_stability.csv"
DEFAULT_LOG = REPO / "logs" / "reannotate_sample.log"


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


def word_labels(spans: list[parse.Span], source: str) -> list[str]:
    """Label per word of ``source``: the label of the first counted span whose text starts there."""
    labels = ["none"] * len(source.split())
    # map character offsets to word indices
    words = source.split()
    starts = []
    pos = 0
    for w in words:
        pos = source.find(w, pos)
        starts.append(pos)
        pos += len(w)
    for s in spans:
        if not s.counted:
            continue
        cpos = source.find(s.text)
        if cpos < 0:
            continue  # normalised-only match: cannot place it exactly, leave unlabelled
        cend = cpos + len(s.text)
        for wi, ws in enumerate(starts):
            if ws >= cpos and ws < cend and labels[wi] == "none":
                labels[wi] = s.label
    return labels


def compare(rec_a: dict, rec_b: dict) -> dict:
    src = rec_a["meta"]["reasoning"]
    sa = parse.parse_annotation(rec_a["text"], src)
    sb = parse.parse_annotation(rec_b["text"], src)
    la, lb = word_labels(sa, src), word_labels(sb, src)
    ma, mb = parse.trace_metrics(sa, src), parse.trace_metrics(sb, src)
    out = {
        "id": rec_a["id"],
        "model": rec_a["meta"]["model"],
        "family": rec_a["meta"]["family"],
        "arm": rec_a["meta"]["arm"],
        "words": len(la),
        "word_agreement": float(np.mean([x == y for x, y in zip(la, lb)])) if la else math.nan,
        "n_spans_a": ma["n_spans_counted"],
        "n_spans_b": mb["n_spans_counted"],
    }
    for b in LABELS:
        A = {i for i, x in enumerate(la) if x == b}
        B = {i for i, x in enumerate(lb) if x == b}
        out[f"jaccard_{b}"] = len(A & B) / len(A | B) if (A | B) else math.nan
        out[f"density_a_{b}"] = ma[f"density_{b}"]
        out[f"density_b_{b}"] = mb[f"density_{b}"]
        out[f"abs_ddensity_{b}"] = abs(ma[f"density_{b}"] - mb[f"density_{b}"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANN)
    ap.add_argument("--rerun-dir", type=Path, default=DEFAULT_RERUN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    log = make_logger(args.log)

    pool: list[tuple[Path, dict]] = []
    for path in sorted(args.annotations.glob("*/*/*.jsonl")):
        for rec in load_results(path).values():
            if record_ok(rec):
                pool.append((path, rec))
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    log(f"sampled {len(sample)} of {len(pool)} annotated traces (seed {args.seed})")

    by_path: dict[Path, list[tuple[Job, dict]]] = {}
    for path, rec in sample:
        job = Job(id=rec["id"] + "#rerun", prompt=venhoff_annotation_prompt(rec["meta"]["reasoning"]), meta=rec["meta"])
        by_path.setdefault(path, []).append((job, rec))

    rows = []
    judge = None if args.compare_only else Judge(concurrency=args.concurrency)
    for path, items in by_path.items():
        rerun_path = args.rerun_dir / path.relative_to(args.annotations)
        if not args.compare_only:
            run_jobs(judge, [j for j, _ in items], rerun_path, log=log)
        rerun = load_results(rerun_path)
        for job, rec in items:
            rb = rerun.get(job.id)
            if rb and record_ok(rb):
                rows.append(compare(rec, rb))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if df.empty:
        log("no comparisons")
        return
    summary = {
        "n": len(df),
        "word_agreement_mean": float(df.word_agreement.mean()),
        "word_agreement_median": float(df.word_agreement.median()),
    }
    for b in LABELS:
        a, bb = df[f"density_a_{b}"], df[f"density_b_{b}"]
        summary[f"density_corr_{b}"] = float(a.corr(bb)) if a.std() > 0 and bb.std() > 0 else math.nan
        summary[f"jaccard_median_{b}"] = float(df[f"jaccard_{b}"].median())
        summary[f"abs_ddensity_mean_{b}"] = float(df[f"abs_ddensity_{b}"].mean())
    log(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
