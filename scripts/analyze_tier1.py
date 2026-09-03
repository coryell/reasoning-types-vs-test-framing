#!/usr/bin/env python
"""First-pass analysis of the Tier 1 annotations (EXECUTION_PLAN_2 §2.2–2.3). Works on partial data.

Reads every ``results/annotations/<model>/<family>/<arm>.jsonl``, parses each annotation against the
reasoning it was made from, computes the per-trace metrics, and writes tables and figures to
``results/tier1/``:

    traces.csv            one row per annotated trace (metrics, execution, shipped judge labels)
    coverage.csv          per arm: how many traces are annotated, truncated, unparseable
    arm_means.csv         per arm × behaviour: mean density / coverage / any with bootstrap 95% CI
    paired_contrasts.csv  each arm vs alpha0.0, paired by item: Δ with CI, paired t, Wilcoxon
    sign_symmetry.csv     Δ at the largest aware alpha vs Δ at the largest unaware alpha
    flips_2x2.csv         actions: compliance 2×2 vs baseline, McNemar
    flips_by_class.csv    actions: Δdensity by flip class
    verbalization.csv     triggers/harmbench: density by the shipped awareness judgment
    verbalization_contrasts.csv  steering Δ among items in the same judged class in both arms
    noise_floor.csv       greedy vs sampled decode of the same item at the same α (Distill-Qwen)
    proxy_vs_judge.csv    lexical proxies against judge densities
    SUMMARY.md            the tables that matter, readable
    fig_*.png             dose–response (density, coverage), length, flip-class deltas

All logic lives in ``d10.analysis`` so it can be unit-tested; this file is the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from d10 import analysis as A

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ANN = REPO / "results" / "annotations"
DEFAULT_OUT = REPO / "results" / "tier1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = A.load_traces(args.annotations, args.models, args.families)
    if df.empty:
        print("no annotations found")
        return
    df.to_csv(args.out / "traces.csv", index=False)
    cov = A.coverage_table(df)
    cov.to_csv(args.out / "coverage.csv", index=False)
    means = A.arm_means(df)
    means.to_csv(args.out / "arm_means.csv", index=False)
    contrasts = A.paired_contrasts(df)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    sym = A.sign_symmetry(contrasts)
    sym.to_csv(args.out / "sign_symmetry.csv", index=False)
    two, by_class = A.flips(df)
    two.to_csv(args.out / "flips_2x2.csv", index=False)
    by_class.to_csv(args.out / "flips_by_class.csv", index=False)
    verb = A.verbalization(df)
    verb.to_csv(args.out / "verbalization.csv", index=False)
    verb_contrasts = A.verbalization_contrasts(df)
    verb_contrasts.to_csv(args.out / "verbalization_contrasts.csv", index=False)
    noise = A.noise_floor(df)
    noise.to_csv(args.out / "noise_floor.csv", index=False)
    proxy = A.proxy_vs_judge(df)
    proxy.to_csv(args.out / "proxy_vs_judge.csv", index=False)

    for fam in sorted(df.family.unique()):
        A.fig_dose(means, "density", fam, args.out / f"fig_dose_density_{fam}.png", f"Dose–response, {fam}")
        A.fig_dose(means, "coverage", fam, args.out / f"fig_dose_coverage_{fam}.png", f"Dose–response, {fam}")
    A.fig_words(means, args.out / "fig_words.png")
    A.fig_flip_classes(by_class, args.out / "fig_flip_classes.png")

    A.write_summary(args.out / "SUMMARY.md", df, cov, means, contrasts, sym, two, by_class, verb, proxy, verb_contrasts, noise)
    print(f"wrote {args.out}/SUMMARY.md and {len(list(args.out.glob('*.csv')))} csv, {len(list(args.out.glob('*.png')))} png")
    print(json.dumps({"traces": len(df), "ok": int(df.ok.sum()), "arms": int(cov.shape[0])}))


if __name__ == "__main__":
    main()
