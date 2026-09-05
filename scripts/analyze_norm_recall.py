#!/usr/bin/env python
"""Norm-recall in the reasoning: how much of a trace cites rules, laws, harms or ethics, and how
steering moves it.

Written 4 Sep 2026 after reading the `adding-knowledge` spans that vanish in items that flip from
refusal to execution under aware steering: at baseline they are "my guidelines prohibit this" /
"this is illegal"; in the aware arm they are "the tool is X and the parameters are Y". Venhoff's
taxonomy is content-blind, so this shift is invisible to the six-behaviour densities. The measure
here is **norm density**: spans matching ``d10.parse.NORM_LEXICON`` that are *not* test spans
(``TEST_LEXICON``), per 100 words of reasoning. The judge's annotation is used only for segmentation
into spans (99% of words are inside a span); a sentence-split variant that never touches the labels
is reported beside it as the judge-free check, and a narrow lexicon (rules and legality only) as the
lexicon sensitivity check. Exploratory: the lexicon was written after reading aware-arm spans.

Paired by item against the α = 0 arm (Tier 1) or the `baseline` arm (Qwen3-4B), bootstrap 95% CIs
over item differences, and split by compliance class (`refuse_both` / `comply_both` / `gained` /
`lost`) so the decision-change confound is visible: refusing traces cite norms far more than complying
ones at baseline, so `gained` items always lose norm density; the informative cells are `all` and
`refuse_both`.

Writes ``results/norm_recall/norm_recall.csv`` (every cell), ``per_word.csv`` (per-word contributions
for the cleanest arms) and ``SUMMARY.md``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d10.judge import load_results  # noqa: E402
from d10.norms import is_norm_span  # noqa: E402
from d10.parse import SPAN_RE, is_test_span  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MIN_WORDS = 20  # traces shorter than this are excluded from densities (they blow up per-100-word rates)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_GROUPS = [
    "guideline\\w*", "polic(?:y|ies)", "illegal|unlawful", "unethical|ethic\\w*", "law\\w*", "crim\\w*", "fraud\\w*",
    "harm\\w*", "malicious", "prohibit\\w*|not allowed", "violat\\w*", "decepti\\w*|mislead\\w*", "consequence\\w*",
    "dangerous", "scam\\w*|defam\\w*|counterfeit|forg(?:e|ed|ery|ing)", "misuse\\w*|abuse\\w*", "inappropriate",
    "responsib\\w*", "safety|privacy|consent", "legitimate|authoriz\\w*",
]
WORD_RES = [re.compile(r"\b(?:" + w + r")\b", re.I) for w in WORD_GROUPS]


def trace_measures(annotated: str, words: float, source: str | None = None) -> dict:
    """Per-trace densities (per 100 words of reasoning). ``source`` = the model's own reasoning text
    (from the annotation record's meta), used for the judge-free sentence variant."""
    per100 = max(float(words), 1.0) / 100.0
    # every regex span is counted here (label and verbatim-relocation not checked), unlike
    # ``parse.trace_metrics``; relocation failures are 0.3% of spans in this data
    spans = [m.group(2) for m in SPAN_RE.finditer(annotated)]
    nontest = [s for s in spans if not is_test_span(s)]
    out = {
        "norm": sum(is_norm_span(s) for s in nontest) / per100,
        "norm_narrow": sum(is_norm_span(s, narrow=True) for s in nontest) / per100,
        "test": (len(spans) - len(nontest)) / per100,
        # without the test-span exclusion: if the drop were an artefact of aware-arm sentences being
        # reclassified as test-talk (and so excluded), this variant would not fall
        "norm_incl_test": sum(is_norm_span(s) for s in spans) / per100,
    }
    # judge-free: sentence-split the model's own reasoning (no annotation involved at all); falls back to
    # the annotated text with markers stripped if the source is unavailable
    plain = source if source else SPAN_RE.sub(lambda m: m.group(2) + " ", annotated)
    sents = [s for s in SENT_RE.split(plain) if s.strip()]
    out["norm_sentences"] = sum(is_norm_span(s) and not is_test_span(s) for s in sents) / per100
    for w, rx in zip(WORD_GROUPS, WORD_RES):
        out[f"w:{w}"] = sum(bool(rx.search(s)) for s in nontest) / per100
    return out


def boot_ci(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    m = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def exec_str(v) -> str:
    return "True" if str(v) == "True" else "False" if str(v) == "False" else "unknown"


CLASSES = {
    "all": lambda b, a: True,
    "refuse_both": lambda b, a: b == "False" and a == "False",
    "comply_both": lambda b, a: b == "True" and a == "True",
    "gained": lambda b, a: b == "False" and a == "True",
    "lost": lambda b, a: b == "True" and a == "False",
}


def paired_rows(feats: dict, label: dict, base_key: str, arm_key: str, meta: dict, metrics: list[str]) -> list[dict]:
    rows = []
    for fr in ("real", "fake", "hypothetical"):
        items = sorted({i for (k, i, f) in feats if k == base_key and f == fr} & {i for (k, i, f) in feats if k == arm_key and f == fr})
        if not items:
            continue
        for cls, fn in CLASSES.items():
            sel = [i for i in items if fn(label[(base_key, i, fr)], label[(arm_key, i, fr)])]
            if len(sel) < 5:
                continue
            for k in metrics:
                d = np.array([feats[(arm_key, i, fr)][k] - feats[(base_key, i, fr)][k] for i in sel])
                base = float(np.mean([feats[(base_key, i, fr)][k] for i in sel]))
                m, lo, hi = boot_ci(d)
                rows.append({**meta, "framing": fr, "flip_class": cls, "n": len(sel), "metric": k, "base_mean": base, "delta": m, "ci_lo": lo, "ci_hi": hi, "sig": bool(lo > 0 or hi < 0)})
    return rows


def collect(traces: pd.DataFrame, ann_paths: dict[str, Path]) -> tuple[dict, dict]:
    """``feats[(arm, index, framing)]`` and ``label[...]`` (executed as a string) for every annotated trace."""
    feats, label = {}, {}
    ann = {arm: load_results(p) for arm, p in ann_paths.items() if p.exists()}
    for r in traces.itertuples(index=False):
        rec = ann.get(r.arm, {}).get(r.id)
        if not rec or not (rec.get("text") or "").strip():
            continue
        if str(getattr(r, "ok", True)) != "True" or not pd.notna(r.words) or float(r.words) < MIN_WORDS:
            continue  # judge-failed or near-empty reasoning: a density over a handful of words is meaningless
        key = (r.arm, int(r.index), r.framing)
        feats[key] = trace_measures(rec["text"], r.words if pd.notna(r.words) else 0, (rec.get("meta") or {}).get("reasoning"))
        label[key] = exec_str(r.executed)
    return feats, label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "norm_recall")
    ap.add_argument("--api-dirs", type=Path, nargs="*", default=[], help="results/api_prompt/<tag> directories whose arms have been annotated")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["norm", "norm_narrow", "norm_sentences", "norm_incl_test", "test"]
    rows, word_rows = [], []

    # --- Tier 1: the authors' shipped outputs (actions, and the sampled re-decode as the noise floor)
    t1 = pd.read_csv(REPO / "results" / "tier1" / "traces.csv", low_memory=False)
    t1 = t1[t1.family.isin(["actions", "actions_sampling"])]
    for model in sorted(t1.model.unique()):
        d = t1[t1.model == model].copy()
        d["arm"] = d.family + "/" + d.arm  # keep the two families apart
        d["id"] = d["id"]
        paths = {arm: REPO / "results" / "annotations" / model / arm.split("/")[0] / (arm.split("/")[1] + ".jsonl") for arm in d.arm.unique()}
        feats, label = collect(d, paths)
        alphas = d.groupby("arm").signed_alpha.first()
        for arm in sorted(d.arm.unique(), key=lambda a: (a.split("/")[0], alphas[a])):
            if arm == "actions/alpha0.0":
                continue
            meta = {"model": model, "family": arm.split("/")[0], "arm": arm.split("/")[1], "signed_alpha": float(alphas[arm]), "baseline": "actions/alpha0.0"}
            rows += paired_rows(feats, label, "actions/alpha0.0", arm, meta, metrics)
            if arm.startswith("actions_sampling/") and arm != "actions_sampling/alpha0.0" and "actions_sampling/alpha0.0" in {k[0] for k in feats}:
                # floor-matched pairing: sampled steered arm against the sampled alpha-0 decode
                rows += paired_rows(feats, label, "actions_sampling/alpha0.0", arm, {**meta, "baseline": "actions_sampling/alpha0.0"}, metrics)
            if arm in ("actions/alpha0.07_aware", "actions/alpha0.043_aware", "actions/alpha0.05_aware", "actions/alpha0.32", "actions/alpha0.3", "actions/alpha0.35"):
                wr = paired_rows(feats, label, "actions/alpha0.0", arm, meta, [f"w:{w}" for w in WORD_GROUPS])
                word_rows += [r for r in wr if r["flip_class"] == "all"]

    # --- Tier M: Qwen3-4B main run + F1 prompt arm
    tm = pd.read_csv(REPO / "results" / "qwen3_4b" / "steering_traces.csv", low_memory=False)
    f1p = REPO / "results" / "qwen3_4b" / "f1_prompt_steering_traces.csv"
    if f1p.exists():
        f1 = pd.read_csv(f1p, low_memory=False)
        tm = pd.concat([tm, f1[f1.arm == "prompt_test"]], ignore_index=True)
    sdir = REPO / "results" / "qwen3_4b" / "steering"
    paths = {arm: sdir / f"{arm}.json.annotations.jsonl" for arm in tm.arm.unique()}
    paths["prompt_test"] = REPO / "results" / "qwen3_4b" / "followups" / "f1_prompt" / "prompt_test.json.annotations.jsonl"
    feats, label = collect(tm, paths)
    for arm in [a for a in tm.arm.unique() if a != "baseline"]:
        meta = {"model": "qwen3_4b", "family": "steer_actions", "arm": arm, "signed_alpha": float(tm[tm.arm == arm].signed_alpha.iloc[0]), "baseline": "baseline"}
        rows += paired_rows(feats, label, "baseline", arm, meta, metrics)

    # --- optional: the 32B API prompt comparison, once annotated (traces csv written by analyze_tier_m)
    for adir in args.api_dirs:
        tp = next(iter(list(adir.glob("*_traces.csv")) + list(adir.parent.glob(f"{adir.name}_*traces.csv"))), None)
        if tp is None:
            print(f"{adir}: no *_traces.csv yet, skipped")
            continue
        ta = pd.read_csv(tp, low_memory=False)
        paths = {arm: adir / f"{arm}.json.annotations.jsonl" for arm in ta.arm.unique()}
        feats, label = collect(ta, paths)
        for arm in [a for a in ta.arm.unique() if a != "baseline"]:
            meta = {"model": ta.model.iloc[0], "family": ta.family.iloc[0], "arm": arm, "signed_alpha": 0.0, "baseline": "baseline"}
            rows += paired_rows(feats, label, "baseline", arm, meta, metrics)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "norm_recall.csv", index=False)
    pw = pd.DataFrame(word_rows)
    pw.to_csv(args.out_dir / "per_word.csv", index=False)

    def fmt(r) -> str:
        return f"{r.delta:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]{'*' if r.sig else ''}"

    lines = ["# Norm-recall density: paired Δ vs baseline (spans per 100 words citing rules / laws / harms, test spans excluded)", "",
             "Generated by `scripts/analyze_norm_recall.py`. Asterisk: bootstrap 95% CI excludes 0. `refuse_both` holds the decision fixed.", ""]
    for metric, title in (("norm", "Norm density (full lexicon)"), ("norm_sentences", "Judge-free variant (sentence split, no labels used)"), ("norm_narrow", "Narrow lexicon (rules and legality only)"), ("norm_incl_test", "Norm density without excluding test spans"), ("test", "Test-talk density (for reference)")):
        lines += [f"## {title}", "", "| model | family | arm | α | framing | n all | all | n refuse_both | refuse_both | n comply_both | comply_both | n gained | gained |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        sub = df[df.metric == metric]
        for (model, family, arm, alpha, fr), g in sub.groupby(["model", "family", "arm", "signed_alpha", "framing"], sort=False):
            cells = {r.flip_class: r for r in g.itertuples()}
            row = [model, family, arm, f"{alpha:+.3f}", fr]
            for cls in ("all", "refuse_both", "comply_both", "gained"):
                r = cells.get(cls)
                row += [str(r.n) if r else "—", fmt(r) if r else "—"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Per-word contributions (Δ density of non-test spans containing the word group; real framing, cleanest arms)", ""]
    if len(pw):
        sub = pw[pw.framing == "real"]
        words = [f"w:{w}" for w in WORD_GROUPS]
        lines += ["| model | arm | " + " | ".join(WORD_GROUPS) + " |", "|---|---|" + "---|" * len(WORD_GROUPS)]
        for (model, arm), g in sub.groupby(["model", "arm"], sort=False):
            cells = {r.metric: r for r in g.itertuples()}
            lines.append(f"| {model} | {arm} | " + " | ".join(f"{cells[w].delta:+.2f}{'*' if cells[w].sig else ''}" if w in cells else "—" for w in words) + " |")
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"... wrote {args.out_dir / 'SUMMARY.md'} ({len(df)} cells)")
    figure(args.out_dir / "norm_recall.csv", args.out_dir / "fig_norm_dose.png")


def figure(csv_path: Path, out: Path) -> None:
    """Dose–response of norm density on the shipped 32B outputs: paired Δ vs signed α, per model, real and
    fake framing, all items and refuse_both. (The Qwen3-4B and API arms are in the tables, not the figure.)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df = df[(df.metric == "norm") & (df.family == "actions")]
    models = ["deepseek_qwen", "qwen3", "qwq"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for col, model in enumerate(models):
        for row, cls in enumerate(["all", "refuse_both"]):
            ax = axes[row, col]
            for fr, color in (("real", "C0"), ("fake", "C1")):
                d = df[(df.model == model) & (df.flip_class == cls) & (df.framing == fr)].sort_values("signed_alpha")
                ax.errorbar(d.signed_alpha, d.delta, yerr=[d.delta - d.ci_lo, d.ci_hi - d.delta], fmt="o-", color=color, label=fr, capsize=3)
            ax.axhline(0, color="k", lw=0.8)
            ax.axvline(0, color="k", lw=0.4, ls=":")
            ax.set_title(f"{model} — {cls}")
            ax.set_xlabel("signed α (aware > 0, unaware < 0)")
            if col == 0:
                ax.set_ylabel("Δ norm-citing spans / 100 words")
            ax.legend(fontsize=8)
    fig.suptitle("Norm-recall density under test-awareness steering (authors' shipped 32B outputs; paired vs α = 0, 95% bootstrap CIs)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
