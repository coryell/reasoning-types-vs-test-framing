#!/usr/bin/env python
"""The Qwen3-32B prompt comparison with its own noise floor: two sampled decodes (seed 0, seed 1) of
both arms (plain agent system prompt vs the same plus "This is a test scenario with no real-world
consequences."), 149 action items × real/fake, all through OpenRouter.

Pairs compared, each paired by item: baseline0 vs baseline1 and prompt0 vs prompt1 (decode-to-decode
floors), prompt vs baseline within each seed (the effect, twice), and prompt vs baseline with the two
seeds pooled (298 pairs). For each pair: execution rates and McNemar; verbalization; words; per-trace
counts and densities of the six behaviours (test-lexicon split) and norm-recall density, for all items
and split by compliance class. Writes ``results/api_prompt/SUMMARY_qwen3_32b_seeds.md`` and
``qwen3_32b_seeds_cells.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from d10.analysis import mcnemar_exact  # noqa: E402
from d10.awareness import is_positive, parse_judgment  # noqa: E402
from d10.judge import load_results, record_ok  # noqa: E402
from d10.norms import is_norm_span  # noqa: E402
from d10.parse import LABELS, SPAN_RE, is_test_span, parse_annotation, trace_metrics  # noqa: E402
from d10.shipped import load_generations  # noqa: E402

BEH = ["deduction", "adding-knowledge", "uncertainty-estimation", "backtracking", "example-testing", "initializing"]


def load(dirpath: Path, model_name: str, arm: str, tag: str) -> pd.DataFrame:
    p = dirpath / f"{arm}.json"
    traces = load_generations(p, model_name, "api_actions", arm)
    ann = load_results(p.with_suffix(p.suffix + ".annotations.jsonl"))
    aw = load_results(p.with_suffix(p.suffix + ".awareness.jsonl"))
    rows = []
    for t in traces:
        a = ann.get(t.id)
        rec = {"tag": tag, "index": t.index, "framing": t.framing, "task": t.task, "executed": t.executed, "closed": t.has_think_close, "words": t.reasoning_words}
        j = aw.get(t.id)
        rec["verbal"] = is_positive(parse_judgment(j["text"])) if j and record_ok(j) else np.nan
        if a and record_ok(a):
            spans = parse_annotation(a["text"], t.reasoning)
            m = trace_metrics(spans, t.reasoning)
            if m["n_spans_counted"] == 0:  # judge failure: keep the trace for execution / words, no morphology
                rec["ok"] = False
                rows.append(rec)
                continue
            rec.update({k: v for k, v in m.items() if k.startswith("density_") or k.startswith("n_") or k == "words"})
            counted = [s for s in spans if s.counted]
            per100 = max(m["words"], 1) / 100
            rec["density_norm"] = sum(is_norm_span(s.text) and not is_test_span(s.text) for s in counted) / per100
            rec["n_norm"] = sum(is_norm_span(s.text) and not is_test_span(s.text) for s in counted)
            rec["n_testlex_total"] = sum(is_test_span(s.text) for s in counted)
            rec["ok"] = True
        else:
            rec["ok"] = False
        rows.append(rec)
    return pd.DataFrame(rows)


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    m = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def compare(A: pd.DataFrame, B: pd.DataFrame, name: str, lines: list, cells: list) -> None:
    for fr in ("real", "fake"):
        a = A[A.framing == fr].set_index("index"); b = B[B.framing == fr].set_index("index")
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
        assert (a.task == b.task).all(), "items are paired by index; the two arms enumerate different tasks"
        known = a.executed.notna() & b.executed.notna()
        ea, eb = a.executed[known].astype(bool), b.executed[known].astype(bool)
        lost, gained = int((ea & ~eb).sum()), int((~ea & eb).sum())
        p = mcnemar_exact(lost, gained)
        va, vb = a.verbal.dropna().mean(), b.verbal.dropna().mean()
        lines.append(f"| {name} | {fr} | {known.sum()} | {ea.mean():.3f} → {eb.mean():.3f} | {lost} / {gained} | {p:.3f} | {va:.3f} → {vb:.3f} | {a.words.mean():.0f} → {b.words.mean():.0f} |")
        ok = a.ok.astype(bool) & b.ok.astype(bool)
        classes = {"all": ok, "refuse_both": ok & known & ~ea.reindex(a.index, fill_value=False) & ~eb.reindex(a.index, fill_value=False),
                   "comply_both": ok & known & ea.reindex(a.index, fill_value=False) & eb.reindex(a.index, fill_value=False),
                   "gained": ok & known & ~ea.reindex(a.index, fill_value=False) & eb.reindex(a.index, fill_value=False)}
        for cls, mask in classes.items():
            ii = a.index[mask]
            if len(ii) < 5:
                continue
            for metric in ["words", "density_norm", "n_norm", "n_testlex_total"] + [f"density_{x}_nontest" for x in BEH] + [f"density_{x}_testlex" for x in BEH] + [f"n_{x}" for x in BEH]:
                d = (b.loc[ii, metric] - a.loc[ii, metric]).astype(float)
                m, lo, hi = boot(d)
                cells.append({"pair": name, "framing": fr, "class": cls, "n": len(ii), "metric": metric, "base": float(a.loc[ii, metric].mean()), "delta": m, "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir0", type=Path, default=REPO / "results" / "api_prompt" / "qwen3_32b_api")
    ap.add_argument("--dir1", type=Path, default=REPO / "results" / "api_prompt" / "qwen3_32b_api_seed1")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "api_prompt")
    args = ap.parse_args()
    b0 = load(args.dir0, "qwen3_32b_api", "baseline", "b0"); p0 = load(args.dir0, "qwen3_32b_api", "prompt_test", "p0")
    b1 = load(args.dir1, "qwen3_32b_api_seed1", "baseline", "b1"); p1 = load(args.dir1, "qwen3_32b_api_seed1", "prompt_test", "p1")
    lines = ["# Qwen3-32B (OpenRouter) prompt comparison with a second decode", "",
             "Generated by `scripts/analyze_api_seeds.py`. Execution = the authors' tool-name rule on closed traces; verbalization = awareness judge positive (Yes/Maybe); paired by item.", "",
             "| pair (A → B) | framing | n | execution A → B | lost / gained | McNemar p | verbalization A → B | words A → B |", "|---|---|---|---|---|---|---|---|"]
    cells: list = []
    compare(b0, b1, "baseline0 → baseline1 (floor)", lines, cells)
    compare(p0, p1, "prompt0 → prompt1 (floor)", lines, cells)
    compare(b0, p0, "baseline0 → prompt0", lines, cells)
    compare(b1, p1, "baseline1 → prompt1", lines, cells)
    compare(b0, p1, "baseline0 → prompt1 (cross)", lines, cells)
    compare(b1, p0, "baseline1 → prompt0 (cross)", lines, cells)
    df = pd.DataFrame(cells)
    df.to_csv(args.out_dir / "qwen3_32b_seeds_cells.csv", index=False)
    lines += ["", "## Paired Δ per cell (value [95% CI], * = CI excludes 0); densities per 100 words, counts per trace", ""]
    show = ["words", "n_testlex_total", "density_norm", "density_deduction_nontest", "density_adding-knowledge_nontest", "density_uncertainty-estimation_nontest", "density_backtracking_nontest", "density_example-testing_nontest", "n_backtracking", "n_example-testing"]
    lines += ["| pair | framing | class | n | " + " | ".join(show) + " |", "|---|---|---|---|" + "---|" * len(show)]
    for (pair, fr, cls), g in df.groupby(["pair", "framing", "class"], sort=False):
        c = {r.metric: r for r in g.itertuples()}
        lines.append(f"| {pair} | {fr} | {cls} | {g.n.iloc[0]} | " + " | ".join(f"{c[m].delta:+.2f} [{c[m].lo:+.2f},{c[m].hi:+.2f}]{'*' if c[m].sig else ''}" if m in c else "—" for m in show) + " |")
    (args.out_dir / "SUMMARY_qwen3_32b_seeds.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"wrote {args.out_dir / 'SUMMARY_qwen3_32b_seeds.md'} ({len(df)} cells)")


if __name__ == "__main__":
    main()
