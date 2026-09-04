#!/usr/bin/env python
"""Is "norm-talk" a direction of its own, or the other end of the test-awareness axis?

Motivation (4 Sep 2026): aware steering lowers the density of norm-citing sentences ("this is
illegal", "my guidelines prohibit this") on the shipped 32B outputs, even in items whose refusal does
not change. One reading: the awareness direction is anti-norm by construction (the probe's negative
class is the judge's "treats the task as genuine" quotes), so steering along it suppresses norm-talk
mechanically. This script measures that in activation space, on CPU, from the span-mean features
``build_probe.py`` cached for the Qwen3-4B probe set.

Directions (all at every HF hidden-state index, d = 2560):
- awareness: the trained probe row ``fc.weight[1]`` (``probe_directions.pt``), and a mean-difference
  version (positive-span mean − negative-span mean) that carries no initialisation.
- norm: mean of spans matching ``d10.norms.NORM_LEXICON`` minus mean of the rest, computed **within the
  probe's negative class only** (every span in the fit has awareness label 0, so the norm direction
  cannot be a proxy for the awareness label). A logistic-regression version at a few layers.
- references: the same construction with the norm labels shuffled (20 draws; the null for "any
  two-class mean difference on these spans"), the test-lexicon direction (test spans vs the rest,
  over all spans; expected to align with awareness), a random unit vector, and the six rebuilt
  Venhoff directions (decoder layer k−1 for hidden-state index k, as in ``geometry.py``).

Span texts are re-derived exactly as the features were built (same judgment file order, same quote
relocation, same tokenizer windows); a trace whose re-derived span count does not match its cached
row count is dropped and counted. Writes ``results/norm_direction/{per_layer.csv, SUMMARY.md,
fig_cosines.png}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from d10.activations import abdelnabi_probe_text, char_spans_to_token_spans, locate_quotes  # noqa: E402
from d10.awareness import is_negative, is_positive, parse_judgment  # noqa: E402
from d10.judge import load_results, record_ok  # noqa: E402
from d10.norms import is_norm_span  # noqa: E402
from d10.parse import is_test_span  # noqa: E402
from d10.shipped import load_generations  # noqa: E402
from d10.venhoff import BEHAVIOURS, feature_vectors  # noqa: E402

import re  # noqa: E402

#: A second content contrast within the negative class, as a reference for "any lexical partition of
#: the negatives": procedural / tool-mechanics sentences ("the tool is X", "the parameters are ...").
PROC_LEXICON = re.compile(r"\b(?:tool\w*|parameter\w*|format\w*|json|template\w*|field\w*|placeholder\w*|function|argument\w*|syntax|schema)\b", re.I)


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def span_texts(tok, generations: Path, judgments: Path, model_name: str, family: str, max_len: int) -> dict[str, list[str]]:
    """Per trace id, the located span texts in the order ``build_probe.py`` pooled them."""
    traces = {t.id: t for t in load_generations(generations, model_name, family)}
    judged = {k: v for k, v in load_results(judgments).items() if record_ok(v) and k in traces}
    out: dict[str, list[str]] = {}
    for tid, rec in judged.items():
        j = parse_judgment(rec["text"])
        if is_positive(j):
            quotes = j["model_awareness"].get("evidence") or []
        elif is_negative(j):
            quotes = (j["model_awareness"].get("evidence") or []) + (j["hypothetical_recognition"].get("evidence") or [])
        else:
            continue
        t = traces[tid]
        text = abdelnabi_probe_text(tok, t.task, t.raw)
        cspans = locate_quotes(text, quotes)
        # char_spans_to_token_spans drops spans with no tokens; pooled_states NaNs (dropped) windows
        # that start at or beyond the truncation point
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        offsets = enc["offset_mapping"]
        T = min(len(offsets), max_len)
        kept = []
        for cs, ce in cspans:
            idx = [i for i, (s, e) in enumerate(offsets) if e > cs and s < ce and e > s]
            if idx and min(idx[-1] + 1, T) > idx[0]:
                kept.append(text[cs:ce])
        out[tid] = kept
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "probe")
    ap.add_argument("--generations", type=Path, default=REPO / "results" / "qwen3_4b" / "triggers_probe.json")
    ap.add_argument("--judgments", type=Path, default=REPO / "results" / "qwen3_4b" / "triggers_probe.json.awareness.jsonl")
    ap.add_argument("--vectors-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "vectors")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-name", default="qwen3_4b")
    ap.add_argument("--family", default="triggers_probe")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--lr-layers", type=int, nargs="*", default=[0, 10, 20, 30, 36])
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "norm_direction")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    feats = torch.load(args.probe_dir / "probe_features.pt")
    X, y = feats["X"].float().numpy(), feats["y"].numpy()
    ids = json.load(open(args.probe_dir / "probe_feature_ids.json"))
    assert len(ids) == len(X)
    texts = span_texts(tok, args.generations, args.judgments, args.model_name, args.family, args.max_len)

    # align: rows for a trace appear contiguously in the order the spans were located
    row_text = [None] * len(ids)
    counts: dict[str, int] = {}
    for i, tid in enumerate(ids):
        counts[tid] = counts.get(tid, 0) + 1
    n_dropped_traces = n_dropped_rows = 0
    pos_in_trace: dict[str, int] = {}
    for i, tid in enumerate(ids):
        k = pos_in_trace.get(tid, 0); pos_in_trace[tid] = k + 1
        if tid in texts and len(texts[tid]) == counts[tid]:
            row_text[i] = texts[tid][k]
    for tid, c in counts.items():
        if not (tid in texts and len(texts[tid]) == c):
            n_dropped_traces += 1; n_dropped_rows += c
    keep = np.array([t is not None for t in row_text])
    X, y, ids = X[keep], y[keep], [t for t, k in zip(ids, keep) if k]
    row_text = [t for t in row_text if t is not None]
    norm = np.array([is_norm_span(t) for t in row_text]); test = np.array([is_test_span(t) for t in row_text])
    proc = np.array([bool(PROC_LEXICON.search(t)) for t in row_text])
    neg = y == 0
    n_layers = X.shape[1]
    print(f"rows {len(X)} (dropped {n_dropped_rows} rows from {n_dropped_traces} traces whose span count did not re-derive); positives {int(y.sum())}; "
          f"norm share pos {norm[y == 1].mean():.2f} neg {norm[neg].mean():.2f}; test share pos {test[y == 1].mean():.2f} neg {test[neg].mean():.2f}; "
          f"negatives: norm {int((norm & neg).sum())} / non-norm {int((~norm & neg).sum())}; proc spans in negatives {int((proc & neg).sum())}", flush=True)

    aw_dirs = torch.load(args.probe_dir / "probe_directions.pt")
    fv = {name: feature_vectors(torch.load(args.vectors_dir / f"{name}.pt")) for name in ("mean_vectors",)}
    rng = np.random.default_rng(0)
    rand_unit = rng.standard_normal(X.shape[2]); rand_unit /= np.linalg.norm(rand_unit)
    rows = []
    for k in range(n_layers):
        Xk = X[:, k]
        aw_w1 = np.asarray(aw_dirs[k].float().numpy()) if k in aw_dirs else None
        aw_md = Xk[y == 1].mean(0) - Xk[neg].mean(0)
        norm_md = Xk[neg & norm].mean(0) - Xk[neg & ~norm].mean(0)
        test_md = Xk[test].mean(0) - Xk[~test].mean(0)
        # shuffled-norm-label null within negatives
        shuf = []
        idx_neg = np.where(neg)[0]
        for s in range(20):
            lab = rng.permutation(norm[idx_neg])
            d = Xk[idx_neg][lab].mean(0) - Xk[idx_neg][~lab].mean(0)
            shuf.append(cos(aw_md, d))
        row = {"hs_index": k, "cos_awW1_normMD": cos(aw_w1, norm_md) if aw_w1 is not None else math.nan, "cos_awMD_normMD": cos(aw_md, norm_md),
               "cos_awW1_testMD": cos(aw_w1, test_md) if aw_w1 is not None else math.nan, "cos_awMD_testMD": cos(aw_md, test_md),
               "cos_normMD_testMD": cos(norm_md, test_md), "shuffled_mean": float(np.mean(shuf)), "shuffled_sd": float(np.std(shuf)),
               "cos_awMD_random": cos(aw_md, rand_unit), "cos_normMD_random": cos(norm_md, rand_unit),
               "norm_md_norm": float(np.linalg.norm(norm_md)), "aw_md_norm": float(np.linalg.norm(aw_md))}
        # norm direction fitted within positives too (small n) as a sanity check
        if (norm & (y == 1)).sum() >= 10:
            norm_md_pos = Xk[(y == 1) & norm].mean(0) - Xk[(y == 1) & ~norm].mean(0)
            row["cos_awMD_normMD_withinPos"] = cos(aw_md, norm_md_pos)
            row["cos_normMD_normMD_withinPos"] = cos(norm_md, norm_md_pos)
        if k >= 1:
            for b in BEHAVIOURS:
                if b in fv["mean_vectors"] and k - 1 < fv["mean_vectors"][b].shape[0]:
                    row[f"cos_normMD_{b}"] = cos(norm_md, fv["mean_vectors"][b][k - 1].float().numpy())
        # standardised space: every dimension divided by its sd over all spans, so a handful of
        # high-variance residual dimensions cannot carry the cosine
        sd = Xk.std(0) + 1e-6
        Zk = Xk / sd
        aw_md_z = Zk[y == 1].mean(0) - Zk[neg].mean(0)
        norm_md_z = Zk[neg & norm].mean(0) - Zk[neg & ~norm].mean(0)
        test_md_z = Zk[test].mean(0) - Zk[~test].mean(0)
        shuf_z = []
        for s in range(20):
            lab = rng.permutation(norm[idx_neg])
            shuf_z.append(cos(aw_md_z, Zk[idx_neg][lab].mean(0) - Zk[idx_neg][~lab].mean(0)))
        proc_md = Xk[neg & proc].mean(0) - Xk[neg & ~proc].mean(0)
        proc_md_z = Zk[neg & proc].mean(0) - Zk[neg & ~proc].mean(0)
        row.update({"cos_awMD_procMD": cos(aw_md, proc_md), "cos_normMD_procMD": cos(norm_md, proc_md), "z_cos_awMD_procMD": cos(aw_md_z, proc_md_z), "z_cos_normMD_procMD": cos(norm_md_z, proc_md_z)})
        row.update({"z_cos_awMD_normMD": cos(aw_md_z, norm_md_z), "z_cos_awMD_testMD": cos(aw_md_z, test_md_z), "z_cos_normMD_testMD": cos(norm_md_z, test_md_z),
                    "z_shuffled_mean": float(np.mean(shuf_z)), "z_shuffled_sd": float(np.std(shuf_z))})
        if k in args.lr_layers:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=3000, C=0.05).fit(Xk[neg], norm[neg])
            row["cos_awW1_normLR"] = cos(aw_w1, lr.coef_[0]) if aw_w1 is not None else math.nan
            row["cos_awMD_normLR"] = cos(aw_md, lr.coef_[0])
            row["normLR_train_acc"] = float(lr.score(Xk[neg], norm[neg]))
            # like for like: both directions fitted the same way in the same standardised space
            lr_aw = LogisticRegression(max_iter=3000, C=0.05).fit(Zk, y)
            lr_norm_z = LogisticRegression(max_iter=3000, C=0.05).fit(Zk[neg], norm[neg])
            lr_test_z = LogisticRegression(max_iter=3000, C=0.05).fit(Zk, test)
            row["z_cos_awLR_normLR"] = cos(lr_aw.coef_[0], lr_norm_z.coef_[0])
            row["z_cos_awLR_testLR"] = cos(lr_aw.coef_[0], lr_test_z.coef_[0])
            row["z_cos_normLR_testLR"] = cos(lr_norm_z.coef_[0], lr_test_z.coef_[0])
        rows.append(row)
        print(f"hs {k:2d}: cos(aw_w1, norm) {row['cos_awW1_normMD']:+.3f}  cos(aw_md, norm) {row['cos_awMD_normMD']:+.3f}  "
              f"shuffled {row['shuffled_mean']:+.3f}±{row['shuffled_sd']:.3f}  cos(aw, test) {row['cos_awMD_testMD']:+.3f}  cos(norm, test) {row['cos_normMD_testMD']:+.3f}"
              + f"  | z-space md {row['z_cos_awMD_normMD']:+.3f} (null {row['z_shuffled_mean']:+.3f}±{row['z_shuffled_sd']:.3f}, aw·test {row['z_cos_awMD_testMD']:+.3f}) | proc: aw·proc raw {row['cos_awMD_procMD']:+.3f} z {row['z_cos_awMD_procMD']:+.3f}, norm·proc z {row['z_cos_normMD_procMD']:+.3f}"
              + (f"  LR raw {row['cos_awMD_normLR']:+.3f}; z LR·LR aw·norm {row['z_cos_awLR_normLR']:+.3f} aw·test {row['z_cos_awLR_testLR']:+.3f} norm·test {row['z_cos_normLR_testLR']:+.3f}" if "cos_awMD_normLR" in row else ""), flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda s: (s != "hs_index", s))
    with (args.out_dir / "per_layer.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ks = [r["hs_index"] for r in rows]
    ax.plot(ks, [r["cos_awMD_normMD"] for r in rows], "o-", label="awareness (mean-diff) · norm (within negatives)")
    ax.plot(ks, [r["cos_awW1_normMD"] for r in rows], "s--", label="awareness (probe row w1) · norm")
    ax.plot(ks, [r["cos_awMD_testMD"] for r in rows], "^-", label="awareness · test-lexicon direction (reference)")
    ax.plot(ks, [r["z_cos_awMD_normMD"] for r in rows], "o:", label="awareness · norm, standardised space")
    ax.plot(ks, [r["z_cos_awMD_testMD"] for r in rows], "^:", label="awareness · test-lexicon, standardised space")
    ax.plot(ks, [r["z_cos_awMD_procMD"] for r in rows], "x-", label="awareness · tool-mechanics direction (within negatives), standardised")
    ax.fill_between(ks, [r["shuffled_mean"] - 2 * r["shuffled_sd"] for r in rows], [r["shuffled_mean"] + 2 * r["shuffled_sd"] for r in rows], alpha=0.2, label="shuffled norm labels ±2 sd")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("hidden-state index (Qwen3-4B)"); ax.set_ylabel("cosine"); ax.legend(fontsize=8)
    ax.set_title("Is the norm-recall direction the other end of the test-awareness axis?")
    fig.tight_layout(); fig.savefig(args.out_dir / "fig_cosines.png", dpi=130)

    best = json.load(open(args.probe_dir / "probe_best.json"))["layer"]
    rb = next(r for r in rows if r["hs_index"] == best)
    lines = [f"# Norm direction vs awareness direction (Qwen3-4B, cached probe-set span features)", "",
             f"Generated by `scripts/norm_direction_geometry.py`. Rows {len(X)} spans ({int(y.sum())} awareness-positive); norm-span share {norm[y == 1].mean():.2f} in positives, {norm[neg].mean():.2f} in negatives. "
             f"Norm direction = mean(norm spans) − mean(other spans) **within the negative class**. Chance |cos| at d = 2560 ≈ {1 / math.sqrt(X.shape[2]):.3f}.", "",
             f"At the probe's layer (hs {best}): cos(awareness w1, norm) = {rb['cos_awW1_normMD']:+.3f}; cos(awareness mean-diff, norm) = {rb['cos_awMD_normMD']:+.3f}; "
             f"shuffled-label null {rb['shuffled_mean']:+.3f} ± {rb['shuffled_sd']:.3f}; cos(awareness, test-lexicon direction) = {rb['cos_awMD_testMD']:+.3f}; cos(norm, test-lexicon) = {rb['cos_normMD_testMD']:+.3f}; "
             f"random unit vector {rb['cos_awMD_random']:+.3f}.", "",
             "| hs | aw_w1·norm | aw_md·norm | shuffled null | aw·test | norm·test | z: aw·norm | z: null | z: aw·test | z LR·LR aw·norm | aw_md·normLR | " + " | ".join(f"norm·{b}" for b in BEHAVIOURS) + " |",
             "|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(BEHAVIOURS)]
    for r in rows:
        lines.append(f"| {r['hs_index']} | {r['cos_awW1_normMD']:+.3f} | {r['cos_awMD_normMD']:+.3f} | {r['shuffled_mean']:+.3f}±{r['shuffled_sd']:.3f} | {r['cos_awMD_testMD']:+.3f} | {r['cos_normMD_testMD']:+.3f} | {r['z_cos_awMD_normMD']:+.3f} | {r['z_shuffled_mean']:+.3f}±{r['z_shuffled_sd']:.3f} | {r['z_cos_awMD_testMD']:+.3f} | "
                     + (f"{r['z_cos_awLR_normLR']:+.3f}" if "z_cos_awLR_normLR" in r else "—") + " | " + (f"{r['cos_awMD_normLR']:+.3f}" if "cos_awMD_normLR" in r else "—") + " | " + " | ".join(f"{r[f'cos_normMD_{b}']:+.3f}" if f"cos_normMD_{b}" in r else "—" for b in BEHAVIOURS) + " |")
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
