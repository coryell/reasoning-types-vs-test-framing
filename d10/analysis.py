"""Tier 1 analysis: loading annotated traces, statistics, tables, figures.

The CLI in ``scripts/analyze_tier1.py`` calls these in order. Definitions follow EXECUTION_PLAN_2
§2.2–2.3. Bootstrap CIs are percentile intervals over 2,000 resamples (seed 0) of traces (arm means)
or of item-level differences (paired contrasts). McNemar is the exact binomial test on the discordant
pairs. Nothing is corrected for multiple comparisons.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from . import parse  # noqa: E402
from .judge import load_results, record_ok  # noqa: E402
from .parse import LABELS  # noqa: E402

#: Metric columns every row must carry (NaN when the annotation was unusable), so group-bys work on
#: partial or entirely failed data.
METRIC_COLUMNS = (
    ["words", "n_spans", "n_spans_counted", "n_spans_unknown_label", "n_spans_not_found_exact",
     "n_spans_found_norm_only", "n_spans_not_found", "labeled_words", "labeled_fraction", "repetition_4gram"]
    + [f"{m}_{b}" for m in ("n", "w", "density", "coverage", "any") for b in LABELS]
    + [f"{m}_{b}_{s}" for m, s in (("n", "testlex"), ("density", "testlex"), ("density", "nontest")) for b in LABELS]
    + [f"lex_{k}_{s}" for k in ("wait", "hmm", "backtrack_lex", "uncertainty_lex") for s in ("n", "per100w")]
)

BASELINE = "alpha0.0"
N_BOOT = 2000
SEED = 0
MIN_N = 5  # do not report a mean over fewer traces than this

# Validated default categorical palette (dataviz skill, references/palette.md), fixed slot order.
COLORS = {"real": "#2a78d6", "fake": "#eb6834", "hypothetical": "#eb6834", "single": "#1baf7a"}
CLASS_COLORS = {"gained": "#2a78d6", "lost": "#eb6834", "same": "#1baf7a"}
SHORT = {
    "initializing": "initializing",
    "deduction": "deduction",
    "adding-knowledge": "adding-knowl.",
    "example-testing": "example-test.",
    "uncertainty-estimation": "uncertainty",
    "backtracking": "backtracking",
}


# ----------------------------------------------------------------------------- loading

def judge_conclusion(judge: dict | None, key: str) -> str | None:
    if not judge:
        return None
    v = judge.get(key)
    if isinstance(v, dict):
        v = v.get("conclusion")
    return v if v in ("Yes", "Maybe", "No") else None


def load_traces(ann_dir: Path, models: list[str] | None, families: list[str] | None) -> pd.DataFrame:
    rows = []
    for path in sorted(ann_dir.glob("*/*/*.jsonl")):
        model, family = path.parts[-3], path.parts[-2]
        if models and model not in models:
            continue
        if families and family not in families:
            continue
        for rec in load_results(path).values():
            m = rec["meta"]
            annotated = bool((rec.get("text") or "").strip())
            usable = record_ok(rec)
            row = {
                "id": rec["id"],
                "model": m["model"],
                "family": m["family"],
                "arm": m["arm"],
                "alpha": m["alpha"],
                "aware": m["aware"],
                "signed_alpha": m["signed_alpha"],
                "index": m["index"],
                "framing": m["framing"] or "single",
                "has_think_close": m["has_think_close"],
                "executed": m["executed"],
                "aware_judged": judge_conclusion(m.get("judge"), "steered_model_awareness"),
                "recog_judged": judge_conclusion(m.get("judge"), "steered_recognition"),
                "task_judged": judge_conclusion(m.get("judge"), "steered_task_performance"),
                "annotated": annotated,
                "finish_reason": rec.get("finish_reason"),
                "error": rec.get("error"),
                "served_model": rec.get("served_model"),
            }
            # Validation set for the test-language pattern: the shipped judge's awareness evidence quotes
            # on traces it judged aware. Recall = caught / total, reported in the summary.
            ev = []
            j = m.get("judge") or {}
            aw = j.get("steered_model_awareness") if isinstance(j, dict) else None
            if isinstance(aw, dict) and aw.get("conclusion") in ("Yes", "Maybe"):
                ev = [q for q in (aw.get("evidence") or []) if isinstance(q, str) and len(q.split()) >= 3]
            row["aw_evidence_n"] = len(ev)
            row["aw_evidence_caught"] = sum(parse.is_test_span(q) for q in ev)
            ok = False
            if usable:
                spans = parse.parse_annotation(rec["text"], m["reasoning"])
                row.update(parse.trace_metrics(spans, m["reasoning"]))
                row.update(parse.lexical_proxy(m["reasoning"]))
                # zero counted spans is a judge failure (refusal, rewrite, marker style the regex
                # misses), not a trace without behaviours
                ok = row["n_spans_counted"] > 0
            row["ok"] = ok
            # a clean response with nothing parseable; truncations and API errors are counted separately
            row["judge_fail"] = usable and not ok
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        for c in METRIC_COLUMNS:
            if c not in df:
                df[c] = np.nan
        df["unknown_labels"] = df.get("unknown_labels", pd.Series([None] * len(df))).apply(
            lambda v: ";".join(v) if isinstance(v, list) else v
        )
    return df


# ----------------------------------------------------------------------------- statistics

def boot_ci(x, seed: int = SEED) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def paired_tests(d) -> tuple[float, float]:
    """Paired t and Wilcoxon p-values on item-level differences; NaN when undefined."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, d[0]):
        return (math.nan, math.nan)
    t_p = float(stats.ttest_1samp(d, 0.0).pvalue)
    nz = d[d != 0]
    w_p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else math.nan
    return (t_p, w_p)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts b (lost) and c (gained)."""
    n = b + c
    if n == 0:
        return math.nan
    return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


# ----------------------------------------------------------------------------- tables

GROUP = ["model", "family", "framing"]
ARM = GROUP + ["arm", "signed_alpha"]


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "family", "arm", "signed_alpha"], sort=True)
    out = g.agg(
        n=("id", "size"),
        n_annotated=("annotated", "sum"),
        n_ok=("ok", "sum"),
        n_judge_fail=("judge_fail", "sum"),
        n_error=("error", lambda s: int(s.notna().sum())),
        n_truncated=("finish_reason", lambda s: int((s == "length").sum())),
        n_no_think_close=("has_think_close", lambda s: int((~s.astype(bool)).sum())),
        mean_words=("words", "mean"),
        mean_spans=("n_spans_counted", "mean"),
        spans_total=("n_spans", "sum"),
        spans_not_found_exact=("n_spans_not_found_exact", "sum"),
        spans_found_norm_only=("n_spans_found_norm_only", "sum"),
        spans_not_found=("n_spans_not_found", "sum"),
        spans_unknown_label=("n_spans_unknown_label", "sum"),
        mean_labeled_fraction=("labeled_fraction", "mean"),
        mean_repetition=("repetition_4gram", "mean"),
    ).reset_index()
    denom = out["spans_total"].replace(0, np.nan)
    out["not_found_exact_rate"] = out["spans_not_found_exact"] / denom
    out["found_norm_only_rate"] = out["spans_found_norm_only"] / denom
    out["not_found_rate"] = out["spans_not_found"] / denom
    out["unknown_label_rate"] = out["spans_unknown_label"] / denom
    return out


def arm_means(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = df[df.ok]
    for keys, sub in ok.groupby(ARM, sort=True):
        base = dict(zip(ARM, keys))
        base["n"] = len(sub)
        for metric in ("density", "coverage", "any"):
            for b in LABELS:
                col = f"{metric}_{b}"
                vals = sub[col].to_numpy(dtype=float)
                lo, hi = boot_ci(vals)
                rows.append({**base, "metric": metric, "behaviour": b, "mean": float(np.nanmean(vals)) if len(vals) else math.nan, "ci_lo": lo, "ci_hi": hi})
        for col in ("words", "n_spans_counted", "repetition_4gram", "executed"):
            vals = sub[col].astype(float).to_numpy() if col in sub else np.array([])
            if col == "executed" and sub[col].isna().all():
                continue
            lo, hi = boot_ci(vals)
            rows.append({**base, "metric": col, "behaviour": "", "mean": float(np.nanmean(vals)) if len(vals) else math.nan, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def paired_contrasts(df: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Each non-baseline arm vs ``baseline`` within (model, family, framing), paired by item index."""
    rows = []
    ok = df[df.ok]
    cols = (
        [f"{m}_{b}" for m in ("density", "coverage", "any") for b in LABELS]
        + [f"density_{b}_{s}" for s in ("testlex", "nontest") for b in LABELS]
        + ["words", "n_spans_counted"]
    )
    for keys, sub in ok.groupby(GROUP, sort=True):
        base = sub[sub.arm == baseline].set_index("index")
        if base.empty:
            continue
        for (arm, sa), arm_df in sub[sub.arm != baseline].groupby(["arm", "signed_alpha"]):
            arm_df = arm_df.set_index("index")
            common = base.index.intersection(arm_df.index)
            if len(common) < MIN_N:
                continue
            for col in cols:
                if col not in arm_df or col not in base:
                    print(f"paired_contrasts: column {col} missing for {keys}/{arm}; skipping")
                    continue
                d = arm_df.loc[common, col].to_numpy(dtype=float) - base.loc[common, col].to_numpy(dtype=float)
                lo, hi = boot_ci(d)
                t_p, w_p = paired_tests(d)
                if col in ("words", "n_spans_counted"):
                    metric, behaviour = col, ""
                elif col.endswith("_testlex") or col.endswith("_nontest"):
                    # density_<behaviour>_testlex → metric "density_testlex", behaviour <behaviour>
                    head, _, suffix = col.rpartition("_")
                    metric, _, behaviour = head.partition("_")
                    metric = f"{metric}_{suffix}"
                else:
                    metric, _, behaviour = col.partition("_")
                rows.append(
                    {
                        **dict(zip(GROUP, keys)),
                        "arm": arm,
                        "signed_alpha": sa,
                        "n_pairs": len(common),
                        "metric": metric,
                        "behaviour": behaviour,
                        "base_mean": float(np.nanmean(base.loc[common, col].to_numpy(dtype=float))),
                        "arm_mean": float(np.nanmean(arm_df.loc[common, col].to_numpy(dtype=float))),
                        "delta": float(np.nanmean(d)),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p_paired_t": t_p,
                        "p_wilcoxon": w_p,
                    }
                )
    return pd.DataFrame(rows)


def sign_symmetry(contrasts: pd.DataFrame) -> pd.DataFrame:
    """For each behaviour: Δdensity at the largest aware alpha and at the largest unaware alpha."""
    if contrasts.empty:
        return pd.DataFrame()
    c = contrasts[contrasts.metric == "density"]
    rows = []
    for keys, sub in c.groupby(GROUP + ["behaviour"], sort=True):
        aware = sub[sub.signed_alpha > 0]
        unaware = sub[sub.signed_alpha < 0]
        if aware.empty or unaware.empty:
            continue
        a = aware.loc[aware.signed_alpha.idxmax()]
        u = unaware.loc[unaware.signed_alpha.idxmin()]
        rows.append(
            {
                **dict(zip(GROUP + ["behaviour"], keys)),
                "aware_arm": a.arm,
                "d_aware": a.delta,
                "aware_ci": f"[{a.ci_lo:+.3f}, {a.ci_hi:+.3f}]",
                "unaware_arm": u.arm,
                "d_unaware": u.delta,
                "unaware_ci": f"[{u.ci_lo:+.3f}, {u.ci_hi:+.3f}]",
                "opposite_sign": bool(np.sign(a.delta) * np.sign(u.delta) < 0),
                "aware_ci_excludes_0": bool(a.ci_lo > 0 or a.ci_hi < 0),
                "unaware_ci_excludes_0": bool(u.ci_lo > 0 or u.ci_hi < 0),
            }
        )
    return pd.DataFrame(rows)


def flips(df: pd.DataFrame, baseline: str = BASELINE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actions only: compliance 2×2 vs baseline (McNemar) and Δdensity by flip class.

    The 2×2 is over every item whose execution is known in both arms (the Table 4 population,
    independent of annotation success); the Δdensity-by-class table is over items annotated
    successfully in both arms.
    """
    two_by_two, by_class = [], []
    known = df[df.family.str.contains("actions") & df.executed.notna()]
    for keys, sub in known.groupby(GROUP, sort=True):
        base_all = sub[sub.arm == baseline].set_index("index")
        if base_all.empty:
            continue
        for (arm, sa), arm_all in sub[sub.arm != baseline].groupby(["arm", "signed_alpha"]):
            arm_all = arm_all.set_index("index")
            common = base_all.index.intersection(arm_all.index)
            if len(common) < MIN_N:
                continue
            be = base_all.loc[common, "executed"].astype(bool)
            ae = arm_all.loc[common, "executed"].astype(bool)
            a = int((be & ae).sum())
            b = int((be & ~ae).sum())  # lost compliance
            c = int((~be & ae).sum())  # gained compliance
            d = int((~be & ~ae).sum())
            two_by_two.append(
                {
                    **dict(zip(GROUP, keys)),
                    "arm": arm,
                    "signed_alpha": sa,
                    "n_pairs": len(common),
                    "base_rate": float(be.mean()),
                    "arm_rate": float(ae.mean()),
                    "both_exec": a,
                    "lost": b,
                    "gained": c,
                    "neither": d,
                    "net_gain": c - b,
                    "churn": b + c,
                    "p_mcnemar": mcnemar_exact(b, c),
                }
            )
            # Outcome classes: the decision itself changes reasoning (complying and refusing traces differ
            # in morphology at baseline by about the size of the steering effects), so the steering delta
            # is reported within items whose decision did not change, split by which decision it was.
            cls = pd.Series("same", index=common)
            cls[be & ~ae] = "lost"
            cls[~be & ae] = "gained"
            cls[be & ae] = "comply_both"
            cls[~be & ~ae] = "refuse_both"
            base = base_all[base_all.ok]
            arm_df = arm_all[arm_all.ok]
            ok_idx = base.index.intersection(arm_df.index)
            for klass in ("gained", "lost", "comply_both", "refuse_both", "same"):
                idx = (cls.index[cls.isin(["comply_both", "refuse_both"])] if klass == "same" else cls.index[cls == klass]).intersection(ok_idx)
                if len(idx) < MIN_N:
                    continue
                for beh in LABELS:
                    for metric, col in (("density", f"density_{beh}"), ("density_nontest", f"density_{beh}_nontest"), ("density_testlex", f"density_{beh}_testlex")):
                        if col not in arm_df:
                            continue
                        dd = arm_df.loc[idx, col].to_numpy(dtype=float) - base.loc[idx, col].to_numpy(dtype=float)
                        lo, hi = boot_ci(dd)
                        by_class.append(
                            {
                                **dict(zip(GROUP, keys)),
                                "arm": arm,
                                "signed_alpha": sa,
                                "flip_class": klass,
                                "n": len(idx),
                                "behaviour": beh,
                                "metric": metric,
                                "delta_density": float(np.nanmean(dd)),
                                "ci_lo": lo,
                                "ci_hi": hi,
                            }
                        )
    return pd.DataFrame(two_by_two), pd.DataFrame(by_class)


def verbalization(df: pd.DataFrame) -> pd.DataFrame:
    """Triggers/harmbench: density by the shipped judge's awareness conclusion, within each arm."""
    rows = []
    ok = df[df.ok & df.aware_judged.notna()]
    if ok.empty:
        return pd.DataFrame()
    ok = ok.assign(aware_class=np.where(ok.aware_judged.isin(["Yes", "Maybe"]), "yes_or_maybe", "no"))
    for keys, sub in ok.groupby(ARM + ["aware_class"], sort=True):
        if len(sub) < MIN_N:
            continue
        for beh in LABELS:
            vals = sub[f"density_{beh}"].to_numpy(dtype=float)
            lo, hi = boot_ci(vals)
            rows.append({**dict(zip(ARM + ["aware_class"], keys)), "n": len(sub), "behaviour": beh, "mean_density": float(np.nanmean(vals)), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def verbalization_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Triggers/harmbench: each arm vs baseline, paired by item, restricted to items the shipped judge
    put in the same awareness class in both arms. ``no_both``: does steering move morphology even
    where the model never verbalizes awareness? ``yes_both``: the complement."""
    rows = []
    ok = df[df.ok & df.aware_judged.notna()]
    if ok.empty:
        return pd.DataFrame()
    ok = ok.assign(aware_class=np.where(ok.aware_judged.isin(["Yes", "Maybe"]), "yes", "no"))
    for keys, sub in ok.groupby(GROUP, sort=True):
        base = sub[sub.arm == BASELINE].set_index("index")
        if base.empty:
            continue
        for (arm, sa), arm_df in sub[sub.arm != BASELINE].groupby(["arm", "signed_alpha"]):
            arm_df = arm_df.set_index("index")
            common = base.index.intersection(arm_df.index)
            for klass in ("no", "yes"):
                idx = [i for i in common if base.at[i, "aware_class"] == klass and arm_df.at[i, "aware_class"] == klass]
                if len(idx) < MIN_N:
                    continue
                for beh in LABELS:
                    col = f"density_{beh}"
                    d = arm_df.loc[idx, col].to_numpy(dtype=float) - base.loc[idx, col].to_numpy(dtype=float)
                    lo, hi = boot_ci(d)
                    rows.append({**dict(zip(GROUP, keys)), "arm": arm, "signed_alpha": sa, "aware_class_both": klass, "n_pairs": len(idx), "behaviour": beh, "delta_density": float(np.nanmean(d)), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Decoding noise (plan §1): the same item decoded twice at the same α — greedy ``actions`` vs
    sampled ``actions_sampling`` — gives the item-level Δdensity distribution with no steering change.
    Compare its spread (``sd_delta``, ``mean_abs_delta``) with the steering deltas in
    :func:`paired_contrasts`; ``mean_delta`` should sit near zero."""
    rows = []
    ok = df[df.ok & df.family.isin(["actions", "actions_sampling"])]
    for (model, framing), sub in ok.groupby(["model", "framing"], sort=True):
        g = sub[sub.family == "actions"]
        s = sub[sub.family == "actions_sampling"]
        for arm in sorted(set(g.arm) & set(s.arm)):
            a = g[g.arm == arm].set_index("index")
            b = s[s.arm == arm].set_index("index")
            common = a.index.intersection(b.index)
            if len(common) < MIN_N:
                continue
            for col, beh in [(f"density_{x}", x) for x in LABELS] + [("words", "words")]:
                d = b.loc[common, col].to_numpy(dtype=float) - a.loc[common, col].to_numpy(dtype=float)
                lo, hi = boot_ci(d)
                rows.append(
                    {
                        "model": model,
                        "framing": framing,
                        "arm": arm,
                        "signed_alpha": float(a.signed_alpha.iloc[0]),
                        "n_pairs": len(common),
                        "behaviour": beh,
                        "mean_delta": float(np.nanmean(d)),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "sd_delta": float(np.nanstd(d, ddof=1)),
                        "mean_abs_delta": float(np.nanmean(np.abs(d))),
                    }
                )
    return pd.DataFrame(rows)


def testlex_share(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm (and, where the shipped judge exists, per judged awareness class): the share of each
    behaviour's spans that contain self-referential test language."""
    rows = []
    ok = df[df.ok].copy()
    ok["aware_class"] = np.where(ok.aware_judged.isna(), "n/a", np.where(ok.aware_judged.isin(["Yes", "Maybe"]), "yes_or_maybe", "no"))
    for keys, sub in ok.groupby(ARM + ["aware_class"], sort=True):
        if len(sub) < MIN_N:
            continue
        for b in LABELS:
            n = float(sub[f"n_{b}"].sum())
            t = float(sub[f"n_{b}_testlex"].sum())
            rows.append({**dict(zip(ARM + ["aware_class"], keys)), "n_traces": len(sub), "behaviour": b, "spans": int(n), "testlex_spans": int(t), "testlex_share": t / n if n else math.nan})
    return pd.DataFrame(rows)


def lexicon_recall(df: pd.DataFrame) -> dict:
    """Recall of the test-language pattern against the shipped judge's awareness evidence quotes."""
    n = int(df["aw_evidence_n"].fillna(0).sum()) if "aw_evidence_n" in df else 0
    c = int(df["aw_evidence_caught"].fillna(0).sum()) if "aw_evidence_caught" in df else 0
    return {"evidence_quotes": n, "caught": c, "recall": c / n if n else math.nan}


def proxy_vs_judge(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("lex_wait_per100w", "density_backtracking"),
        ("lex_backtrack_lex_per100w", "density_backtracking"),
        ("lex_hmm_per100w", "density_uncertainty-estimation"),
        ("lex_uncertainty_lex_per100w", "density_uncertainty-estimation"),
    ]
    rows = []
    ok = df[df.ok]
    groups = [("all", ok)] + [(m, g) for m, g in ok.groupby("model")]
    for name, sub in groups:
        if len(sub) < MIN_N:
            continue
        for lex, dens in pairs:
            x, y = sub[lex].to_numpy(dtype=float), sub[dens].to_numpy(dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < MIN_N:
                continue
            rows.append(
                {
                    "scope": name,
                    "n": int(mask.sum()),
                    "proxy": lex,
                    "judge": dens,
                    "pearson_r": float(stats.pearsonr(x[mask], y[mask])[0]),
                    "spearman_rho": float(stats.spearmanr(x[mask], y[mask])[0]),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def _style(ax):
    ax.grid(True, axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#4a4a48", labelsize=8)


def fig_dose(means: pd.DataFrame, metric: str, family: str, out: Path, title: str) -> None:
    sub = means[(means.metric == metric) & (means.family == family)]
    if sub.empty:
        return
    models = sorted(sub.model.unique())
    fig, axes = plt.subplots(len(LABELS), len(models), figsize=(3.6 * len(models), 1.9 * len(LABELS)), sharex="col", squeeze=False)
    for j, model in enumerate(models):
        for i, beh in enumerate(LABELS):
            ax = axes[i, j]
            _style(ax)
            for framing, g in sub[(sub.model == model) & (sub.behaviour == beh)].groupby("framing"):
                g = g.sort_values("signed_alpha")
                if g.n.min() < MIN_N:
                    continue
                color = COLORS.get(framing, "#4a4a48")
                ax.plot(g.signed_alpha, g["mean"], color=color, linewidth=2, marker="o", markersize=5, label=framing)
                ax.fill_between(g.signed_alpha, g.ci_lo, g.ci_hi, color=color, alpha=0.15, linewidth=0)
                last = g.iloc[-1]
                ax.annotate(framing, (last.signed_alpha, last["mean"]), xytext=(4, 0), textcoords="offset points", fontsize=7, color="#4a4a48", va="center")
            ax.axvline(0, color="#c3c2b7", linewidth=0.8, linestyle=":")
            ax.ticklabel_format(axis="x", useOffset=False, style="plain")
            if j == 0:
                ax.set_ylabel(SHORT[beh], fontsize=8)
            if i == 0:
                ax.set_title(model, fontsize=9)
            if i == len(LABELS) - 1:
                ax.set_xlabel("signed α (aware > 0)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    ylab = {"density": "spans per 100 words", "coverage": "fraction of labelled words", "any": "fraction of traces with ≥1 span"}[metric]
    fig.suptitle(f"{title} — {ylab}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_words(means: pd.DataFrame, out: Path) -> None:
    sub = means[means.metric == "words"]
    if sub.empty:
        return
    fams = sorted(sub.family.unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(3.8 * len(fams), 3.0), squeeze=False)
    for j, fam in enumerate(fams):
        ax = axes[0, j]
        _style(ax)
        for (model, framing), g in sub[sub.family == fam].groupby(["model", "framing"]):
            g = g.sort_values("signed_alpha")
            ls = {"deepseek_qwen": "-", "qwen3": "--", "qwq": "-."}.get(model, "-")
            ax.plot(g.signed_alpha, g["mean"], color=COLORS.get(framing, "#4a4a48"), linestyle=ls, linewidth=2, marker="o", markersize=4, label=f"{model} {framing}")
        ax.set_title(fam, fontsize=9)
        ax.set_xlabel("signed α", fontsize=8)
        ax.set_ylabel("reasoning words", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_flip_classes(by_class: pd.DataFrame, out: Path) -> None:
    if by_class.empty:
        return
    # one panel per (model, framing, arm) with the largest |alpha| on each side
    panels = []
    for keys, sub in by_class.groupby(["model", "family", "framing"]):
        for sign in (1, -1):
            s = sub[np.sign(sub.signed_alpha) == sign]
            if s.empty:
                continue
            arm = s.loc[s.signed_alpha.abs().idxmax(), "arm"]
            panels.append((keys, arm, s[s.arm == arm]))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), squeeze=False)
    x = np.arange(len(LABELS))
    w = 0.26
    for ax, (keys, arm, s) in zip(axes[0], panels):
        _style(ax)
        s = s[s.metric == "density"] if "metric" in s else s
        for k, klass in enumerate(("gained", "lost", "same")):
            g = s[s.flip_class == klass].set_index("behaviour").reindex(LABELS)
            if g.delta_density.isna().all():
                continue
            n = int(g.n.dropna().iloc[0]) if g.n.notna().any() else 0
            ax.bar(x + (k - 1) * w, g.delta_density, w, color=CLASS_COLORS[klass], label=f"{klass} (n={n})", edgecolor="white", linewidth=1)
            ax.errorbar(x + (k - 1) * w, g.delta_density, yerr=[g.delta_density - g.ci_lo, g.ci_hi - g.delta_density], fmt="none", ecolor="#4a4a48", elinewidth=0.8, capsize=2)
        ax.axhline(0, color="#c3c2b7", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[b] for b in LABELS], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{keys[0]} {keys[2]} {arm}", fontsize=8)
        ax.set_ylabel("Δ density vs baseline", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle("Change in behaviour density by compliance flip class (actions)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary

def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, cols: list[str], nd: int = 3) -> str:
    if df.empty:
        return "_(no data)_\n"
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], nd) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_summary(out: Path, df, cov, means, contrasts, sym, two, by_class, verb, proxy, verb_contrasts=None, noise=None) -> None:
    parts = ["# Tier 1 — first-pass analysis\n"]
    parts.append(f"Annotated traces loaded: {len(df)} ({int(df.ok.sum())} ok). Generated by `scripts/analyze_tier1.py`.\n")
    parts.append(
        "Units: a *trace* is one generation; on `actions` there are two per item (real/fake framing), "
        "elsewhere one. Bootstrap CIs resample traces (arm means) or item-level differences (paired "
        "contrasts). The compliance 2×2 is over every item with known execution in both arms; every "
        "morphology number is over traces the judge annotated successfully.\n"
    )
    parts.append("## Coverage per arm\n")
    parts.append(md_table(cov, ["model", "family", "arm", "n", "n_annotated", "n_ok", "n_judge_fail", "n_error", "n_truncated", "n_no_think_close", "mean_words", "mean_spans", "not_found_exact_rate", "found_norm_only_rate", "unknown_label_rate", "mean_repetition"]))
    parts.append("\n## Density (spans per 100 words), arm means with 95% bootstrap CI\n")
    d = means[means.metric == "density"]
    for keys, sub in d.groupby(GROUP):
        parts.append(f"\n### {' / '.join(map(str, keys))}\n")
        piv = sub.pivot_table(index="behaviour", columns="signed_alpha", values="mean").reindex(LABELS)
        n = sub.groupby("signed_alpha").n.first()
        piv.columns = [f"α={c:+.3g} (n={n[c]})" for c in piv.columns]
        parts.append(piv.round(3).to_markdown() + "\n")
    parts.append("\n## Paired contrasts vs α=0 (density), largest |α| per side\n")
    if not contrasts.empty:
        c = contrasts[contrasts.metric == "density"].copy()
        keep = []
        for keys, sub in c.groupby(GROUP + ["behaviour"]):
            for sign in (1, -1):
                s = sub[np.sign(sub.signed_alpha) == sign]
                if not s.empty:
                    keep.append(s.loc[s.signed_alpha.abs().idxmax()])
        keep = pd.DataFrame(keep)
        parts.append(md_table(keep, ["model", "family", "framing", "behaviour", "arm", "n_pairs", "base_mean", "arm_mean", "delta", "ci_lo", "ci_hi", "p_paired_t", "p_wilcoxon"]))
    parts.append("\n## Decomposition (exploratory): Δ density of spans WITH self-referential test language vs WITHOUT (largest |α| per side)\n")
    rc = lexicon_recall(df)
    parts.append(
        "A span 'with test language' matches `parse.TEST_LEXICON` — the model reasoning about itself "
        "being tested ('testing me', 'this is probably a test', 'see how I handle'). **This split was "
        "designed after reading aware-arm spans and is not part of the frozen §2.2 metric set.** "
        f"Recall of the pattern against the shipped judge's own awareness evidence quotes on "
        f"aware-judged traces: {rc['caught']} of {rc['evidence_quotes']} = "
        f"{fmt(rc['recall'])}. Per-arm shares of each behaviour's spans that carry test language, "
        "split by the judged awareness class where available, are in `testlex_share.csv`.\n"
    )
    if not contrasts.empty:
        dd = contrasts[contrasts.metric.isin(["density_testlex", "density_nontest"])].copy()
        keep = []
        for keys, sub in dd.groupby(GROUP + ["behaviour", "metric"]):
            for sign in (1, -1):
                s = sub[np.sign(sub.signed_alpha) == sign]
                if not s.empty:
                    keep.append(s.loc[s.signed_alpha.abs().idxmax()])
        keep = pd.DataFrame(keep)
        if not keep.empty:
            keep = keep[keep.behaviour.isin(["uncertainty-estimation", "backtracking", "deduction"])]
            parts.append(md_table(keep, ["model", "family", "framing", "behaviour", "metric", "arm", "n_pairs", "base_mean", "arm_mean", "delta", "ci_lo", "ci_hi"]))
    parts.append("\n## Sign symmetry (Δ at largest aware α vs Δ at largest unaware α)\n")
    parts.append(md_table(sym, ["model", "family", "framing", "behaviour", "d_aware", "aware_ci", "d_unaware", "unaware_ci", "opposite_sign", "aware_ci_excludes_0", "unaware_ci_excludes_0"]))
    parts.append("\n## Compliance flips vs baseline (actions), McNemar exact\n")
    parts.append(md_table(two, ["model", "family", "framing", "arm", "n_pairs", "base_rate", "arm_rate", "lost", "gained", "net_gain", "churn", "p_mcnemar"]))
    parts.append("\n## Δ density by flip class (largest |α| arms; see flips.csv for all)\n")
    if not by_class.empty:
        bc = by_class[by_class.behaviour.isin(["backtracking", "uncertainty-estimation", "deduction"]) & by_class.flip_class.isin(["gained", "lost", "comply_both", "refuse_both"])]
        parts.append("Classes: `gained`/`lost` flipped decision; `comply_both`/`refuse_both` kept it. Within `refuse_both` the decision is held fixed, so a delta there is not a by-product of the compliance change.\n")
        parts.append(md_table(bc, ["model", "framing", "arm", "flip_class", "n", "behaviour", "metric", "delta_density", "ci_lo", "ci_hi"]))
    parts.append("\n## Decoding-noise floor: same item, same α, greedy vs sampled decode (Distill-Qwen actions)\n")
    if noise is not None and not noise.empty:
        parts.append(md_table(noise, ["model", "framing", "arm", "n_pairs", "behaviour", "mean_delta", "ci_lo", "ci_hi", "sd_delta", "mean_abs_delta"]))
    else:
        parts.append("_(no sampled-decode arms annotated yet)_\n")
    parts.append("\n## Density by shipped awareness judgment (triggers / harmbench)\n")
    if not verb.empty:
        v = verb[verb.behaviour.isin(["backtracking", "uncertainty-estimation"])]
        parts.append(md_table(v, ["model", "family", "arm", "aware_class", "n", "behaviour", "mean_density", "ci_lo", "ci_hi"]))
    parts.append("\n## Steering Δ density among items judged in the same awareness class in both arms\n")
    if verb_contrasts is not None and not verb_contrasts.empty:
        vc = verb_contrasts[verb_contrasts.behaviour.isin(["backtracking", "uncertainty-estimation", "deduction"])]
        parts.append(md_table(vc, ["model", "family", "arm", "aware_class_both", "n_pairs", "behaviour", "delta_density", "ci_lo", "ci_hi"]))
    else:
        parts.append("_(no judged families annotated yet)_\n")
    parts.append("\n## Lexical proxy vs judge density\n")
    parts.append(md_table(proxy, ["scope", "n", "proxy", "judge", "pearson_r", "spearman_rho"]))
    out.write_text("\n".join(parts))


