#!/usr/bin/env python
"""First-pass analysis of the mechanistic arm (EXECUTION_PLAN_2 §2b): probe, geometry, α sweep, and
the main steering run's four readouts, each arm against baseline paired by item. Reuses the Tier 1
analysis functions by building a trace table in the same schema.

Writes ``results/qwen3_4b/SUMMARY_M.md`` plus CSVs alongside.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from d10 import analysis as A
from d10 import parse
from d10.awareness import is_positive, parse_judgment
from d10.judge import load_results, record_ok
from d10.parse import LABELS
from d10.shipped import execution_rate, load_generations

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "qwen3_4b"
ORDER = ["baseline", "baseline_seed1", "unaware", "aware", "unaware_strong", "aware_strong", "random_aware", "shuffled_aware", "random_samerows_aware", "shuffled_samerows_aware"]


def build_trace_table(steering_dir: Path, model_name: str) -> pd.DataFrame:
    rows = []
    for p in sorted(steering_dir.glob("*.json")):
        if p.name.count(".") != 1:  # sidecars: .gen.json, .stats.json, .projections.json
            continue
        arm = p.stem
        side = json.loads(p.with_suffix(p.suffix + ".gen.json").read_text())
        alpha, aware = side.get("alpha", 0.0), bool(side.get("aware"))
        ann = load_results(p.with_suffix(p.suffix + ".annotations.jsonl"))
        aw = load_results(p.with_suffix(p.suffix + ".awareness.jsonl"))
        projf = p.with_suffix(p.suffix + ".projections.json")
        proj = {r["id"]: r for r in json.loads(projf.read_text())["rows"]} if projf.exists() else {}
        for t in load_generations(p, model_name, "steer_actions", arm, alpha, aware):
            rec = ann.get(t.id)
            usable = bool(rec and record_ok(rec))
            row = {"id": t.id, "model": model_name, "family": "steer_actions", "arm": arm, "alpha": alpha, "aware": aware,
                   "signed_alpha": (alpha if aware else -alpha) if arm != "baseline" else 0.0, "index": t.index, "framing": t.framing,
                   "has_think_close": t.has_think_close, "executed": t.executed, "annotated": usable, "error": None if usable else "missing",
                   "finish_reason": rec.get("finish_reason") if rec else None, "served_model": rec.get("served_model") if rec else None,
                   "aware_judged": None, "recog_judged": None, "task_judged": None, "aw_evidence_n": 0, "aw_evidence_caught": 0}
            j = aw.get(t.id)
            if j and record_ok(j):
                pj = parse_judgment(j["text"])
                row["aware_judged"] = "Yes" if is_positive(pj) else "No"
            ok = False
            if usable:
                spans = parse.parse_annotation(rec["text"], t.reasoning)
                row.update(parse.trace_metrics(spans, t.reasoning))
                row.update(parse.lexical_proxy(t.reasoning))
                ok = row["n_spans_counted"] > 0
            row["ok"] = ok
            row["judge_fail"] = usable and not ok
            pr = proj.get(t.id) or {}
            for k, v in pr.items():
                if k.startswith("proj_"):
                    row[k] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in A.METRIC_COLUMNS:
        if c not in df:
            df[c] = np.nan
    df["unknown_labels"] = df.get("unknown_labels", pd.Series([None] * len(df))).apply(lambda v: ";".join(v) if isinstance(v, list) else v)
    return df


def md(df: pd.DataFrame, cols: list[str], nd: int = 3) -> str:
    return A.md_table(df, cols, nd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--model-name", default="qwen3_4b")
    args = ap.parse_args()
    parts = ["# Mechanistic arm (Qwen3-4B) — first-pass analysis\n"]

    # gate + probe
    for name in ("triggers_gate.json.awareness.stats.json", "triggers_probe.json.awareness.stats.json"):
        f = args.root / name
        if f.exists():
            s = json.loads(f.read_text())
            parts.append(f"- `{name}`: positives {s['positives']} / judged {s['judged']} / generated {s['generated']} → {s['verdict']} (rate over generated {s['positive_rate_generated']:.3f})")
    pm = args.root / "probe" / "probe_metrics.json"
    if pm.exists():
        m = json.loads(pm.read_text())
        parts.append(f"\n## Probe\n\nbest hidden-state index {m['best_layer']}: acc {m['best_acc']:.3f}, AUROC {m['best_auroc']:.3f}; spans pos/neg (balanced) {m['n_spans_pos_balanced']}/{m['n_spans_neg_balanced']}; controls: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m['controls'].items()})}\n")
        pl = pd.DataFrame(m["per_layer"])
        pl.to_csv(args.root / "probe_per_layer.csv", index=False)
        parts.append(md(pl.iloc[::3], ["layer", "n_train", "n_test", "acc", "auroc", "sk_acc", "sk_auroc"]))
    pool_rows = []
    for pooling in ("span", "whole", "last"):
        f = args.root / ("probe" if pooling == "span" else f"probe_{pooling}") / "probe_metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            c = m["controls"]
            pool_rows.append({"pooling": pooling, "best_layer": m["best_layer"], "acc": m["best_acc"], "auroc": m["best_auroc"], "shuffled_auroc": c.get("shuffled_auroc"), "layer0_auroc": c.get("layer0_auroc"), "n_pos": c.get("n_pos"), "n_neg": c.get("n_neg")})
    if len(pool_rows) > 1:
        parts.append("\n## Probe token-position ablation (their §4.2): span mean vs whole-reasoning mean vs last token\n")
        parts.append(md(pd.DataFrame(pool_rows), ["pooling", "best_layer", "acc", "auroc", "shuffled_auroc", "layer0_auroc", "n_pos", "n_neg"]))
    geo = args.root / "geometry.json"
    if geo.exists():
        g = json.loads(geo.read_text())
        parts.append(f"\n## Geometry at the probe's layer (hidden_states[{g['probe_best_hs_index']}]); chance |cos| = {g['chance_abs_cos']:.4f}\n")
        ab = g["at_best_layer"]
        rows = [{"behaviour": b, "all_spans": ab.get(f"mean_vectors/{b}"), "nontest_spans": ab.get(f"mean_vectors_nontest/{b}"), "testlex_spans": ab.get(f"mean_vectors_testlex/{b}"), "shuffled_probe": g["shuffled_probe_at_best"].get(b)} for b in LABELS]
        parts.append(md(pd.DataFrame(rows), ["behaviour", "all_spans", "nontest_spans", "testlex_spans", "shuffled_probe"]))
        parts.append(f"random-direction |cos| at this layer: {ab['random']:.4f}\n")
    sw = args.root / "sweep" / "chosen.json"
    if sw.exists():
        s = json.loads(sw.read_text())
        parts.append("\n## α sweep\n")
        parts.append(md(pd.DataFrame(s["arms"]), ["arm", "alpha", "aware", "n", "closure", "repetition", "n_judged", "verbalization"]))
        parts.append(f"chosen: {json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('arm', 'alpha', 'verbalization', 'closure', 'repetition', 'flagged_no_qualifying_arm')} for k, v in s['chosen'].items()})}\n")

    sd = args.root / "steering"
    if sd.exists() and any(sd.glob("*.json")):
        df = build_trace_table(sd, args.model_name)
        df.to_csv(args.root / "steering_traces.csv", index=False)
        parts.append(f"\n## Main run: {len(df)} traces, {int(df.ok.sum())} with usable morphology annotation\n")
        # compliance + verbalization per arm
        rows = []
        base = df[df.arm == "baseline"].set_index(["index", "framing"])
        for arm in [a for a in ORDER if a in set(df.arm)]:
            sub = df[df.arm == arm]
            for fr in ("real", "fake"):
                s = sub[sub.framing == fr]
                ex = s.executed.tolist()
                r = {"arm": arm, "framing": fr, "n": len(s), "exec_rate_closed": execution_rate(ex, "closed"), "exec_rate_all": execution_rate(ex, "all"), "closure": float(s.has_think_close.mean()),
                     "verbalization": float((s.aware_judged[s.aware_judged.notna()] == "Yes").mean()) if s.aware_judged.notna().any() else math.nan, "repetition": float(s.repetition_4gram.mean()), "words": float(s.words.mean()),
                     "lost": math.nan, "gained": math.nan, "p_mcnemar": math.nan}
                if arm != "baseline":
                    b = base.loc[[(i, fr) for i in s["index"] if (i, fr) in base.index]]
                    ss = s.set_index(["index", "framing"]).loc[b.index]
                    be, ae = b.executed, ss.executed
                    known = be.notna() & ae.notna()
                    lost = int((be[known].astype(bool) & ~ae[known].astype(bool)).sum())
                    gained = int((~be[known].astype(bool) & ae[known].astype(bool)).sum())
                    r.update({"lost": lost, "gained": gained, "p_mcnemar": A.mcnemar_exact(lost, gained)})
                rows.append(r)
        parts.append(md(pd.DataFrame(rows), ["arm", "framing", "n", "exec_rate_closed", "exec_rate_all", "lost", "gained", "p_mcnemar", "verbalization", "closure", "repetition", "words"]))
        # morphology contrasts (reuse Tier 1 machinery): arm vs baseline paired by item
        contrasts = A.paired_contrasts(df, baseline="baseline")
        two, by_class = A.flips(df, baseline="baseline")
        two.to_csv(args.root / "steering_flips_2x2.csv", index=False)
        by_class.to_csv(args.root / "steering_flips_by_class.csv", index=False)
        contrasts.to_csv(args.root / "steering_contrasts.csv", index=False)
        c = contrasts[contrasts.metric.isin(["density", "density_testlex", "density_nontest"]) & contrasts.behaviour.isin(["uncertainty-estimation", "deduction", "adding-knowledge", "backtracking"])] if not contrasts.empty else contrasts
        parts.append("\n## Morphology: Δ spans per 100 words vs baseline, paired by item\n")
        parts.append(md(c, ["framing", "arm", "n_pairs", "behaviour", "metric", "base_mean", "arm_mean", "delta", "ci_lo", "ci_hi", "p_paired_t"]))
        if not by_class.empty:
            bc = by_class[by_class.behaviour.isin(["uncertainty-estimation", "deduction", "backtracking"]) & by_class.flip_class.isin(["gained", "lost", "comply_both", "refuse_both"])]
            parts.append("\n## Morphology Δ by compliance class (decision-change confound): `refuse_both` / `comply_both` hold the decision fixed\n")
            parts.append(md(bc, ["framing", "arm", "flip_class", "n", "behaviour", "metric", "delta_density", "ci_lo", "ci_hi"]))
        # projections
        pcols = [c for c in df.columns if c.startswith("proj_")]
        if pcols:
            rows = []
            for arm in [a for a in ORDER if a in set(df.arm) and a != "baseline"]:
                for fr in ("real", "fake"):
                    a = df[(df.arm == arm) & (df.framing == fr)].set_index("index")
                    b = df[(df.arm == "baseline") & (df.framing == fr)].set_index("index")
                    idx = a.index.intersection(b.index)
                    for pc in pcols:
                        d = a.loc[idx, pc].to_numpy(dtype=float) - b.loc[idx, pc].to_numpy(dtype=float)
                        d = d[~np.isnan(d)]
                        lo, hi = A.boot_ci(d)
                        rows.append({"arm": arm, "framing": fr, "projection": pc, "n_pairs": len(d), "base_mean": float(np.nanmean(b.loc[idx, pc])), "delta": float(d.mean()) if len(d) else math.nan, "ci_lo": lo, "ci_hi": hi})
            pr = pd.DataFrame(rows)
            pr.to_csv(args.root / "steering_projections.csv", index=False)
            parts.append("\n## Representation-level morphology: Δ mean projection onto the rebuilt directions (steered model's own activations)\n")
            parts.append(md(pr, ["arm", "framing", "projection", "n_pairs", "base_mean", "delta", "ci_lo", "ci_hi"], 4))
    (args.root / "SUMMARY_M.md").write_text("\n".join(parts) + "\n")
    print(f"wrote {args.root / 'SUMMARY_M.md'}")


if __name__ == "__main__":
    main()
