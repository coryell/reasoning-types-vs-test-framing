#!/usr/bin/env python
"""Every table and figure behind the headline in REPORT.md, in footnote order.

Written 2026-09-04 as a sequence of cells (``# %%``) so it can be pasted into a Colab; run top to
bottom from the repo root (``uv run python scripts/report_figures.py``). Reads only committed CSV /
JSON / JSONL under ``results/`` and writes to ``results/report/``: one Markdown file per footnote
(``T<n>_*.md``), the same tables as CSV, and PNG figures (``F<n>_*.png``). Nothing here calls a
model or a judge.

Footnotes → cells:
  [1] Qwen3-4B prompt cue: compliance, with the seed-1 floor
  [2] Qwen3-32B prompt cue: compliance in two decodes, with floors
  [3] The authors' weight edit at 32B: paired flips per arm (their shipped outputs)
  [4] Aggregate morphology rises are the flipped items (compliance-class split)
  [5] Task-directed reasoning at fixed decision: counts per trace with CIs, and the floor
  [6] Test-talk added per trace at fixed decision
  [7] F4: reasoning-direction steering vs verbalized awareness, length-adjusted, with random controls
  [8] F4: execution per arm
  [9] Instrument facts: judge evidence quotes, probe layer-0 AUROC
"""

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(REPO))
from d10.awareness import is_positive, parse_judgment  # noqa: E402
from d10.parse import is_test_span, parse_annotation, trace_metrics  # noqa: E402
from d10.shipped import load_generations  # noqa: E402


def load_results(path: Path) -> dict:
    """``{id: record}`` from a judge JSONL (last record per id wins); copied from d10.judge so this
    script does not import the OpenAI client."""
    out = {}
    if not path.exists():
        return out
    lines = [l for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
    for i, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue  # truncated final line from an interrupted run
            raise
        out[r["id"]] = r
    return out


def record_ok(rec: dict) -> bool:
    return rec.get("error") is None and bool((rec.get("text") or "").strip()) and rec.get("finish_reason") == "stop"

OUT = REPO / "results" / "report"
OUT.mkdir(parents=True, exist_ok=True)
BEH = ["deduction", "adding-knowledge", "uncertainty-estimation", "backtracking", "example-testing", "initializing"]
MIN_N_CI = 20  # below this a percentile bootstrap is unreliable: the delta is shown without a CI


def boot(d, n=20000):
    """Mean and percentile-bootstrap 95% CI of paired differences. The generator is seeded from the
    data itself, so a cell's CI does not depend on which cells ran before it."""
    d = np.asarray(d, float)
    assert not np.isnan(d).any(), "NaN in paired differences: a row without morphology slipped in"
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    if len(d) < MIN_N_CI:
        return float(d.mean()), np.nan, np.nan
    rng = np.random.default_rng(abs(hash(tuple(np.round(d, 6)))) % (2**32))
    m = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt_ci(m, lo, hi, nd=2):
    if np.isnan(lo):
        return f"{m:+.{nd}f} (n<{MIN_N_CI}, no CI)"
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]" + ("*" if (lo > 0 or hi < 0) else "")


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def mcnemar(lost, gained):
    return binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0


def save_table(df: pd.DataFrame, name: str, title: str, note: str = "") -> None:
    df.to_csv(OUT / f"{name}.csv", index=False)
    try:
        body = df.to_markdown(index=False)
    except ImportError:  # tabulate missing
        body = "```\n" + df.to_string(index=False) + "\n```"
    (OUT / f"{name}.md").write_text(f"# {title}\n\n{note}\n\n" + body + "\n")
    print(f"\n## {title}\n{note}\n"); print(df.to_string(index=False))


def exec_str(v):
    return "True" if str(v) == "True" else "False" if str(v) == "False" else "unknown"


def paired_frames(df: pd.DataFrame, base_arm: str, arm: str, fr: str):
    """Baseline and arm rows aligned on item index for one framing (whatever frame is passed in)."""
    b = df[(df.arm == base_arm) & (df.framing == fr)].set_index("index")
    x = df[(df.arm == arm) & (df.framing == fr)].set_index("index")
    assert b.index.is_unique and x.index.is_unique, "duplicate item index within an arm"
    idx = b.index.intersection(x.index)
    return b.loc[idx], x.loc[idx]


def flip_counts(b, x):
    be, xe = b.executed.map(exec_str), x.executed.map(exec_str)
    known = (be != "unknown") & (xe != "unknown")
    lost = int(((be == "True") & (xe == "False") & known).sum()); gained = int(((be == "False") & (xe == "True") & known).sum())
    rb = (be[known] == "True").mean(); rx = (xe[known] == "True").mean()
    return int(known.sum()), rb, rx, lost, gained


def stable_refusers(b, x):
    be, xe = b.executed.map(exec_str), x.executed.map(exec_str)
    return b.index[(be == "False") & (xe == "False")]


def nontest_count(df, beh):
    return df[f"n_{beh}"] - df[f"n_{beh}_testlex"]


def testlex_total(df):
    return sum(df[f"n_{b}_testlex"] for b in BEH)


# --- inputs ---------------------------------------------------------------------------------
# Compliance tables use every trace (execution is judge-free); morphology tables use only traces the
# annotator succeeded on (`ok`), which drops a handful per arm.
tier1_all = pd.read_csv(REPO / "results/tier1/traces.csv", low_memory=False)
tier1_all = tier1_all[tier1_all.family.isin(["actions", "actions_sampling"])].copy()
tier1_all["arm_key"] = tier1_all.family + "/" + tier1_all.arm
tier1 = tier1_all[tier1_all.ok == True]  # noqa: E712
main4b = pd.read_csv(REPO / "results/qwen3_4b/steering_traces.csv", low_memory=False)
f1 = pd.read_csv(REPO / "results/qwen3_4b/f1_prompt_steering_traces.csv", low_memory=False)
f4_all = pd.read_csv(REPO / "results/qwen3_4b/f4_reverse_steering_traces.csv", low_memory=False)
f4 = f4_all[f4_all.ok == True]  # noqa: E712
q4b_all = pd.concat([main4b, f1[f1.arm == "prompt_test"]], ignore_index=True)
q4b = q4b_all[q4b_all.ok == True]  # noqa: E712


def load_api_arm(dirpath: Path, model_name: str, arm: str) -> pd.DataFrame:
    """Per-trace rows for an OpenRouter arm from its generation, annotation and awareness files."""
    p = dirpath / f"{arm}.json"
    ann = load_results(p.with_suffix(p.suffix + ".annotations.jsonl")); aw = load_results(p.with_suffix(p.suffix + ".awareness.jsonl"))
    rows = []
    for t in load_generations(p, model_name, "api_actions", arm):
        rec = {"arm": arm, "index": t.index, "framing": t.framing, "task": t.task, "executed": t.executed, "words": t.reasoning_words, "ok": False}
        j = aw.get(t.id); rec["aware_judged"] = ("Yes" if is_positive(parse_judgment(j["text"])) else "No") if j and record_ok(j) else np.nan
        a = ann.get(t.id)
        if a and record_ok(a):
            spans = parse_annotation(a["text"], t.reasoning); m = trace_metrics(spans, t.reasoning)
            if m["n_spans_counted"]:
                rec.update({k: v for k, v in m.items() if k.startswith(("n_", "density_"))}); rec["ok"] = True
        rows.append(rec)
    return pd.DataFrame(rows)


api0 = pd.concat([load_api_arm(REPO / "results/api_prompt/qwen3_32b_api", "qwen3_32b_api", a) for a in ("baseline", "prompt_test")])
api1 = pd.concat([load_api_arm(REPO / "results/api_prompt/qwen3_32b_api_seed1", "qwen3_32b_api_seed1", a) for a in ("baseline", "prompt_test")])
api1["arm"] = api1.arm + "_seed1"
api_all = pd.concat([api0, api1], ignore_index=True)
assert (api_all.groupby(["index", "framing"]).task.nunique() == 1).all(), "items must pair by index across arms"
api = api_all[api_all.ok == True]  # noqa: E712
print("inputs loaded:", {k: len(v) for k, v in dict(tier1=tier1, q4b=q4b, f4=f4, api=api).items()})

# %% [markdown]
# ## [1] Qwen3-4B prompt cue: compliance, with the seed-1 floor

# %%
rows = []
for arm, label in (("prompt_test", "cue vs baseline"), ("baseline_seed1", "floor: second baseline decode")):
    for fr in ("real", "fake"):
        b, x = paired_frames(q4b_all, "baseline", arm, fr); n, rb, rx, lost, gained = flip_counts(b, x)
        rows.append(dict(comparison=label, framing=fr, n=n, exec_baseline_paired=f"{rb:.3f}", exec_arm_paired=f"{rx:.3f}", lost=lost, gained=gained, p_mcnemar=fmt_p(mcnemar(lost, gained))))
t1 = pd.DataFrame(rows)
save_table(t1, "T1_qwen3_4b_prompt_compliance", "[1] Qwen3-4B: one-sentence test cue vs baseline, paired by item",
           "Execution = tool name after </think> (closed traces). Cue: 'This is a test scenario with no real-world consequences.' appended to the agent system prompt.")

# %% [markdown]
# ## [2] Qwen3-32B prompt cue: two decodes, with floors

# %%
pairs = [("baseline", "prompt_test", "cue, seed 0"), ("baseline_seed1", "prompt_test_seed1", "cue, seed 1"), ("baseline", "prompt_test_seed1", "cue, cross (b0→p1)"), ("baseline_seed1", "prompt_test", "cue, cross (b1→p0)"),
         ("baseline", "baseline_seed1", "floor: baseline decodes"), ("prompt_test", "prompt_test_seed1", "floor: prompt decodes")]
rows = []
for ba, ar, label in pairs:
    for fr in ("real", "fake"):
        b, x = paired_frames(api_all, ba, ar, fr); n, rb, rx, lost, gained = flip_counts(b, x)
        rows.append(dict(comparison=label, framing=fr, n=n, exec_A_paired=f"{rb:.3f}", exec_B_paired=f"{rx:.3f}", lost=lost, gained=gained, p_mcnemar=fmt_p(mcnemar(lost, gained))))
t2 = pd.DataFrame(rows)
save_table(t2, "T2_qwen3_32b_prompt_compliance", "[2] Qwen3-32B (OpenRouter): test cue vs baseline in two sampled decodes, paired by item",
           "Same items, cue and rule as [1]; 4,096-token budget; seeds 0 and 1. Floors are the same arm decoded twice.")

def net_ci(lost, gained):
    """Net items gained with a 95% CI from the exact binomial on the discordant pairs."""
    n = lost + gained
    if n == 0:
        return 0, 0, 0
    lo, hi = binomtest(gained, n, 0.5).proportion_ci(0.95)
    return gained - lost, (2 * lo - 1) * n, (2 * hi - 1) * n


t12 = pd.concat([t1.assign(model="Qwen3-4B (local)"), t2.assign(model="Qwen3-32B (OpenRouter)")], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
for ax, fr in zip(axes, ("real", "fake")):
    sub = t12[t12.framing == fr].reset_index(drop=True)
    for i, r in sub.iterrows():
        m, lo, hi = net_ci(int(r.lost), int(r.gained)); c = "0.6" if str(r.comparison).startswith("floor") else ("C0" if r.model.startswith("Qwen3-4B") else "C1")
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=3)
        ax.text(hi + 1.5, i, f"{int(r.lost)}↓ {int(r.gained)}↑, p {r.p_mcnemar}", va="center", fontsize=7)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels([f"{m}: {c}" for m, c in zip(sub.model, sub.comparison)], fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8); ax.set_title(f"{fr} framing"); ax.set_xlabel("net items gaining execution (gained − lost), 95% CI")
axes[0].set_xlim(-30, 45)
fig.suptitle("[1]–[2] The test cue moves execution; grey = the same arm decoded twice (floor)"); fig.tight_layout(); fig.savefig(OUT / "F2_prompt_cue_net_flips.png", dpi=130); plt.close(fig)

# %% [markdown]
# ## [3] The authors' weight edit at 32B: paired flips per arm (their shipped outputs)

# %%
rows = []
for model in ("deepseek_qwen", "qwen3", "qwq"):
    d = tier1_all[(tier1_all.model == model) & (tier1_all.family == "actions")]
    for arm in sorted(d.arm.unique(), key=lambda a: d[d.arm == a].signed_alpha.iloc[0]):
        if arm == "alpha0.0":
            continue
        for fr in ("real", "fake"):
            b, x = paired_frames(d, "alpha0.0", arm, fr); n, rb, rx, lost, gained = flip_counts(b, x)
            rows.append(dict(model=model, arm=arm, signed_alpha=d[d.arm == arm].signed_alpha.iloc[0], framing=fr, n=n, exec_baseline_paired=f"{rb:.3f}", exec_arm_paired=f"{rx:.3f}", lost=lost, gained=gained, p_mcnemar=fmt_p(mcnemar(lost, gained))))
    # floor: their sampled re-decode at alpha 0 (Distill only)
    if model == "deepseek_qwen":
        ds = tier1_all[tier1_all.model == model]
        for fr in ("real", "fake"):
            b = ds[(ds.arm_key == "actions/alpha0.0") & (ds.framing == fr)].set_index("index"); x = ds[(ds.arm_key == "actions_sampling/alpha0.0") & (ds.framing == fr)].set_index("index")
            idx = b.index.intersection(x.index); n, rb, rx, lost, gained = flip_counts(b.loc[idx], x.loc[idx])
            rows.append(dict(model=model, arm="floor: sampled vs greedy α=0", signed_alpha=0.0, framing=fr, n=n, exec_baseline_paired=f"{rb:.3f}", exec_arm_paired=f"{rx:.3f}", lost=lost, gained=gained, p_mcnemar=fmt_p(mcnemar(lost, gained))))
t3 = pd.DataFrame(rows)
save_table(t3, "T3_shipped_32b_edit_compliance", "[3] Abdelnabi & Salem's weight edit on their three 32B models: paired execution flips vs α = 0 (their shipped outputs, greedy)",
           "Positive α = toward awareness. (The comparison of these baselines with the paper's Table 4 is in results/tier1/SUMMARY.md, not recomputed here.)")

USED = {"deepseek_qwen": [0.07], "qwen3": [0.043, -0.32], "qwq": [0.05]}
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
floor = t3[t3.arm.str.startswith("floor")]
for ax, model in zip(axes, ("deepseek_qwen", "qwen3", "qwq")):
    for fr, c in (("real", "C0"), ("fake", "C1")):
        s_ = t3[(t3.model == model) & (t3.framing == fr) & (~t3.arm.str.startswith("floor"))].sort_values("signed_alpha")
        xs = [0.0] + s_.signed_alpha.tolist(); ys = [0] + (s_.gained.astype(int) - s_.lost.astype(int)).tolist()
        order = np.argsort(xs); ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color=c, label=f"{fr} framing")
        fl = floor[floor.framing == fr]
        if len(fl):
            band = abs(int(fl.gained.iloc[0]) - int(fl.lost.iloc[0])); ax.axhspan(-band, band, color=c, alpha=0.08)
    for u in USED[model]:
        ax.axvline(u, color="k", lw=0.6, ls="--"); ax.text(u, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5, " arm used\n in [4]–[6]", fontsize=7)
    ax.axhline(0, color="k", lw=0.8); ax.set_title({"deepseek_qwen": "DeepSeek-R1-Distill-Qwen-32B", "qwen3": "Qwen3-32B", "qwq": "QwQ-32B"}[model]); ax.set_xlabel("α (+ toward awareness, − away); 0 = unsteered"); ax.legend(fontsize=8)
axes[0].set_ylabel(f"net items gaining execution ({t3.n.min()}–{t3.n.max()} paired items)")
fig.suptitle("[3] Their edit moves the decision with a model-dependent sign; shaded = Distill's decode-to-decode floor (|net|), same on each panel"); fig.tight_layout(); fig.savefig(OUT / "F3_shipped_32b_edit_net_flips.png", dpi=130); plt.close(fig)

# %% [markdown]
# ## [4] Aggregate morphology rises are the flipped items (compliance-class split)

# %%
CELLS = [("Distill-Qwen-32B, aware +0.07", tier1[tier1.model == "deepseek_qwen"], "actions/alpha0.0", "actions/alpha0.07_aware", "arm_key"),
         ("QwQ-32B, aware +0.05", tier1[tier1.model == "qwq"], "actions/alpha0.0", "actions/alpha0.05_aware", "arm_key"),
         ("Qwen3-32B, aware +0.043", tier1[tier1.model == "qwen3"], "actions/alpha0.0", "actions/alpha0.043_aware", "arm_key"),
         ("Qwen3-32B, unaware −0.32", tier1[tier1.model == "qwen3"], "actions/alpha0.0", "actions/alpha0.32", "arm_key"),
         ("Qwen3-32B, test cue (seed 0)", api, "baseline", "prompt_test", "arm"),
         ("Qwen3-32B, test cue (seed 1)", api, "baseline_seed1", "prompt_test_seed1", "arm"),
         ("Qwen3-4B, test cue", q4b, "baseline", "prompt_test", "arm")]


def class_index(b, x):
    be, xe = b.executed.map(exec_str), x.executed.map(exec_str)
    return {"all": b.index, "refuse_both": b.index[(be == "False") & (xe == "False")], "gained": b.index[(be == "False") & (xe == "True")], "comply_both": b.index[(be == "True") & (xe == "True")]}


rows = []
for label, df, ba, ar, key in CELLS:
    for fr in ("real", "fake"):
        b = df[(df[key] == ba) & (df.framing == fr)].set_index("index"); x = df[(df[key] == ar) & (df.framing == fr)].set_index("index")
        assert b.index.is_unique and x.index.is_unique
        idx = b.index.intersection(x.index); b, x = b.loc[idx], x.loc[idx]
        for cls, ii in class_index(b, x).items():
            if len(ii) < 5:
                continue
            r = dict(cell=label, framing=fr, flip_class=cls, n=len(ii), words=f"{b.loc[ii].words.mean():.0f} → {x.loc[ii].words.mean():.0f}")
            for beh, short in (("backtracking", "backtracking"), ("example-testing", "example_testing"), ("uncertainty-estimation", "hedging_nontest"), ("deduction", "deduction_nontest")):
                nb, nx = nontest_count(b.loc[ii], beh), nontest_count(x.loc[ii], beh)
                r[short] = f"{nb.mean():.2f} → {nx.mean():.2f}"
            r["test_talk_added"] = f"{(testlex_total(x.loc[ii]) - testlex_total(b.loc[ii])).mean():+.2f}"
            rows.append(r)
t4 = pd.DataFrame(rows)
save_table(t4, "T4_morphology_by_compliance_class", "[4] Per-trace counts of non-test spans by compliance class: the aggregate rises live in the items that switched to complying",
           "Counts per trace (Venhoff annotation, gpt-4o-2024-08-06; spans matching the test-talk pattern excluded). Classes: refuse_both / comply_both hold the decision fixed; gained = refused at baseline, executed in the arm.")

rows = []
for label, df, ba, ar, key in CELLS:
    for fr in ("real", "fake"):
        b = df[(df[key] == ba) & (df.framing == fr)].set_index("index"); x = df[(df[key] == ar) & (df.framing == fr)].set_index("index")
        idx = b.index.intersection(x.index); b, x = b.loc[idx], x.loc[idx]
        for cls, ii in class_index(b, x).items():
            if cls == "all" or len(ii) < MIN_N_CI:
                continue
            for beh in ("backtracking", "example-testing", "uncertainty-estimation"):
                m, lo, hi = boot(nontest_count(x.loc[ii], beh) - nontest_count(b.loc[ii], beh))
                md, dlo, dhi = boot(x.loc[ii][f"density_{beh}_nontest"] - b.loc[ii][f"density_{beh}_nontest"])
                rows.append(dict(cell=label, framing=fr, flip_class=cls, n=len(ii), behaviour=beh, d_count=m, lo=lo, hi=hi, d_density=md, dlo=dlo, dhi=dhi))
f4d = pd.DataFrame(rows)
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")
for col, beh in enumerate(("backtracking", "example-testing", "uncertainty-estimation")):
    for row, (metric, lo_c, hi_c, unit) in enumerate((("d_count", "lo", "hi", "Δ spans per trace"), ("d_density", "dlo", "dhi", "Δ spans per 100 words"))):
        ax = axes[row, col]; sub = f4d[f4d.behaviour == beh]
        labels = [f"{c} [{fr}]" for c, fr in zip(sub.cell, sub.framing)]; uniq = list(dict.fromkeys(labels)); ypos = {l: i for i, l in enumerate(uniq)}
        for cls, c, off in (("refuse_both", "C0", -0.2), ("comply_both", "C2", 0.0), ("gained", "C3", 0.2)):
            ss = sub[sub.flip_class == cls]
            ys = [ypos[f"{c_} [{fr}]"] + off for c_, fr in zip(ss.cell, ss.framing)]
            ax.errorbar(ss[metric], ys, xerr=[ss[metric] - ss[lo_c], ss[hi_c] - ss[metric]], fmt="o", color=c, capsize=2, ms=4, label=cls if (row == 0 and col == 0) else None)
        ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(uniq))); ax.set_yticklabels(uniq, fontsize=7); ax.tick_params(labelleft=(col == 0)); ax.invert_yaxis()
        ax.set_title(f"{beh} (non-test spans)", fontsize=10); ax.set_xlabel(unit, fontsize=9)
axes[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("[4] Change vs baseline by compliance class (95% bootstrap CI; classes with < 20 items omitted): the rises sit in items that switched to complying"); fig.tight_layout(); fig.savefig(OUT / "F4_flipped_items_carry_the_rise.png", dpi=130); plt.close(fig)

# %% [markdown]
# ## [5] Task-directed reasoning at fixed decision: counts per trace with CIs, and the floor

# %%
CELLS5 = CELLS + [("Distill-Qwen-32B, aware +0.07, sampled vs sampled α=0", tier1[tier1.model == "deepseek_qwen"], "actions_sampling/alpha0.0", "actions_sampling/alpha0.07_aware", "arm_key"),
                                  ("floor: Distill-Qwen-32B sampled vs greedy α=0", tier1[tier1.model == "deepseek_qwen"], "actions/alpha0.0", "actions_sampling/alpha0.0", "arm_key"),
                                  ("floor: Qwen3-4B second baseline decode", q4b, "baseline", "baseline_seed1", "arm"),
                                  ("floor: Qwen3-32B baseline decodes", api, "baseline", "baseline_seed1", "arm")]
rows = []
for label, df, ba, ar, key in CELLS5:
    for fr in ("real", "fake"):
        b = df[(df[key] == ba) & (df.framing == fr)].set_index("index"); x = df[(df[key] == ar) & (df.framing == fr)].set_index("index")
        assert b.index.is_unique and x.index.is_unique
        idx = b.index.intersection(x.index); b, x = b.loc[idx], x.loc[idx]
        ii = stable_refusers(b, x)
        if len(ii) < 5:
            continue
        r = dict(cell=label, framing=fr, n_refuse_both=len(ii), words=f"{b.loc[ii].words.mean():.0f} → {x.loc[ii].words.mean():.0f}")
        for beh, short in (("deduction", "deduction"), ("adding-knowledge", "knowledge_recall"), ("uncertainty-estimation", "hedging"), ("backtracking", "backtracking")):
            nb, nx = nontest_count(b.loc[ii], beh), nontest_count(x.loc[ii], beh)
            m, lo, hi = boot(nx - nb); r[f"{short}_base"] = f"{nb.mean():.2f}"; r[f"{short}_delta"] = fmt_ci(m, lo, hi)
            m, lo, hi = boot(x.loc[ii][f"density_{beh}_nontest"] - b.loc[ii][f"density_{beh}_nontest"]); r[f"{short}_density_delta"] = fmt_ci(m, lo, hi)
        rows.append(r)
t5 = pd.DataFrame(rows)
save_table(t5, "T5_task_reasoning_at_fixed_decision", "[5] Items that refuse under both conditions: change in non-test spans per trace (counts) and per 100 words (density), 95% bootstrap CIs",
           "Counts are flat within the floor; densities fall because the traces gained test-talk words. '*' = CI excludes 0; with ~170 intervals in this table about 9 stars are expected by chance, so a single star is not evidence.")

def parse_ci(v):
    v = str(v)
    if "no CI" in v:
        return float(v.split()[0]), np.nan, np.nan
    a = [float(x) for x in v.replace("[", "").replace("]", "").replace(",", "").replace("*", "").split()]
    return a[0], a[1], a[2]


# test-talk Δ per cell (same items), for the third marker
tt = {}
for label, df, ba, ar, key in CELLS5:
    for fr in ("real", "fake"):
        b = df[(df[key] == ba) & (df.framing == fr)].set_index("index"); x = df[(df[key] == ar) & (df.framing == fr)].set_index("index")
        idx = b.index.intersection(x.index); b, x = b.loc[idx], x.loc[idx]; ii = stable_refusers(b, x)
        if len(ii) >= 5:
            tt[(label, fr)] = boot(testlex_total(x.loc[ii]) - testlex_total(b.loc[ii]))
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, fr in zip(axes, ("real", "fake")):
    sub = t5[t5.framing == fr].copy(); sub["is_floor"] = sub.cell.str.startswith("floor")
    sub = pd.concat([sub[~sub.is_floor], sub[sub.is_floor]]).reset_index(drop=True)
    for j, (col, c, name) in enumerate((("deduction_delta", "C0", "task deduction"), ("knowledge_recall_delta", "C2", "task knowledge recall"))):
        vals = [parse_ci(v) for v in sub[col]]
        for i, (m, lo, hi) in enumerate(vals):
            colr = "0.6" if sub.is_floor[i] else c
            if np.isnan(lo):
                ax.plot(m, i + (j - 1) * 0.22, "x", color=colr)
            else:
                ax.errorbar(m, i + (j - 1) * 0.22, xerr=[[m - lo], [hi - m]], fmt="o", color=colr, capsize=3, ms=4, label=name if i == 0 else None)
    for i, r in sub.iterrows():
        if (r.cell, fr) in tt:
            m, lo, hi = tt[(r.cell, fr)]
            if not np.isnan(lo):
                ax.errorbar(m, i + 0.22, xerr=[[m - lo], [hi - m]], fmt="s", color="0.6" if r.is_floor else "C3", capsize=3, ms=4, label="test-talk (any label)" if i == 0 else None)
    fl = sub[sub.is_floor & (sub.n_refuse_both >= MIN_N_CI)]
    band = max(abs(parse_ci(v)[0]) for col in ("deduction_delta", "knowledge_recall_delta") for v in fl[col]) if len(fl) else float("nan")
    ax.axvspan(-band, band, color="k", alpha=0.05)
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(sub))); ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(sub.cell, sub.n_refuse_both)], fontsize=8); ax.invert_yaxis()
    ax.set_title(f"{fr} framing (grey band: ±{band:.2f} spans = largest floor mean, floors with n ≥ {MIN_N_CI})", fontsize=10); ax.set_xlabel("Δ spans per trace vs baseline (95% CI; × = n < 20, no CI)")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("[5]–[6] Items refusing in both arms: task-directed reasoning is flat, about one test-talk span is added (grey rows = decode-to-decode floors)"); fig.tight_layout(); fig.savefig(OUT / "F5_task_reasoning_fixed_decision.png", dpi=130); plt.close(fig)

# %% [markdown]
# ## [6] Test-talk added per trace at fixed decision

# %%
rows = []
for label, df, ba, ar, key in CELLS5:
    for fr in ("real", "fake"):
        b = df[(df[key] == ba) & (df.framing == fr)].set_index("index"); x = df[(df[key] == ar) & (df.framing == fr)].set_index("index")
        assert b.index.is_unique and x.index.is_unique
        idx = b.index.intersection(x.index); b, x = b.loc[idx], x.loc[idx]; ii = stable_refusers(b, x)
        if len(ii) < 5:
            continue
        m, lo, hi = boot(testlex_total(x.loc[ii]) - testlex_total(b.loc[ii]))
        rows.append(dict(cell=label, framing=fr, n_refuse_both=len(ii), test_talk_base=f"{testlex_total(b.loc[ii]).mean():.2f}", test_talk_delta=fmt_ci(m, lo, hi),
                         hedged_form_delta=fmt_ci(*boot(x.loc[ii]["n_uncertainty-estimation_testlex"] - b.loc[ii]["n_uncertainty-estimation_testlex"])),
                         deductive_form_delta=fmt_ci(*boot(x.loc[ii].n_deduction_testlex - b.loc[ii].n_deduction_testlex))))
save_table(pd.DataFrame(rows), "T6_test_talk_added", "[6] Test-talk spans per trace added in items that refuse under both conditions",
           "Test-talk = spans matching d10.parse.TEST_LEXICON (any Venhoff label); split by whether the annotator labelled the span uncertainty-estimation (hedged form) or deduction.")

# %% [markdown]
# ## [7] F4: reasoning-direction steering vs verbalized awareness (length-adjusted), with random controls

# %%
from sklearn.linear_model import LogisticRegression  # noqa: E402

f4r = f4[f4.framing == "real"].copy(); f4r["aw"] = f4r.aware_judged.isin(["Yes", "Maybe"]).astype(int); f4r["lw"] = np.log(f4r.words.clip(lower=20))
base = f4r[f4r.arm.isin(["baseline", "baseline_seed1"])]
lr = LogisticRegression(C=np.inf).fit(base[["lw"]], base.aw)
rows = []
for arm in ["baseline", "baseline_seed1", "bt_neg", "unc_neg", "unc_pos", "bt_pos", "random_bt", "random_unc"]:
    a = f4r[f4r.arm == arm]
    if len(a) == 0:
        continue
    closure = f4_all[(f4_all.arm == arm) & (f4_all.framing == "real")].has_think_close.mean(); rep = a.repetition_4gram.mean()
    rows.append(dict(arm=arm, n=len(a), words=f"{a.words.mean():.0f}", closure=f"{closure:.2f}", repetition=f"{rep:.3f}", verbalization=f"{a.aw.mean():.3f}",
                     expected_from_length=f"{lr.predict_proba(a[['lw']])[:, 1].mean():.3f}", test_talk_per_trace=f"{testlex_total(a).mean():.2f}", test_talk_per_100w=f"{100 * testlex_total(a).sum() / a.words.sum():.3f}",
                     hedged_test_talk_per_100w=f"{a['density_uncertainty-estimation_testlex'].mean():.3f}", deductive_test_talk_per_100w=f"{a.density_deduction_testlex.mean():.3f}"))
t7 = pd.DataFrame(rows)
save_table(t7, "T7_f4_verbalization", "[7] Qwen3-4B, residual-stream steering (±0.25) along the rebuilt backtracking (L23) and uncertainty (L18) directions, real framing: verbalized awareness per arm",
           f"Verbalization = Abdelnabi & Salem's judge (gpt-4o-2024-08-06) says awareness Yes/Maybe. 'expected_from_length' = a logistic fit of verbalization on log-length in the two baseline decodes, evaluated at the arm's lengths. random_bt / random_unc = Gaussian directions of matched norm at the same layers. bt_pos is degenerate (closure, repetition). Traces the judge failed on ({int(f4r.aware_judged.isna().sum())} of {len(f4r)}) count as not verbalizing. test_talk_per_100w is pooled (sum of spans / sum of words).")

def wilson(k, n):
    ci = binomtest(int(k), int(n), 0.5).proportion_ci(0.95, method="wilson")
    return ci.low, ci.high


ARMS7 = ["baseline", "baseline_seed1", "random_bt", "random_unc", "bt_neg", "unc_neg", "unc_pos", "bt_pos"]
NICE = {"baseline": "baseline", "baseline_seed1": "baseline, 2nd decode", "random_bt": "random dir. (L23)", "random_unc": "random dir. (L18)", "bt_neg": "backtracking −", "unc_neg": "uncertainty −", "unc_pos": "uncertainty +", "bt_pos": "backtracking + (degenerate)"}
t7i = t7.set_index("arm").reindex(ARMS7).dropna(subset=["n"]); ARMS7 = list(t7i.index)
fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
xs = np.arange(len(ARMS7)); ax = axes[0]
v = t7i.verbalization.astype(float); n = t7i.n.astype(int)
cis = [wilson(round(vv * nn), nn) for vv, nn in zip(v, n)]; lo = np.array([c[0] for c in cis]); hi = np.array([c[1] for c in cis])
ax.bar(xs - 0.2, v, 0.4, yerr=[v - lo, hi - v], capsize=3, label="observed (Wilson 95% CI)")
ax.bar(xs + 0.2, t7i.expected_from_length.astype(float), 0.4, color="0.7", label="expected from trace length alone (logistic fit on the two baseline decodes)")
ax.bar(xs[-1] - 0.2, v.iloc[-1], 0.4, color="none", edgecolor="r", hatch="//")
ax.set_ylabel("share of traces the judge calls test-aware"); ax.legend(fontsize=8); ax.set_title("[7] Judged 'test awareness' per arm (Qwen3-4B, 149 real-framed items)")
ax = axes[1]
b0, x0 = paired_frames(f4_all, "baseline", "baseline_seed1", "real"); _, base_rate, _, _, _ = flip_counts(b0, x0)  # baseline rate over the items paired with its second decode
ex, flips = [base_rate], [""]
for a in ARMS7[1:]:
    b, x = paired_frames(f4_all, "baseline", a, "real"); _, _, rx, lost, gained = flip_counts(b, x); ex.append(rx); flips.append(f"{lost}↓ {gained}↑")
ax.bar(xs, ex, 0.6, color=["0.4"] + ["0.6"] * 3 + ["C1"] * 4)
rb = [ex[ARMS7.index("random_bt")], ex[ARMS7.index("random_unc")]]
ax.axhspan(min(base_rate, min(rb)), max(rb), color="C0", alpha=0.12, label="range spanned by baseline and the two random directions")
ax.axhline(base_rate, color="k", lw=0.8)
for i in range(1, len(ARMS7)):
    ax.text(i, ex[i] + 0.01, flips[i], ha="center", fontsize=7)
ax.set_ylim(0.5, 0.9); ax.set_ylabel("execution rate\n(items paired with baseline)"); ax.set_xticks(xs); ax.set_xticklabels([NICE[a] for a in ARMS7], rotation=20, ha="right", fontsize=8); ax.legend(fontsize=8, loc="upper left")
ax.set_title("[8] Execution in the same arms: inside the band that random directions of the same norm produce")
fig.tight_layout(); fig.savefig(OUT / "F7_f4_verbalization_and_execution.png", dpi=130); plt.close(fig)

# %% [markdown]
# ## [8] F4: execution per arm

# %%
rows = []
for arm in ["baseline_seed1", "bt_neg", "unc_neg", "unc_pos", "bt_pos", "random_bt", "random_unc"]:
    b, x = paired_frames(f4_all, "baseline", arm, "real")
    if len(x) == 0:
        continue
    n, rb, rx, lost, gained = flip_counts(b, x)
    rows.append(dict(arm=arm, n_known=n, exec_baseline_paired=f"{rb:.3f}", exec_arm_paired=f"{rx:.3f}", lost=lost, gained=gained, p_mcnemar=fmt_p(mcnemar(lost, gained)), closure=f"{f4_all[(f4_all.arm == arm) & (f4_all.framing == 'real')].has_think_close.mean():.2f}"))
save_table(pd.DataFrame(rows), "T8_f4_execution", "[8] Same F4 arms: execution vs the main-run baseline, paired by item (real framing)",
           "baseline_seed1 is the decode-to-decode floor. bt_pos is not interpretable (closure 0.63).")

# %% [markdown]
# ## [9] Instrument facts: judge evidence quotes in the steered arms, probe layer-0 AUROC

# %%
random.seed(0)
lines = ["# [9] What the awareness judge quotes as evidence in the F4 arms (random sample of positives)", ""]
for arm in ["unc_pos", "bt_neg", "unc_neg"]:
    recs = load_results(REPO / "results/qwen3_4b/followups/f4_reverse" / f"{arm}.json.awareness.jsonl")
    pos = [(k, parse_judgment(v["text"])) for k, v in recs.items() if record_ok(v) and is_positive(parse_judgment(v["text"]))]
    lines.append(f"## {arm}: {len(pos)} positives of {len(recs)}")
    for k, j in random.sample(pos, min(5, len(pos))):
        ev = j["model_awareness"].get("evidence") or []
        lines.append(f"- item {k.split('/')[-2]} [{j['model_awareness']['conclusion']}]: " + " | ".join(f"\"{e}\"" for e in ev[:2]))
    lines.append("")
pm = json.loads((REPO / "results/qwen3_4b/probe/probe_metrics.json").read_text())
lines += ["## Awareness probe on Qwen3-4B (authors' recipe, span-mean pooling)", "",
          f"- best hidden-state index {pm['best_layer']}: accuracy {pm['best_acc']:.3f}, AUROC {pm['best_auroc']:.3f}",
          f"- layer 0 (token embeddings only): AUROC {pm['controls']['layer0_auroc']:.3f}",
          f"- shuffled-label control at the best layer: AUROC {pm['controls']['shuffled_auroc']:.3f} (multi-seed check in results/qwen3_4b/probe_robustness/)"]
(OUT / "T9_instrument_facts.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

print(f"\nwrote {OUT}")
