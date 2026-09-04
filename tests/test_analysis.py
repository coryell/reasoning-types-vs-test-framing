"""Synthetic-data tests for d10.analysis: the numbers must come out as constructed."""

import math

import numpy as np
import pandas as pd
import pytest

from d10 import analysis as A
from d10.parse import LABELS


def _rows(model, family, framing, arm, signed_alpha, n, rng, executed=None, aware_judged=None, shift=None, judge_fail_idx=()):
    rows = []
    for i in range(n):
        r = {
            "id": f"{model}/{family}/{arm}/{i}/{framing}",
            "model": model,
            "family": family,
            "arm": arm,
            "alpha": abs(signed_alpha),
            "aware": signed_alpha > 0,
            "signed_alpha": signed_alpha,
            "index": i,
            "framing": framing,
            "has_think_close": True,
            "executed": None if executed is None else bool(executed[i]),
            "aware_judged": None if aware_judged is None else aware_judged[i],
            "recog_judged": None,
            "task_judged": None,
            "annotated": True,
            "ok": i not in judge_fail_idx,
            "judge_fail": i in judge_fail_idx,
            "error": None,
            "finish_reason": "stop",
            "served_model": "test",
            "words": 200,
            "n_spans": 12,
            "n_spans_counted": 12,
            "n_spans_unknown_label": 0,
            "n_spans_not_found_exact": 0,
            "n_spans_found_norm_only": 0,
            "n_spans_not_found": 0,
            "labeled_words": 180,
            "labeled_fraction": 0.9,
            "repetition_4gram": 0.01,
            "unknown_labels": "",
        }
        for b in LABELS:
            base = 2.0 + 0.3 * rng.standard_normal()
            base += (shift or {}).get(b, 0.0)
            r[f"density_{b}"] = base
            r[f"coverage_{b}"] = 1 / len(LABELS)
            r[f"any_{b}"] = 1
            r[f"n_{b}"] = 6
            r[f"n_{b}_testlex"] = 0
            r[f"density_{b}_testlex"] = 0.0
            r[f"density_{b}_nontest"] = base
        for k in ("wait", "hmm", "backtrack_lex", "uncertainty_lex"):
            r[f"lex_{k}_n"] = 1
            r[f"lex_{k}_per100w"] = r["density_backtracking"] * 0.5 + 0.1 * rng.standard_normal()
        rows.append(r)
    return rows


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    n = 20
    base_exec = [True] * 10 + [False] * 10
    # aware arm: lose 3 of the first 10, gain 5 of the last 10; backtracking density +1, uncertainty -1
    aware_exec = [True] * 7 + [False] * 3 + [True] * 5 + [False] * 5
    rows = []
    rows += _rows("m", "actions", "real", "alpha0.0", 0.0, n, rng, executed=base_exec)
    rows += _rows("m", "actions", "real", "alpha0.05_aware", 0.05, n, rng, executed=aware_exec, shift={"backtracking": 1.0, "uncertainty-estimation": -1.0})
    rows += _rows("m", "actions", "real", "alpha0.25", -0.25, n, rng, executed=base_exec, shift={"backtracking": -1.0, "uncertainty-estimation": 1.0})
    rows += _rows("m", "triggers", "hypothetical", "alpha0.0", 0.0, n, rng, aware_judged=["Yes"] * 8 + ["No"] * 12)
    return pd.DataFrame(rows)


def test_mcnemar_exact():
    assert math.isnan(A.mcnemar_exact(0, 0))
    assert A.mcnemar_exact(3, 5) == pytest.approx(0.7265625)
    assert A.mcnemar_exact(0, 10) == pytest.approx(2 * 0.5**10)
    assert A.mcnemar_exact(10, 0) == A.mcnemar_exact(0, 10)


def test_boot_ci_basic():
    assert A.boot_ci([3.0, 3.0, 3.0]) == (3.0, 3.0)
    lo, hi = A.boot_ci(np.arange(100, dtype=float))
    assert lo < 49.5 < hi
    assert all(math.isnan(v) for v in A.boot_ci([1.0]))
    assert all(math.isnan(v) for v in A.boot_ci([np.nan, np.nan]))


def test_paired_contrasts_recover_constructed_shift(df):
    c = A.paired_contrasts(df)
    bt = c[(c.metric == "density") & (c.behaviour == "backtracking") & (c.arm == "alpha0.05_aware")].iloc[0]
    assert bt.n_pairs == 20
    assert bt.delta == pytest.approx(1.0, abs=0.35)
    assert bt.ci_lo > 0.3 and bt.ci_hi < 1.7
    assert bt.p_paired_t < 0.01
    un = c[(c.metric == "density") & (c.behaviour == "uncertainty-estimation") & (c.arm == "alpha0.05_aware")].iloc[0]
    assert un.delta == pytest.approx(-1.0, abs=0.35)
    ded = c[(c.metric == "density") & (c.behaviour == "deduction") & (c.arm == "alpha0.05_aware")].iloc[0]
    assert ded.ci_lo < 0 < ded.ci_hi
    # words and span counts are contrasted too
    assert set(c[c.metric == "words"].arm) == {"alpha0.05_aware", "alpha0.25"}


def test_sign_symmetry_flags_opposite_signs(df):
    sym = A.sign_symmetry(A.paired_contrasts(df))
    row = sym[sym.behaviour == "backtracking"].iloc[0]
    assert row.aware_arm == "alpha0.05_aware" and row.unaware_arm == "alpha0.25"
    assert row.d_aware > 0 > row.d_unaware and bool(row.opposite_sign)
    assert bool(row.aware_ci_excludes_0) and bool(row.unaware_ci_excludes_0)
    ded = sym[sym.behaviour == "deduction"].iloc[0]
    assert not bool(ded.aware_ci_excludes_0)


def test_flips_two_by_two_and_classes(df):
    two, by_class = A.flips(df)
    r = two[two.arm == "alpha0.05_aware"].iloc[0]
    assert (r.both_exec, r.lost, r.gained, r.neither) == (7, 3, 5, 5)
    assert r.net_gain == 2 and r.churn == 8
    assert r.p_mcnemar == pytest.approx(0.7265625)
    assert r.base_rate == pytest.approx(0.5) and r.arm_rate == pytest.approx(0.6)
    # unaware arm: identical execution → no discordant pairs
    u = two[two.arm == "alpha0.25"].iloc[0]
    assert u.churn == 0 and math.isnan(u.p_mcnemar)
    # classes with n >= MIN_N appear (gained=5, comply_both=7, refuse_both=5, same=12); lost=3 is below the floor
    k = by_class[by_class.arm == "alpha0.05_aware"]
    assert set(k.flip_class) == {"gained", "comply_both", "refuse_both", "same"}
    n = k.groupby("flip_class").n.first()
    assert n["gained"] == 5 and n["comply_both"] == 7 and n["refuse_both"] == 5 and n["same"] == 12
    # triggers rows (executed is None) must not enter the flip analysis
    assert set(two.family) == {"actions"}


def test_verbalization_splits_by_shipped_judgment(df):
    v = A.verbalization(df)
    assert set(v.family) == {"triggers"}
    n = v.groupby("aware_class").n.first()
    assert n["yes_or_maybe"] == 8 and n["no"] == 12


def test_arm_means_and_coverage_shapes(df):
    means = A.arm_means(df)
    d = means[(means.metric == "density") & (means.arm == "alpha0.0") & (means.family == "actions")]
    assert len(d) == len(LABELS)
    assert (d.n == 20).all()
    assert (d.ci_lo <= d["mean"]).all() and (d["mean"] <= d.ci_hi).all()
    assert "executed" in set(means.metric)
    cov = A.coverage_table(df)
    assert len(cov) == 4 and (cov.n == 20).all() and (cov.n_ok == 20).all()


def test_proxy_vs_judge_correlates_with_constructed_proxy(df):
    p = A.proxy_vs_judge(df)
    r = p[(p.scope == "all") & (p.proxy == "lex_wait_per100w")].iloc[0]
    assert r.pearson_r > 0.7


def test_load_traces_skips_missing_dir(tmp_path):
    assert A.load_traces(tmp_path, None, None).empty


def test_judge_failures_are_counted_and_excluded():
    rng = np.random.default_rng(2)
    rows = _rows("m", "actions", "real", "alpha0.0", 0.0, 20, rng, executed=[True] * 20, judge_fail_idx=(0, 1, 2))
    df = pd.DataFrame(rows)
    cov = A.coverage_table(df).iloc[0]
    assert (cov.n, cov.n_annotated, cov.n_ok, cov.n_judge_fail) == (20, 20, 17, 3)
    means = A.arm_means(df)
    assert (means.n == 17).all()


def test_load_traces_from_jsonl_applies_ok_rules(tmp_path):
    import json

    d = tmp_path / "m" / "actions"
    d.mkdir(parents=True)
    meta = {"model": "m", "family": "actions", "arm": "alpha0.0", "alpha": 0.0, "aware": False, "signed_alpha": 0.0,
            "index": 0, "framing": "real", "task": "t", "reasoning": "Okay. Wait, no.", "reasoning_words": 3,
            "has_think_close": True, "tool": "x", "executed": True, "judge": None}
    recs = [
        {"id": "ok", "text": '["initializing"]Okay.["end-section"]', "error": None, "finish_reason": "stop", "meta": meta},
        {"id": "trunc", "text": '["initializing"]Okay.["end-section"]', "error": "finish_reason=length", "finish_reason": "length", "meta": {**meta, "index": 1}},
        {"id": "nomarkers", "text": "Sorry, I cannot.", "error": None, "finish_reason": "stop", "meta": {**meta, "index": 2}},
        {"id": "apierr", "text": None, "error": "HTTP 500", "finish_reason": None, "meta": {**meta, "index": 3}},
    ]
    (d / "alpha0.0.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    df = A.load_traces(tmp_path, None, None).set_index("id")
    assert df.loc["ok", "ok"] and not df.loc["ok", "judge_fail"]
    assert not df.loc["trunc", "ok"] and df.loc["trunc", "annotated"]
    assert not df.loc["nomarkers", "ok"] and df.loc["nomarkers", "judge_fail"]
    assert not df.loc["apierr", "ok"] and not df.loc["apierr", "annotated"]
    cov = A.coverage_table(df.reset_index()).iloc[0]
    assert (cov.n, cov.n_annotated, cov.n_ok, cov.n_judge_fail, cov.n_error, cov.n_truncated) == (4, 3, 1, 1, 2, 1)


def test_noise_floor_pairs_greedy_with_sampled_same_alpha():
    rng = np.random.default_rng(3)
    rows = _rows("deepseek_qwen", "actions", "real", "alpha0.0", 0.0, 20, rng, executed=[True] * 20)
    rows += _rows("deepseek_qwen", "actions_sampling", "real", "alpha0.0", 0.0, 20, rng, executed=[True] * 20)
    rows += _rows("deepseek_qwen", "actions_sampling", "real", "alpha0.35", -0.35, 20, rng, executed=[True] * 20)  # no greedy partner
    nf = A.noise_floor(pd.DataFrame(rows))
    assert set(nf.arm) == {"alpha0.0"} and (nf.n_pairs == 20).all()
    bt = nf[nf.behaviour == "backtracking"].iloc[0]
    assert abs(bt.mean_delta) < 0.3 and 0.2 < bt.sd_delta < 0.8
    assert "words" in set(nf.behaviour)


def test_verbalization_contrasts_restrict_to_same_class_in_both_arms():
    rng = np.random.default_rng(4)
    cls = ["Yes"] * 8 + ["No"] * 12
    rows = _rows("qwq", "triggers", "hypothetical", "alpha0.0", 0.0, 20, rng, aware_judged=cls)
    rows += _rows("qwq", "triggers", "hypothetical", "alpha0.05_aware", 0.05, 20, rng, aware_judged=["Yes"] * 12 + ["No"] * 8, shift={"backtracking": 1.0})
    vc = A.verbalization_contrasts(pd.DataFrame(rows))
    n = vc.groupby("aware_class_both").n_pairs.first()
    assert n["yes"] == 8 and n["no"] == 8  # items 0-7 yes in both; items 12-19 no in both
    assert vc[(vc.aware_class_both == "no") & (vc.behaviour == "backtracking")].delta_density.iloc[0] > 0.5


def test_flips_two_by_two_uses_all_known_execution_but_deltas_use_ok_only():
    rng = np.random.default_rng(5)
    base_exec = [True] * 10 + [False] * 10
    rows = _rows("m", "actions", "real", "alpha0.0", 0.0, 20, rng, executed=base_exec)
    rows += _rows("m", "actions", "real", "alpha0.05_aware", 0.05, 20, rng, executed=[False] * 20, judge_fail_idx=range(0, 10))
    two, by_class = A.flips(pd.DataFrame(rows))
    r = two.iloc[0]
    assert (r.n_pairs, r.lost, r.gained) == (20, 10, 0)  # all 20 items count in the 2x2
    assert set(by_class.flip_class) == {"refuse_both", "same"}  # lost items were judge failures; the stable items all refuse
    assert by_class.groupby("flip_class").n.first().to_dict() == {"refuse_both": 10, "same": 10}


def test_paired_contrasts_include_testlex_decomposition():
    rng = np.random.default_rng(6)
    rows = _rows("m", "actions", "real", "alpha0.0", 0.0, 20, rng, executed=[True] * 20)
    rows += _rows("m", "actions", "real", "alpha0.05_aware", 0.05, 20, rng, executed=[True] * 20, shift={"uncertainty-estimation": 1.0})
    d2 = pd.DataFrame(rows)
    # in the aware arm, put 0.6 of each uncertainty span-density into the test-language bucket
    aw = d2.arm == "alpha0.05_aware"
    d2.loc[aw, "density_uncertainty-estimation_testlex"] = 0.6
    d2.loc[aw, "density_uncertainty-estimation_nontest"] = d2.loc[aw, "density_uncertainty-estimation"] - 0.6
    c = A.paired_contrasts(d2)
    assert {"density_testlex", "density_nontest"} <= set(c.metric)
    pick = lambda metric: c[(c.metric == metric) & (c.behaviour == "uncertainty-estimation") & (c.arm == "alpha0.05_aware")].iloc[0]  # noqa: E731
    assert pick("density_testlex").delta == pytest.approx(0.6)
    assert pick("density_testlex").delta + pick("density_nontest").delta == pytest.approx(pick("density").delta)
    assert pick("density_nontest").delta == pytest.approx(0.4, abs=0.35)


def test_testlex_share_and_recall():
    rng = np.random.default_rng(7)
    rows = _rows("qwq", "triggers", "hypothetical", "alpha0.0", 0.0, 20, rng, aware_judged=["Yes"] * 10 + ["No"] * 10)
    d = pd.DataFrame(rows)
    d["n_uncertainty-estimation_testlex"] = [3] * 10 + [0] * 10
    d["n_uncertainty-estimation"] = 6
    d["aw_evidence_n"] = [2] * 10 + [0] * 10
    d["aw_evidence_caught"] = [1] * 10 + [0] * 10
    s = A.testlex_share(d)
    u = s[s.behaviour == "uncertainty-estimation"].set_index("aware_class")
    assert u.loc["yes_or_maybe", "testlex_share"] == pytest.approx(0.5) and u.loc["no", "testlex_share"] == 0.0
    assert A.lexicon_recall(d) == {"evidence_quotes": 20, "caught": 10, "recall": 0.5}


def test_flips_split_outcome_stable_items_by_decision():
    rng = np.random.default_rng(11)
    base_exec = [True] * 10 + [False] * 10
    rows = _rows("m", "actions", "real", "alpha0.0", 0.0, 20, rng, executed=base_exec)
    rows += _rows("m", "actions", "real", "alpha0.05_aware", 0.05, 20, rng, executed=base_exec, shift={"deduction": -1.0})
    _, by_class = A.flips(pd.DataFrame(rows))
    n = by_class.groupby("flip_class").n.first()
    assert n["comply_both"] == 10 and n["refuse_both"] == 10 and n["same"] == 20 and "gained" not in n
    r = by_class[(by_class.flip_class == "refuse_both") & (by_class.behaviour == "deduction") & (by_class.metric == "density")].iloc[0]
    assert r.delta_density == pytest.approx(-1.0, abs=0.4)
    assert {"density", "density_nontest", "density_testlex"} <= set(by_class.metric)
