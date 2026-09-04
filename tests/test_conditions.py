"""Condition plumbing shared by the follow-up generator and the readouts."""

import json

import pytest
import torch

from d10 import conditions as C
from tests.test_tier_m import TinyModel


def test_roundtrip_and_signed():
    c = C.Condition("x", kind="weight_edit", direction="probe", alpha=0.2, aware=False)
    d = json.loads(json.dumps(c.to_dict()))
    assert C.Condition.from_dict(d) == c
    assert c.signed == (0.2, False)
    r = C.Condition("y", kind="residual_add", direction="backtracking", layer=3, coefficient=-1.0)
    assert r.signed == (1.0, False)
    assert C.Condition("b").signed == (0.0, True)
    p = C.Condition("p", kind="prompt", prompt_suffix=C.TEST_SCENARIO_SENTENCE)
    assert p.signed == (0.0, True)
    assert p.system_prompt("AGENT") == "AGENT\n\n" + C.TEST_SCENARIO_SENTENCE
    assert p.system_prompt(None) == C.TEST_SCENARIO_SENTENCE
    assert r.system_prompt("AGENT") == "AGENT"


def test_validation():
    with pytest.raises(ValueError):
        C.Condition("bad", kind="weight_edit")
    with pytest.raises(ValueError):
        C.Condition("bad", kind="residual_add", direction="backtracking")
    with pytest.raises(ValueError):
        C.Condition("bad", kind="prompt")
    with pytest.raises(ValueError):
        C.Condition("bad", kind="nope")


def test_legacy_arm_names():
    assert C.legacy_condition("baseline", 0.0, True).kind == "none"
    assert C.legacy_condition("baseline_seed1", 0.0, True).kind == "none"
    a = C.legacy_condition("aware_strong", 0.2, True)
    assert (a.kind, a.direction, a.alpha, a.aware, a.rows_from) == ("weight_edit", "probe", 0.2, True, "edit direction")
    u = C.legacy_condition("unaware", 0.1, False)
    assert (u.direction, u.aware) == ("probe", False)
    assert C.legacy_condition("random_aware", 0.05, True).direction == "random"
    s = C.legacy_condition("shuffled_samerows_aware", 0.05, True)
    assert (s.direction, s.rows_from) == ("shuffled", "probe direction")
    assert C.legacy_condition("random_samerows_aware", 0.05, True).rows_from == "probe direction"


def test_condition_from_sidecar_prefers_serialised(tmp_path):
    c = C.Condition("prompt_test", kind="prompt", prompt_suffix="hi")
    side = {"alpha": 0.0, "aware": True, "condition": c.to_dict(), "config": {"seed": 3}}
    assert C.condition_from_sidecar(tmp_path / "prompt_test.json", side) == c
    legacy = C.condition_from_sidecar(tmp_path / "aware.json", {"alpha": 0.05, "aware": True, "config": {"seed": 3}})
    assert legacy.kind == "weight_edit" and legacy.seed == 3


def test_applied_weight_edit_undoes():
    torch.manual_seed(0)
    m = TinyModel()
    probe = {"direction": [1.0, 0.0, 0.0, 0.0], "shuffled_direction": [0.0, 2.0, 0.0, 0.0]}
    before = [l.mlp.gate_proj.weight.detach().clone() for l in m.model.layers]
    c = C.Condition("aware", kind="weight_edit", direction="probe", alpha=0.5, aware=True)
    with C.applied(m, c, probe=probe):
        assert c.info["rows_edited"] == 18  # top-800 capped by 3 layers x 6 rows
        assert any(not torch.equal(l.mlp.gate_proj.weight, b) for l, b in zip(m.model.layers, before))
    assert all(torch.equal(l.mlp.gate_proj.weight, b) for l, b in zip(m.model.layers, before))
    # the three directions: shuffled is norm-matched to the probe, random is seeded
    v = C.probe_vectors(probe, seed=1)
    assert torch.isclose(v["shuffled"].norm(), v["probe"].norm())
    assert torch.equal(v["random"], C.probe_vectors(probe, seed=1)["random"])
    assert not torch.equal(v["random"], C.probe_vectors(probe, seed=2)["random"])
    # same-rows control edits the rows the probe direction selects
    r = C.Condition("random_samerows_aware", kind="weight_edit", direction="random", alpha=0.5, aware=True, rows_from="probe direction")
    with C.applied(m, r, probe=probe):
        pass
    assert all(torch.equal(l.mlp.gate_proj.weight, b) for l, b in zip(m.model.layers, before))


def test_residual_vector_matches_venhoff_recipe(tmp_path):
    overall = torch.ones(4, 3) * 2.0
    bt = torch.ones(4, 3) * 2.0
    bt[1] = torch.tensor([3.0, 2.0, 2.0])
    mv = {"overall": {"mean": overall}, "backtracking": {"mean": bt}}
    torch.save(mv, tmp_path / "mean_vectors.pt")
    c = C.Condition("bt_neg", kind="residual_add", direction="backtracking", vectors="mean_vectors", layer=1, coefficient=-1.0)
    v = C.residual_vector(c, tmp_path)
    # (mean - overall) rescaled to ||overall|| at layer 1, times the coefficient
    expected = -torch.tensor([1.0, 0.0, 0.0]) * overall[1].norm()
    assert torch.allclose(v, expected)


def test_followup_experiment_specs():
    import importlib.util, pathlib, sys
    scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("run_followup", scripts / "run_followup.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    layers = {"uncertainty-estimation": 18, "backtracking": 23}
    for name in ("f1_prompt", "f2_uncertainty", "f3_triggers", "f4_reverse"):
        e = rf.experiment(name, layers, 0.2)
        names = [c.name for c in e["conds"]]
        assert len(names) == len(set(names))
        assert e["task"] in ("actions", "triggers")
    f2 = rf.experiment("f2_uncertainty", layers, 0.2)
    assert [c.name for c in f2["conds"]] == ["baseline", "unc_all_pos", "unc_all_neg", "unc_nontest_pos", "unc_nontest_neg", "unc_testlex_pos", "unc_testlex_neg"]
    assert all(c.layer == 18 for c in f2["conds"][1:]) and f2["conds"][2].coefficient == -1.0 and f2["conds"][3].vectors == "mean_vectors_nontest"
    f3 = rf.experiment("f3_triggers", layers, 0.2)
    assert f3["conds"][2].aware is False and f3["conds"][3].direction == "random" and f3["conds"][1].alpha == 0.2
    f4 = rf.experiment("f4_reverse", layers, 0.2)
    assert f4["framings"] == ["real"] and f4["conds"][0].layer == 23 and f4["conds"][2].direction == "uncertainty-estimation"
    half = rf.experiment("f2_uncertainty", layers, 0.2, coefficient=0.5)
    assert [c.coefficient for c in half["conds"][1:]] == [0.5, -0.5] * 3
    assert rf.experiment("f4_reverse", layers, 0.2, coefficient=-0.25)["conds"][1].coefficient == -0.25
    with pytest.raises(SystemExit):
        rf.experiment("nope", layers, 0.2)
    # held-out trigger selection is disjoint from the gate + probe sets (offsets 0-600 under seed 0)
    held = rf.trigger_entries(5, rf.HELD_OUT_OFFSET, rf.SELECTION_SEED)
    used = {e["source_index"] for e in rf.trigger_entries(600, 0, 0)}
    assert rf.SELECTION_SEED == 0 and len(held) == 5 and not ({h["source_index"] for h in held} & used)


def test_applied_matches_run_steering_path():
    """The Condition path must edit exactly the weights run_steering.py's inline code edits."""
    from d10.steer import apply_edit, random_direction_like, rank_rows, undo_edit
    torch.manual_seed(1)
    probe = {"direction": [0.3, -1.0, 0.2, 0.5], "shuffled_direction": [1.0, 1.0, -1.0, 0.0]}
    v_pos = torch.tensor(probe["direction"])
    v_shuf = torch.tensor(probe["shuffled_direction"]); v_shuf = v_shuf / v_shuf.norm() * v_pos.norm()
    rnd = random_direction_like(v_pos, seed=0)
    for arm, vec, alpha, aware in (("unaware", v_pos, 0.1, False), ("random_aware", rnd, 0.05, True), ("shuffled_samerows_aware", v_shuf, 0.05, True)):
        m1, m2 = TinyModel(), TinyModel()
        m2.load_state_dict(m1.state_dict())
        rows = rank_rows(m1, v_pos) if arm.endswith("_samerows_aware") else None
        edit = apply_edit(m1, vec, alpha, aware, rows=rows)
        cond = C.legacy_condition(arm, alpha, aware)
        with C.applied(m2, cond, probe=probe):
            assert all(torch.equal(a.mlp.gate_proj.weight, b.mlp.gate_proj.weight) for a, b in zip(m1.model.layers, m2.model.layers)), arm
        undo_edit(m1, edit)


def test_api_prompt_serialisation():
    import importlib.util, pathlib, sys, types
    scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("run_api_prompt", scripts / "run_api_prompt.py")
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    from d10.shipped import split_think

    msg = types.SimpleNamespace(content="  Final answer.", model_extra={"reasoning": "  I think.\n"})
    text, src = ra.serialise(msg)
    assert src == "reasoning_field" and text.startswith("<think>\nI think.\n</think>\n\nFinal answer.")
    reasoning, answer, closed = split_think(text)
    assert closed and reasoning.strip() == "I think." and answer.strip() == "Final answer."
    inline = types.SimpleNamespace(content="<think>\nhmm\n</think>\n\nok", model_extra={})
    assert ra.serialise(inline) == (inline.content, "inline")
    bare = types.SimpleNamespace(content="just an answer", model_extra={"reasoning": None})
    assert ra.serialise(bare) == ("just an answer", "none")
    none = types.SimpleNamespace(content=None, model_extra={})
    assert ra.serialise(none) == ("", "none")
    # truncated: reasoning but no answer / no clean stop → unclosed think block, executed == None downstream
    cut = types.SimpleNamespace(content="", model_extra={"reasoning": "partial"})
    text, _ = ra.serialise(cut, "length")
    assert text == "<think>\npartial" and split_think(text)[2] is False
    text, _ = ra.serialise(types.SimpleNamespace(content="ans", model_extra={"reasoning": "r"}), "length")
    assert not split_think(text)[2]
    both = types.SimpleNamespace(content="<think>\ndup\n</think>\n\nanswer", model_extra={"reasoning": "field"})
    text, _ = ra.serialise(both)
    assert text.count("<think>") == 1 and split_think(text)[1].strip() == "answer" and "field" in split_think(text)[0]
    alt = types.SimpleNamespace(content="a", model_extra={"reasoning_content": "rc"})
    assert ra.serialise(alt)[1] == "reasoning_field"
    names = [c.name for c in ra.conditions()]
    assert names == ["baseline", "prompt_test"] and ra.conditions()[1].system_prompt("AGENT").endswith(ra.TEST_SCENARIO_SENTENCE)
