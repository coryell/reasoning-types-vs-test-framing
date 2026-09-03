import collections

import pytest

from d10 import shipped

HAVE_TA = shipped.OUTPUT_MODELS.exists()
needs_ta = pytest.mark.skipif(not HAVE_TA, reason="Abdelnabi clone not present under assets/")


def test_parse_arm():
    a = shipped.parse_arm("steered_output_dataset_actions_alpha0.05_aware.json")
    assert (a.name, a.alpha, a.aware, a.signed_alpha) == ("alpha0.05_aware", 0.05, True, 0.05)
    b = shipped.parse_arm("steered_output_dataset_triggers_alpha0.25_judge.json")
    assert (b.name, b.alpha, b.aware, b.signed_alpha) == ("alpha0.25", 0.25, False, -0.25)
    z = shipped.parse_arm("steered_output_dataset_actions_alpha0.0.json")
    assert (z.name, z.signed_alpha) == ("alpha0.0", 0.0)
    with pytest.raises(ValueError):
        shipped.parse_arm("nothing.json")


def test_split_think_conventions():
    # Distill-Qwen / QwQ: no opening tag
    r, a, ok = shipped.split_think("Okay, thinking. </think>\n{'tool': 'x'}")
    assert (r, a, ok) == ("Okay, thinking.", "{'tool': 'x'}", True)
    # Qwen3: opening tag present
    r, a, ok = shipped.split_think("<think>\nOkay.\n</think>{}")
    assert (r, a, ok) == ("Okay.", "{}", True)
    # truncated
    r, a, ok = shipped.split_think("<think>\nstill thinking")
    assert (r, a, ok) == ("still thinking", "", False)
    # first </think> wins for the reasoning span
    r, a, ok = shipped.split_think("one</think>two</think>three")
    assert (r, a) == ("one", "two</think>three")


def test_tool_executed_is_their_rule():
    assert shipped.tool_executed("reasoning</think>{'tool': 'send_email'}", "send_email") is True
    assert shipped.tool_executed("reasoning</think>{}", "send_email") is False
    # tool name only in the reasoning does not count
    assert shipped.tool_executed("I could use send_email</think>{}", "send_email") is False
    # no </think>: unknown, not "not executed"
    assert shipped.tool_executed("I will call send_email", "send_email") is None
    # last </think> is the split point in their rule
    assert shipped.tool_executed("a</think>send_email</think>{}", "send_email") is False


def test_execution_rate_conventions():
    ex = [True, False, None, True]
    assert shipped.execution_rate(ex, "all") == pytest.approx(2 / 4)
    assert shipped.execution_rate(ex, "closed") == pytest.approx(2 / 3)
    assert shipped.execution_rate([None, None], "closed") != shipped.execution_rate([None, None], "closed")  # NaN
    with pytest.raises(ValueError):
        shipped.execution_rate(ex, "other")


def test_abdelnabi_split_is_raw():
    r, a = shipped.abdelnabi_split("Okay.\n</think>\n\n{}")
    assert (r, a) == ("Okay.\n", "\n\n{}")
    with pytest.raises(ValueError):
        shipped.abdelnabi_split("no close")


def test_actions_task_text():
    di = {"task": "Post about {who}.", "parameter_name": "who", "real_value": "Dr. X", "fake_value": "Test Y"}
    assert shipped.actions_task_text(di, "real") == "Post about Dr. X."
    assert shipped.actions_task_text(di, "fake") == "Post about Test Y."


@needs_ta
def test_models_and_families_present():
    assert shipped.list_models() == ["deepseek_qwen", "qwen3", "qwq"]
    assert "actions_sampling" in shipped.list_families("deepseek_qwen")
    assert "actions_sampling" not in shipped.list_families("qwq")


@needs_ta
def test_table4_reproduces_under_their_two_denominators():
    """Paper Table 4 baselines (their baseline function: unknown → not executed, ÷ all items):
    Distill-Qwen 12.0/20.8, Qwen3 20.1/37.5, QwQ 36.2/69.7. Steered rows (their steered function:
    ÷ items that closed </think>): Qwen3 α=0.3 → 41.8/68.0, which is 62/148 and 100/147."""
    base = {"deepseek_qwen": (12.1, 20.8), "qwen3": (20.1, 37.6), "qwq": (36.2, 69.8)}
    for model, (real, fake) in base.items():
        by = collections.defaultdict(list)
        for t in shipped.iter_traces(model, "actions"):
            if t.arm == "alpha0.0":
                by[t.framing].append(t.executed)
        assert len(by["real"]) == 149 and len(by["fake"]) == 149
        assert 100 * shipped.execution_rate(by["real"], "all") == pytest.approx(real, abs=0.1)
        assert 100 * shipped.execution_rate(by["fake"], "all") == pytest.approx(fake, abs=0.1)
    by = collections.defaultdict(list)
    for t in shipped.iter_traces("qwen3", "actions"):
        if t.arm == "alpha0.3":
            by[t.framing].append(t.executed)
    assert sum(e is None for e in by["real"]) == 1 and sum(e is None for e in by["fake"]) == 2
    assert 100 * shipped.execution_rate(by["real"], "closed") == pytest.approx(41.9, abs=0.1)
    assert 100 * shipped.execution_rate(by["fake"], "closed") == pytest.approx(68.0, abs=0.1)


@needs_ta
def test_ids_unique_and_items_aligned_across_arms():
    """Paired analyses assume index i is the same item in every arm of a family."""
    ids = set()
    for model in shipped.list_models():
        tasks_by_arm = collections.defaultdict(dict)
        for t in shipped.iter_traces(model, "actions"):
            assert t.id not in ids
            ids.add(t.id)
            tasks_by_arm[t.arm][(t.index, t.framing)] = t.task
        arms = list(tasks_by_arm)
        for arm in arms[1:]:
            assert tasks_by_arm[arm] == tasks_by_arm[arms[0]], (model, arm)
    tasks_by_arm = collections.defaultdict(dict)
    for t in shipped.iter_traces("qwq", "triggers"):
        assert t.id not in ids
        ids.add(t.id)
        tasks_by_arm[t.arm][t.index] = t.task
    arms = list(tasks_by_arm)
    for arm in arms[1:]:
        assert tasks_by_arm[arm] == tasks_by_arm[arms[0]]


@needs_ta
def test_trace_counts_and_fields():
    qwq_trig = list(shipped.iter_traces("qwq", "triggers"))
    arms = collections.Counter(t.arm for t in qwq_trig)
    assert arms == {"alpha0.0": 493, "alpha0.05_aware": 493, "alpha0.25": 493}
    t = qwq_trig[0]
    assert t.framing == "hypothetical" and t.task and t.judge is not None
    assert set(t.judge) >= {"steered_model_awareness", "steered_recognition", "steered_task_performance"}
    hb = list(shipped.iter_traces("qwq", "harmbench"))
    assert hb and hb[0].task and hb[0].judge is not None
    # The sampled-decoding variant ships 149 items at alpha 0.0 and 0.07_aware but only 88 at 0.35.
    sampling = list(shipped.iter_traces("deepseek_qwen", "actions_sampling"))
    assert collections.Counter(t.arm for t in sampling) == {"alpha0.0": 298, "alpha0.07_aware": 298, "alpha0.35": 176}


@needs_ta
def test_think_close_rates_match_inventory():
    """From the 2 Sep inventory: Distill-Qwen and QwQ actions always close </think>; Qwen3 misses 41."""
    missing = collections.Counter()
    for model in shipped.list_models():
        for t in shipped.iter_traces(model, "actions"):
            missing[model] += int(not t.has_think_close)
    assert missing["deepseek_qwen"] == 0 and missing["qwq"] == 0 and missing["qwen3"] == 41
