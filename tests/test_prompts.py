import pytest

from d10 import prompts


def test_venhoff_template_matches_upstream_source():
    if not prompts.VENHOFF_UTILS.exists():
        pytest.skip("Venhoff clone not present under assets/")
    upstream = prompts.extract_venhoff_template_from_source()
    assert prompts.VENHOFF_ANNOTATION_TEMPLATE == upstream


def test_venhoff_prompt_substitutes_without_format_hazards():
    thinking = 'Okay {weird} braces ["label"] and \\backslashes'
    p = prompts.venhoff_annotation_prompt(thinking)
    assert thinking in p
    assert "{thinking}" not in p
    assert p.count("Available labels:") == 1


def test_abdelnabi_template_fills_like_their_code():
    if not prompts.ABDELNABI_JUDGE_PROMPT.exists():
        pytest.skip("Abdelnabi clone not present under assets/")
    t = prompts.load_abdelnabi_template()
    p = prompts.abdelnabi_judge_prompt(t, task="TASK-X", reasoning="REASON-Y", answer="ANSWER-Z")
    assert '"task": "TASK-X"' in p
    assert '"reasoning": "REASON-Y"' in p
    assert '"answer": "ANSWER-Z"' in p
    # the {{ }} escapes must have collapsed to single braces for the JSON skeleton
    assert '"model_awareness": {' in p
    assert "{{" not in p
