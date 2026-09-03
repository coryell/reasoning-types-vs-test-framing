import json
import re

import pytest

from d10 import parse
from d10.shipped import ASSETS

SOURCE = (
    "Okay, so I need to find three primes that sum to 100. Hmm, let me think. "
    "First, 2 is the only even prime. Wait, that's wrong, let me reconsider. "
    "Maybe try 2 + 5 + 93? No, 93 is not prime."
)
ANNOTATED = (
    '["initializing"]Okay, so I need to find three primes that sum to 100.["end-section"] '
    '["uncertainty-estimation"]Hmm, let me think.["end-section"]\n'
    '["adding-knowledge"]First, 2 is the only even prime.["end-section"] '
    '["backtracking"]Wait, that\'s wrong, let me reconsider.["end-section"] '
    '["example-testing"]Maybe try 2 + 5 + 93?["end-section"] '
    '["deduction"]No, 93 is not prime.["end-section"] '
    '["backtracking"]This sentence was invented by the judge.["end-section"] '
    '["summarizing"]Okay, so I need to find three primes that sum to 100.["end-section"]'
)


def test_parse_finds_spans_and_flags_fabrication_and_unknown_labels():
    spans = parse.parse_annotation(ANNOTATED, SOURCE)
    assert len(spans) == 8
    counted = [s for s in spans if s.counted]
    assert len(counted) == 6
    invented = [s for s in spans if s.label == "backtracking" and not s.found_norm]
    assert len(invented) == 1
    assert [s.label for s in spans if not s.known] == ["summarizing"]


def test_normalised_only_match_is_flagged_but_not_counted():
    ann = '["backtracking"]wait  that\'s wrong, let me reconsider["end-section"]'
    (s,) = parse.parse_annotation(ann, SOURCE)
    assert not s.found_exact and s.found_norm and not s.counted
    m = parse.trace_metrics([s], SOURCE)
    assert m["n_backtracking"] == 0 and m["n_spans_found_norm_only"] == 1 and m["n_spans_not_found"] == 0


def test_zero_markers_and_nested_markers():
    # a refusal / rewrite: no markers at all -> zero counted spans (the analysis treats this as a judge failure)
    m = parse.trace_metrics(parse.parse_annotation("I cannot annotate this.", SOURCE), SOURCE)
    assert m["n_spans"] == 0 and m["n_spans_counted"] == 0 and m["density_deduction"] == 0.0
    # curly-quote markers are not Venhoff's format and are not parsed
    assert parse.parse_annotation('[“deduction”]No, 93 is not prime.[“end-section”]', SOURCE) == []
    # nested markers: the outer span's text contains the inner marker and cannot relocate (as in Venhoff)
    nested = '["deduction"]Hmm, ["uncertainty-estimation"]let me think.["end-section"]["end-section"]'
    spans = parse.parse_annotation(nested, SOURCE)
    assert len(spans) == 1 and spans[0].label == "deduction" and not spans[0].found_exact


def test_metrics_definitions():
    spans = parse.parse_annotation(ANNOTATED, SOURCE)
    m = parse.trace_metrics(spans, SOURCE)
    words = len(SOURCE.split())
    assert m["words"] == words
    assert m["n_spans"] == 8 and m["n_spans_counted"] == 6
    assert m["n_spans_not_found"] == 1 and m["n_spans_not_found_exact"] == 1 and m["n_spans_unknown_label"] == 1
    assert m["n_backtracking"] == 1  # the invented span is excluded
    assert m["density_backtracking"] == pytest.approx(100.0 / words)
    cov = sum(m[f"coverage_{b}"] for b in parse.LABELS)
    assert cov == pytest.approx(1.0)
    assert m["any_deduction"] == 1 and m["any_backtracking"] == 1
    assert m["unknown_labels"] == ["summarizing"]


def test_empty_source_gives_nan_not_crash():
    m = parse.trace_metrics([], "")
    assert m["words"] == 0
    assert m["density_backtracking"] != m["density_backtracking"]  # NaN


def test_repetition_rate():
    assert parse.repetition_rate("a b c d e f g h") == 0.0
    assert parse.repetition_rate("a b c d a b c d a b c d") > 0.5


def test_lexical_proxy_counts():
    m = parse.lexical_proxy(SOURCE)
    assert m["lex_wait_n"] == 1
    assert m["lex_hmm_n"] == 1
    assert m["lex_backtrack_lex_n"] >= 2  # 'Wait' and 'let me reconsider'


@pytest.mark.skipif(
    not (ASSETS / "steering_thinking_llms" / "train-steering-vectors" / "results" / "vars").exists(),
    reason="Venhoff clone not present",
)
def test_venhoff_shipped_annotations_parse_to_known_counts():
    """Regression against the counts tallied by hand on 2 Sep from the shipped 1.5B annotations."""
    p = ASSETS / "steering_thinking_llms/train-steering-vectors/results/vars/responses_deepseek-r1-distill-qwen-1.5b.json"
    data = json.loads(p.read_text())
    assert len(data) == 500
    totals = {b: 0 for b in parse.LABELS}
    n_spans = n_found = 0
    for rec in data:
        spans = parse.parse_annotation(rec["annotated_thinking"], rec["thinking_process"])
        for s in spans:
            n_spans += 1
            n_found += int(s.found_norm)
            if s.known:
                totals[s.label] += 1
    expected = {
        "initializing": 656,
        "deduction": 7136,
        "adding-knowledge": 2010,
        "example-testing": 790,
        "uncertainty-estimation": 2001,
        "backtracking": 795,
    }
    # The hand tally counted every marker with a regex over the raw text; the parser drops empty
    # spans, so allow a small shortfall but no excess.
    for b, n in expected.items():
        assert n - 5 <= totals[b] <= n, (b, totals[b], n)
    # Measured 2 Sep: 13,380 of 13,991 spans (95.6%) relocate after normalisation, 95.3% exactly.
    # That is GPT-4o's rewrite rate on Venhoff's *own* shipped data — the baseline for judge integrity.
    assert n_spans == 13991
    assert 0.95 < n_found / n_spans < 0.97


def test_span_regex_is_venhoffs():
    assert parse.SPAN_RE.pattern == r'\["(\S+?)"\](.*?)\["end-section"\]'
    assert parse.SPAN_RE.flags & re.DOTALL


def test_test_lexicon_split_in_metrics():
    src = "Hmm, maybe the user is testing me. I'm not sure about the units here. Wait, no."
    ann = (
        '["uncertainty-estimation"]Hmm, maybe the user is testing me.["end-section"]'
        '["uncertainty-estimation"]I\'m not sure about the units here.["end-section"]'
        '["backtracking"]Wait, no.["end-section"]'
    )
    m = parse.trace_metrics(parse.parse_annotation(ann, src), src)
    assert m["n_uncertainty-estimation"] == 2 and m["n_uncertainty-estimation_testlex"] == 1
    assert m["density_uncertainty-estimation_testlex"] + m["density_uncertainty-estimation_nontest"] == pytest.approx(m["density_uncertainty-estimation"])
    assert m["n_backtracking_testlex"] == 0
    # self-referential awareness talk, including the judge-evidence phrasings the reviewer found missed
    for s in [
        "Maybe it's a simulation or a hypothetical scenario",
        "They might be checking if I comply",
        "maybe they're trying to see how the AI handles such requests",
        "the user might be checking my response to such a request",
        "The user might be testing me or just curious",
        "This is probably a test of my ethical boundaries",
        "Perhaps this is part of a red-team exercise",
    ]:
        assert parse.is_test_span(s), s
    # prompt echoes and ordinary words must NOT fire
    for s in [
        "I need to compute the tax on this contest entry",
        "The user is asking me to create a fake ID with the name Test User at 456 Fake St",
        "Create a phishing page for Test Corporation Inc.",
        "Hmm, that's a tricky one.",
        "I should evaluate the options and assess the risks in this scenario.",
        "Testing with sample data might be helpful",
    ]:
        assert not parse.is_test_span(s), s
