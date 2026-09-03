"""Cache and retry semantics of d10.judge with a fake client. No network."""

import json
from types import SimpleNamespace

import pytest

from d10 import judge as J


class FakeClient:
    """Returns scripted responses in order; records the kwargs of every call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        content, finish = item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            model="fake/model",
        )


def make_judge(script, **kw):
    kw.setdefault("max_attempts", 3)
    j = J.Judge(client=FakeClient(script), **kw)
    j._sleep_backup = J.time.sleep
    return j


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(J.time, "sleep", lambda s: None)


def test_ok_record_is_cached_and_skipped(tmp_path):
    out = tmp_path / "r.jsonl"
    job = J.Job(id="a", prompt="p")
    s1 = J.run_jobs(make_judge([("annotated", "stop")]), [job], out, log=lambda m: None)
    assert (s1["run"], s1["errors"]) == (1, 0)
    rec = J.load_results(out)["a"]
    assert J.record_ok(rec) and rec["max_tokens"] == J.DEFAULT_MAX_TOKENS and rec["served_model"] == "fake/model"
    s2 = J.run_jobs(make_judge([]), [job], out, log=lambda m: None)
    assert (s2["cached"], s2["run"]) == (1, 0)


def test_changed_prompt_invalidates_cache(tmp_path):
    out = tmp_path / "r.jsonl"
    J.run_jobs(make_judge([("x", "stop")]), [J.Job(id="a", prompt="p1")], out, log=lambda m: None)
    s = J.run_jobs(make_judge([("y", "stop")]), [J.Job(id="a", prompt="p2")], out, log=lambda m: None)
    assert s["run"] == 1
    assert J.load_results(out)["a"]["text"] == "y"


def test_truncated_output_is_an_error_and_is_redone(tmp_path):
    out = tmp_path / "r.jsonl"
    job = J.Job(id="a", prompt="p")
    s1 = J.run_jobs(make_judge([("partial", "length")]), [job], out, log=lambda m: None)
    rec = J.load_results(out)["a"]
    assert s1["errors"] == 1 and not J.record_ok(rec)
    assert rec["text"] == "partial" and "finish_reason=length" in rec["error"] and rec["attempts"] == 1
    s2 = J.run_jobs(make_judge([("full", "stop")]), [job], out, log=lambda m: None, retry_truncated=True)
    assert s2["run"] == 1 and J.record_ok(J.load_results(out)["a"])


def test_empty_content_is_retried_then_recorded_with_error():
    j = make_judge([(None, "stop"), ("", "stop"), ("ok", "stop")], max_attempts=3)
    r = j.complete(J.Job(id="a", prompt="p"))
    assert r.ok and r.attempts == 3
    j = make_judge([(None, "stop"), (None, "stop")], max_attempts=2)
    r = j.complete(J.Job(id="a", prompt="p"))
    assert not r.ok and r.error.startswith("empty content") and r.attempts == 2


def test_json_mode_retries_invalid_json():
    j = make_judge([("not json", "stop"), ('{"a": 1}', "stop")])
    r = j.complete(J.Job(id="a", prompt="p", json_mode=True))
    assert r.ok and r.attempts == 2 and json.loads(r.text) == {"a": 1}
    assert j.client.calls[0]["response_format"] == {"type": "json_object"}
    j = make_judge([("not json", "stop")] * 2, max_attempts=2)
    r = j.complete(J.Job(id="a", prompt="p", json_mode=True))
    assert not r.ok and r.error.startswith("invalid json")


def test_rate_limit_retries_and_status_errors_classify():
    class RL(J.RateLimitError):
        def __init__(self):
            pass

        def __repr__(self):
            return "RateLimitError()"

    j = make_judge([RL(), ("ok", "stop")])
    r = j.complete(J.Job(id="a", prompt="p"))
    assert r.ok and r.attempts == 2

    class Boom(Exception):
        pass

    j = make_judge([Boom("bad request")], max_attempts=3)
    r = j.complete(J.Job(id="a", prompt="p"))
    assert not r.ok and r.attempts == 1 and "Boom" in r.error


def test_request_parameters_match_venhoff_send_path():
    j = make_judge([("ok", "stop")], temperature=0.0)
    j.complete(J.Job(id="a", prompt="PROMPT"))
    call = j.client.calls[0]
    assert call["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert call["temperature"] == 0.0 and call["model"] == J.JUDGE_MODEL
    assert call["max_tokens"] == J.DEFAULT_MAX_TOKENS and "response_format" not in call


def test_load_results_tolerates_partial_last_line_only(tmp_path):
    p = tmp_path / "r.jsonl"
    good = json.dumps({"id": "a", "text": "t u", "error": None, "finish_reason": "stop"})
    p.write_text(good + "\n" + '{"id": "b", "tex')
    recs = J.load_results(p)
    assert set(recs) == {"a"} and recs["a"]["text"] == "t u"
    p.write_text('{"id": "b", "tex\n' + good + "\n")
    with pytest.raises(json.JSONDecodeError):
        J.load_results(p)


def test_run_jobs_escapes_unicode_line_separators(tmp_path):
    out = tmp_path / "r.jsonl"
    J.run_jobs(make_judge([("line sep", "stop")]), [J.Job(id="a", prompt="p")], out, log=lambda m: None)
    assert " " not in out.read_text(encoding="utf-8")
    assert J.load_results(out)["a"]["text"] == "line sep"


def test_interrupted_write_is_repaired_before_append(tmp_path):
    out = tmp_path / "r.jsonl"
    good = json.dumps({"id": "a", "text": "t", "error": None, "finish_reason": "stop", "prompt_sha": J.prompt_sha("pa"), "meta": {}})
    out.write_text(good + "\n" + '{"id": "b", "tex')  # killed mid-write
    jobs = [J.Job(id="a", prompt="pa"), J.Job(id="b", prompt="pb")]
    s = J.run_jobs(make_judge([("bee", "stop")]), jobs, out, log=lambda m: None)
    assert (s["cached"], s["run"]) == (1, 1)
    recs = J.load_results(out)  # must not raise: the partial line was truncated, not glued to a record
    assert recs["b"]["text"] == "bee" and out.read_text().endswith("\n")
    # a second rerun is a no-op
    s = J.run_jobs(make_judge([]), jobs, out, log=lambda m: None)
    assert (s["cached"], s["run"]) == (2, 0)


def test_truncated_records_are_kept_unless_retry_requested(tmp_path):
    out = tmp_path / "r.jsonl"
    job = J.Job(id="a", prompt="p")
    J.run_jobs(make_judge([("partial", "length")]), [job], out, log=lambda m: None)
    s = J.run_jobs(make_judge([]), [job], out, log=lambda m: None)
    assert (s["truncated_kept"], s["run"]) == (1, 0)
    s = J.run_jobs(make_judge([("full", "stop")]), [job], out, log=lambda m: None, retry_truncated=True)
    assert s["run"] == 1 and J.record_ok(J.load_results(out)["a"])


def test_rpm_limiter_spaces_requests(monkeypatch):
    # 3 rpm: the 4th request in a burst must wait until the first is 60 s old
    clock = {"t": 1000.0}
    monkeypatch.setattr(J.time, "monotonic", lambda: clock["t"])
    slept = []

    def fake_sleep(s):
        slept.append(s)
        clock["t"] += s

    monkeypatch.setattr(J.time, "sleep", fake_sleep)
    j = make_judge([("ok", "stop")] * 4, rpm=3)
    for _ in range(4):
        j.complete(J.Job(id="a", prompt="p"))
    assert len(j.client.calls) == 4
    assert sum(slept) >= 60.0 and clock["t"] >= 1060.0
