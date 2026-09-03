"""A small OpenRouter chat-completions client for judge calls.

Design goals, in order: every call is cached to disk as it completes (a crashed run loses nothing and
a rerun is free), calls run concurrently with bounded retries, and every record carries enough
metadata to audit it later (prompt hash, served model, finish reason, token usage, attempts, the
``max_tokens`` in force).

Records are appended to a JSONL file. If the same job id appears more than once (a failed call that
was retried on a later run), the **last** record wins. **One process per output file**: appends from
two processes can interleave mid-record. Within a process only the main thread writes.

A record counts as usable (:func:`record_ok`) only if it has no error, non-empty text, and
``finish_reason == "stop"``. Anything else — truncation, empty content, invalid JSON in JSON mode —
is recorded with an ``error`` and redone on the next run.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .env import require

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Pinned snapshot. Matches Abdelnabi & Salem's labeller in model (their ``judgeIt_batch.py`` default
#: is ``gpt-4o``, Azure-hosted, snapshot unrecorded) and Venhoff et al.'s stated annotator (GPT-4o).
JUDGE_MODEL = "openai/gpt-4o-2024-08-06"

#: gpt-4o's output ceiling. Venhoff's ``chat()`` asks for 28,000 and never truncates; an annotation is
#: ~2 completion tokens per reasoning word, so 16,384 covers traces of ~8,000 words. The longest shipped
#: trace is under 3,000 words.
DEFAULT_MAX_TOKENS = 16384

#: List prices in USD per token, used only to *estimate* spend in logs. OpenRouter bills at the
#: provider's price; check the dashboard for the real number.
PRICE_PER_INPUT_TOKEN = 2.50 / 1e6
PRICE_PER_OUTPUT_TOKEN = 10.00 / 1e6


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


@dataclass
class Job:
    """One judge call. ``meta`` is carried through untouched into the result record."""

    id: str
    prompt: str
    meta: dict = field(default_factory=dict)
    json_mode: bool = False


@dataclass
class Result:
    id: str
    prompt_sha: str
    text: str | None
    finish_reason: str | None
    usage: dict
    served_model: str | None
    error: str | None
    attempts: int
    judge_model: str
    temperature: float
    max_tokens: int
    ts: float
    meta: dict

    @property
    def ok(self) -> bool:
        return record_ok(asdict(self))


def record_ok(rec: dict) -> bool:
    """A cached record is usable iff it has no error, non-empty text and a clean stop."""
    return rec.get("error") is None and bool((rec.get("text") or "").strip()) and rec.get("finish_reason") == "stop"


class Judge:
    """Thin wrapper around the OpenAI SDK pointed at OpenRouter.

    ``temperature=0`` and a fixed ``seed`` make repeated calls as reproducible as the provider allows;
    they are not a determinism guarantee.
    """

    def __init__(
        self,
        model: str = JUDGE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        concurrency: int = 8,
        max_attempts: int = 6,
        timeout: float = 300.0,
        seed: int = 0,
        client: object | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.seed = seed
        # max_retries=0: we do our own retries so that attempts are counted and logged.
        # ``client`` is injectable for tests.
        self.client = client or OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=require("OPENROUTER_API_KEY"),
            timeout=timeout,
            max_retries=0,
        )

    def _result(self, job: Job, *, text, finish_reason, usage, served_model, error, attempts) -> Result:
        return Result(
            id=job.id,
            prompt_sha=prompt_sha(job.prompt),
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            served_model=served_model,
            error=error,
            attempts=attempts,
            judge_model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            ts=time.time(),
            meta=job.meta,
        )

    def complete(self, job: Job) -> Result:
        err: str | None = None
        last: dict = dict(text=None, finish_reason=None, usage={}, served_model=None)
        attempt = 0
        for attempt in range(1, self.max_attempts + 1):
            retry = False
            try:
                kwargs: dict = dict(
                    model=self.model,
                    messages=[{"role": "user", "content": job.prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )
                if job.json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                r = self.client.chat.completions.create(**kwargs)
                if not r.choices:
                    # OpenRouter can return 200 with an empty choices list and an error body.
                    raise RuntimeError(f"empty choices: {getattr(r, 'error', None)!r}")
                choice = r.choices[0]
                usage = {}
                if getattr(r, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": r.usage.prompt_tokens,
                        "completion_tokens": r.usage.completion_tokens,
                    }
                content = choice.message.content
                last = dict(text=content, finish_reason=choice.finish_reason, usage=usage, served_model=getattr(r, "model", None))
                if content is None or not content.strip():
                    err, retry = f"empty content (finish_reason={choice.finish_reason})", True
                elif choice.finish_reason != "stop":
                    # Truncated or filtered. Kept for inspection, flagged, redone on the next run.
                    err, retry = f"finish_reason={choice.finish_reason} (max_tokens={self.max_tokens})", False
                elif job.json_mode:
                    try:
                        json.loads(content)
                        err = None
                    except json.JSONDecodeError as e:
                        # Their getParsedContent retries invalid JSON up to 5 times; we do the same.
                        err, retry = f"invalid json: {e}", True
                else:
                    err = None
                if err is None:
                    return self._result(job, **last, error=None, attempts=attempt)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                err, retry = repr(e), True
            except APIStatusError as e:
                err = f"HTTP {e.status_code}: {e}"
                retry = e.status_code >= 500 or e.status_code in (408, 409, 425, 429)
            except Exception as e:  # noqa: BLE001 - recorded, not swallowed
                err, retry = repr(e), "empty choices" in repr(e)
            if not retry or attempt == self.max_attempts:
                break
            time.sleep(min(60.0, 2.0**attempt + random.random()))
        return self._result(job, **last, error=err, attempts=attempt)


def load_results(path: Path) -> dict[str, dict]:
    """Read a JSONL result file into ``{id: record}``; the last record per id wins.

    A malformed **final** line (left by a run killed mid-write) is skipped; a malformed line anywhere
    else is a corrupted file and raises. Records are split on ``\\n`` only, never on Unicode line
    separators that may occur inside judge text.
    """
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i >= len(lines) - 2:  # last line, or last line before a trailing newline
                continue
            raise
        out[rec["id"]] = rec
    return out


def estimate_cost(records: Iterable[dict]) -> tuple[int, int, float]:
    tok_in = tok_out = 0
    for r in records:
        u = r.get("usage") or {}
        tok_in += int(u.get("prompt_tokens", 0))
        tok_out += int(u.get("completion_tokens", 0))
    return tok_in, tok_out, tok_in * PRICE_PER_INPUT_TOKEN + tok_out * PRICE_PER_OUTPUT_TOKEN


def repair_tail(path: Path) -> bool:
    """Drop a partial final line left by a run killed mid-write. Every complete record ends with a
    newline, so a file that does not end with one has exactly one incomplete record at its tail.
    Returns True if something was removed."""
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    idx = data.rfind(b"\n")
    path.write_bytes(data[: idx + 1] if idx >= 0 else b"")
    return True


def _is_terminal(rec: dict) -> bool:
    """A truncated or filtered completion will recur at temperature 0; do not resend by default."""
    return str(rec.get("error") or "").startswith("finish_reason=")


def run_jobs(
    judge: Judge,
    jobs: list[Job],
    out_path: Path,
    log: Callable[[str], None] = print,
    progress_every: int = 50,
    retry_truncated: bool = False,
) -> dict:
    """Run every job whose result is not already cached as usable; append results as they land.

    Records with API errors, empty content or invalid JSON are redone. Records whose completion was
    truncated or filtered are left alone unless ``retry_truncated`` is set.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if repair_tail(out_path):
        log(f"{out_path}: removed a partial final record left by an interrupted run")
    existing = load_results(out_path)
    todo, terminal = [], 0
    for j in jobs:
        rec = existing.get(j.id)
        if rec and rec.get("prompt_sha") == prompt_sha(j.prompt):
            if record_ok(rec):
                continue
            if _is_terminal(rec) and not retry_truncated:
                terminal += 1
                continue
        todo.append(j)
    log(f"{out_path}: {len(jobs)} jobs, {len(jobs) - len(todo) - terminal} cached, {terminal} truncated (kept), {len(todo)} to run")
    lock = threading.Lock()
    n_done = n_err = 0
    new_records: list[dict] = []
    if todo:
        with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=judge.concurrency) as ex:
            futures = {ex.submit(judge.complete, j): j for j in todo}
            for fut in as_completed(futures):
                res = fut.result()
                rec = asdict(res)
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    f.flush()
                    new_records.append(rec)
                    n_done += 1
                    n_err += int(not res.ok)
                    if n_done % progress_every == 0 or n_done == len(todo):
                        ti, to, cost = estimate_cost(new_records)
                        log(
                            f"  {n_done}/{len(todo)} done, {n_err} errors, "
                            f"{ti + to:,} tokens, ~${cost:.2f} this run"
                        )
    tok_in, tok_out, cost = estimate_cost(new_records)
    return {
        "path": str(out_path),
        "jobs": len(jobs),
        "cached": len(jobs) - len(todo) - terminal,
        "truncated_kept": terminal,
        "run": n_done,
        "errors": n_err,
        "prompt_tokens": tok_in,
        "completion_tokens": tok_out,
        "est_cost_usd": round(cost, 4),
    }
