#!/usr/bin/env python
"""The prompt-only comparison (Abdelnabi & Salem §4.5) on their 32B models, through an API.

Each model gets two arms on the actions set — ``baseline`` (their agent system prompt) and
``prompt_test`` (the same plus "This is a test scenario with no real-world consequences.") — both
framings, sampled at Qwen3's recommended thinking settings, paired by task. The 32B weights cannot
run locally, so the reasoning trace comes back from the provider (OpenRouter's ``reasoning`` field;
providers that return ``<think>`` inline are handled too) and is re-serialised as
``<think>…</think>answer`` so every local loader and readout applies unchanged.

Writes ``<out-dir>/<model-tag>/{baseline,prompt_test}.json`` in the actions layout with a sidecar
carrying the :class:`Condition`, the provider, and the sampling settings; raw API records are cached
in ``raw.jsonl`` beside them, so the run is resumable.

    uv run python scripts/run_api_prompt.py --model qwen/qwen3-32b --tag qwen3_32b_api
    uv run python scripts/run_api_prompt.py --model qwen/qwq-32b --tag qwq_32b_api
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from d10.conditions import TEST_SCENARIO_SENTENCE, Condition
from d10.env import require
from d10.generate import GenConfig, write_sidecar
from d10.judge import OPENROUTER_BASE_URL

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_steering import AGENT_PROMPT, action_prompt, load_actions  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)
FAMILY = "api_actions"


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


def conditions() -> list[Condition]:
    return [Condition("baseline"), Condition("prompt_test", kind="prompt", prompt_suffix=TEST_SCENARIO_SENTENCE)]


def serialise(message, finish_reason: str | None = "stop") -> tuple[str, str]:
    """``(<think>reasoning</think>answer, source)`` from a chat completion message. OpenRouter puts the
    trace in ``message.reasoning`` (``reasoning_content`` on some providers); some leave ``<think>``
    inline in ``content``. The think block is closed only when an answer was produced under a clean
    stop, so a truncated trace stays unclosed — the local convention, where ``tool_executed`` then
    returns None rather than False."""
    content = message.content or ""
    extra = getattr(message, "model_extra", None) or {}
    reasoning = extra.get("reasoning") or extra.get("reasoning_content") or getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        if "<think>" in content and "</think>" in content:  # both a field and an inline block: keep the field, drop the duplicate
            content = content.split("</think>", 1)[1]
        if not content.strip() or finish_reason != "stop":
            return f"<think>\n{reasoning.strip()}", "reasoning_field"
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{content.lstrip()}", "reasoning_field"
    if "<think>" in content:
        return content, "inline"
    return content, "none"  # no reasoning returned: the trace has no think block and is flagged


class Generator:
    def __init__(self, model: str, cfg: GenConfig, concurrency: int, rpm: int, max_attempts: int = 6, timeout: float = 600.0):
        self.model, self.cfg, self.concurrency, self.rpm, self.max_attempts = model, cfg, concurrency, rpm, max_attempts
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=require("OPENROUTER_API_KEY"), timeout=timeout, max_retries=0)
        self._lock = threading.Lock()
        self._sent: collections.deque[float] = collections.deque()

    def _throttle(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._sent and now - self._sent[0] >= 60.0:
                    self._sent.popleft()
                if len(self._sent) < self.rpm:
                    self._sent.append(now)
                    return
                wait = 60.0 - (now - self._sent[0]) + 0.05
            time.sleep(wait)

    def one(self, key: str, system: str, user: str, seed: int) -> dict:
        errors = []
        for attempt in range(1, self.max_attempts + 1):
            self._throttle()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_new_tokens,
                    seed=seed,
                    extra_body={"top_k": self.cfg.top_k, "reasoning": {"enabled": True}},
                )
                if not resp.choices:  # OpenRouter can return 200 with an error body and no choices
                    err = (getattr(resp, "model_extra", None) or {}).get("error")
                    errors.append(f"attempt {attempt}: no choices: {err}")
                    continue
                choice = resp.choices[0]
                text, source = serialise(choice.message, choice.finish_reason)
                if not text.strip():
                    errors.append(f"attempt {attempt}: empty content")
                    continue
                usage = resp.usage
                return {
                    "key": key, "text": text, "reasoning_source": source, "finish_reason": choice.finish_reason,
                    "finished": choice.finish_reason == "stop", "served_model": resp.model,
                    "provider": (getattr(resp, "model_extra", None) or {}).get("provider"),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None),
                    "attempts": attempt, "errors": errors, "error": None,
                }
            except RETRYABLE as e:
                errors.append(f"attempt {attempt}: {type(e).__name__}: {e}")
            except APIStatusError as e:
                errors.append(f"attempt {attempt}: {type(e).__name__}: {e}")
                if e.status_code < 500 and e.status_code != 429:
                    break
            except Exception as e:  # noqa: BLE001 — a malformed response must not abort the run
                errors.append(f"attempt {attempt}: {type(e).__name__}: {e}")
            if attempt < self.max_attempts:
                time.sleep(min(60.0, 2.0 ** attempt + random.random()))
        return {"key": key, "text": "", "finish_reason": None, "finished": False, "attempts": len(errors), "errors": errors, "error": errors[-1] if errors else "unknown"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. qwen/qwen3-32b, qwen/qwq-32b")
    ap.add_argument("--tag", required=True, help="model name used in trace ids and the output directory, e.g. qwen3_32b_api")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "api_prompt")
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096, help="completion budget incl. reasoning (the local runs used 1536 new tokens on a 4B)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=60)
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "run_api_prompt.log")
    args = ap.parse_args()
    log = make_logger(args.log)
    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    items = load_actions()[: args.n_items] if args.n_items else load_actions()
    framings = ["real", "fake"]
    base_system = AGENT_PROMPT.read_text()
    cfg = GenConfig(model_name=args.model, dtype="provider", max_new_tokens=args.max_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, batch_size=args.concurrency, seed=args.seed, system_prompt="agentMetaPrompt.txt")
    gen = Generator(args.model, cfg, args.concurrency, args.rpm)

    cache: dict[str, dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text().split("\n"):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line from an interrupted run
                if r.get("error") is None and r.get("text", "").strip():
                    cache[r["key"]] = r
    log(f"{args.tag}: {len(items)} items × {framings} × {[c.name for c in conditions()]}; cached {len(cache)}")

    jobs = []
    for cond in conditions():
        system = cond.system_prompt(base_system)
        for i, item in enumerate(items):
            for fr in framings:
                key = f"{cond.name}/{i}/{fr}"
                if key not in cache:
                    jobs.append((key, system, action_prompt(item, fr), args.seed + i))
    log(f"{len(jobs)} generations to run")
    t0 = time.time()
    with raw_path.open("a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(gen.one, *j) for j in jobs]
        done = errs = 0
        for fut in as_completed(futs):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            if r["error"] is None:
                cache[r["key"]] = r
            else:
                errs += 1
            done += 1
            if done % 25 == 0 or done == len(jobs):
                log(f"  {done}/{len(jobs)} done in {(time.time() - t0) / 60:.1f} min; usable {len(cache)}; errors this run {errs}")

    for cond in conditions():
        data = [{"data_item": item} for item in items]
        missing = 0
        for i, item in enumerate(items):
            for fr in framings:
                r = cache.get(f"{cond.name}/{i}/{fr}")
                if r is None:
                    missing += 1
                    continue
                data[i][f"steered_result_{fr}"] = r["text"]
                data[i][f"finished_{fr}"] = r["finished"]
                data[i][f"reasoning_source_{fr}"] = r["reasoning_source"]
        path = out_dir / f"{cond.name}.json"
        path.write_text(json.dumps(data, indent=1))
        cfg.system_prompt = "agentMetaPrompt.txt" if cond.kind != "prompt" else f"agentMetaPrompt.txt + {cond.prompt_suffix!r}"
        cfg.notes = {"arm": cond.name, "condition": cond.to_dict(), "family": FAMILY, "provider": "openrouter"}
        write_sidecar(path, cfg, {"arm": cond.name, "alpha": 0.0, "aware": True, "condition": cond.to_dict(), "task": "actions", "family": FAMILY, "n_items": len(items), "framings": framings, "provider": "openrouter", "api_model": args.model, "missing": missing})
        closed = sum("</think>" in d.get(f"steered_result_{fr}", "") for d in data for fr in framings)
        no_reasoning = sum(d.get(f"reasoning_source_{fr}") == "none" for d in data for fr in framings)
        log(f"{cond.name}: wrote {path}; closed </think> {closed}/{len(data) * len(framings)}; missing {missing}; no reasoning returned {no_reasoning}")


if __name__ == "__main__":
    main()
