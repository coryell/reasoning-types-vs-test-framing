"""Local generation with a Hugging Face causal LM, mirroring the authors' scripts.

Abdelnabi & Salem generate their probe-training traces with vLLM at default sampling and their steered
outputs with HF ``generate`` (``do_sample=False``, ``max_new_tokens=4000``, ``pad_to_multiple_of=8``,
``skip_special_tokens=True`` on decode); Qwen3 models get ``enable_thinking=True`` in the chat template
and no system prompt for triggers (``genOutputs.py``), the agentic system prompt for actions
(``genOutputs_steered_batch.py``). Venhoff et al. generate with greedy decoding and 1,000 max tokens.

Every run writes a sidecar ``<out>.gen.json`` with the exact configuration, git commit, and library
versions, so a generation file can be traced to the code that produced it.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]


@dataclass
class GenConfig:
    model_name: str = "Qwen/Qwen3-4B"
    dtype: str = "bfloat16"
    max_new_tokens: int = 2048
    do_sample: bool = True
    #: Qwen3's recommended thinking-mode sampling; greedy decoding is documented to loop.
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    batch_size: int = 8
    seed: int = 0
    enable_thinking: bool = True
    system_prompt: str | None = None
    notes: dict = field(default_factory=dict)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_model(model_name: str, dtype: str = "bfloat16"):
    """bf16 on the GPU, loaded shard by shard so the 10 GB WSL RAM cap is not hit."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=getattr(torch, dtype), device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    return model, tok


def format_prompt(tok, user: str, system: str | None = None, enable_thinking: bool = True) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:  # templates without the flag (R1-Distill etc.)
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@dataclass
class Generation:
    text: str
    finished: bool  # an EOS was produced before max_new_tokens
    n_new_tokens: int


@torch.inference_mode()
def generate(model, tok, prompts: list[str], cfg: GenConfig, log: Callable[[str], None] = print) -> list[Generation]:
    """Batched generation. Prompts are processed longest-first to limit padding; results come back in
    the original order. Their loop: ``pad_to_multiple_of=8``, ``pad_token_id=eos``,
    ``skip_special_tokens=True`` on decode (``<think>`` tags are ordinary tokens and survive)."""
    torch.manual_seed(cfg.seed)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out: list[Generation | None] = [None] * len(prompts)
    eos_ids = set(tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id])
    if getattr(model.generation_config, "eos_token_id", None) is not None:
        e = model.generation_config.eos_token_id
        eos_ids |= set(e if isinstance(e, list) else [e])
    t0 = time.time()
    done = 0
    for s in range(0, len(order), cfg.batch_size):
        idx = order[s : s + cfg.batch_size]
        batch = [prompts[i] for i in idx]
        enc = tok(batch, return_tensors="pt", padding=True, pad_to_multiple_of=8).to(model.device)
        kwargs = dict(max_new_tokens=cfg.max_new_tokens, pad_token_id=tok.pad_token_id, do_sample=cfg.do_sample)
        if cfg.do_sample:
            kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
        gen = model.generate(**enc, **kwargs)
        n_in = enc.input_ids.shape[1]
        for j, i in enumerate(idx):
            new = gen[j, n_in:]
            new_list = new.tolist()
            # strip trailing padding after EOS
            finished = any(t in eos_ids for t in new_list)
            if finished:
                first_eos = next(k for k, t in enumerate(new_list) if t in eos_ids)
                new_list = new_list[: first_eos + 1]
            out[i] = Generation(text=tok.decode(new_list, skip_special_tokens=True), finished=finished, n_new_tokens=len(new_list))
        done += len(idx)
        el = time.time() - t0
        log(f"  gen {done}/{len(prompts)}  {el / 60:.1f} min  eta {el / done * (len(prompts) - done) / 60:.1f} min  unfinished so far {sum(1 for g in out if g and not g.finished)}")
    return out  # type: ignore[return-value]


def write_sidecar(out_path: Path, cfg: GenConfig, extra: dict | None = None) -> Path:
    import transformers

    side = out_path.with_suffix(out_path.suffix + ".gen.json")
    side.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "git_commit": git_commit(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                **(extra or {}),
            },
            indent=1,
        )
    )
    return side
