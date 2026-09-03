"""Hidden-state extraction, span pooling, projections and a residual-add hook.

Two upstream conventions live here and must not be mixed up:

* **Abdelnabi & Salem** (``getRepFromEvidence.py``): text = chat template of the *lowercased* prompt
  + the *lowercased* reasoning up to and including ``</think>``; evidence quotes are relocated as
  lowercased substrings; ``outputs.hidden_states[layer]`` (HF indexing: 0 = embeddings, k = output
  of decoder layer k−1) is mean-pooled over each quote's tokens.
* **Venhoff et al.** (``train_vectors.py``): the raw response string alone, no template, tokenised
  without special tokens; ``model.model.layers[i].output[0]`` = HF ``hidden_states[i+1]``.

Everything returns float32 CPU tensors.
"""

from __future__ import annotations

from typing import Iterable

import torch

THINK_CLOSE = "</think>"


def abdelnabi_probe_text(tok, prompt: str, raw_result: str) -> str:
    """Their ``prepare_texts_from_examples``: lowercase both, keep reasoning through ``</think>``."""
    reasoning = raw_result.rsplit(THINK_CLOSE, 1)[0] + THINK_CLOSE
    messages = [{"role": "user", "content": prompt.lower()}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + reasoning.lower()


def locate_quotes(text: str, quotes: Iterable[str]) -> list[tuple[int, int]]:
    """Character spans of each quote found as a lowercased substring (their relocation rule);
    quotes not found are dropped, as in their ``get_evidence_indices``."""
    out = []
    for q in quotes:
        if not isinstance(q, str):
            continue
        q = q.lower().strip()
        if not q:
            continue
        pos = text.find(q)
        if pos >= 0:
            out.append((pos, pos + len(q)))
    return out


def char_spans_to_token_spans(tok, text: str, char_spans: list[tuple[int, int]], add_special_tokens: bool) -> list[tuple[int, int]]:
    """Map character spans to half-open token spans via the tokenizer's offset mapping."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    offsets = enc["offset_mapping"]
    spans = []
    for cs, ce in char_spans:
        idx = [i for i, (s, e) in enumerate(offsets) if e > cs and s < ce and e > s]
        if idx:
            spans.append((idx[0], idx[-1] + 1))
    return spans


@torch.inference_mode()
def pooled_states(model, tok, text: str, token_windows: list[tuple[int, int]], add_special_tokens: bool, max_len: int = 3072) -> torch.Tensor:
    """Mean of every layer's hidden state over each token window.

    Returns ``[n_windows, n_hidden_states, d]`` with HF indexing (index 0 = embeddings). Windows
    that fall entirely beyond ``max_len`` are returned as NaN rows and must be dropped by the caller.
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=add_special_tokens, truncation=True, max_length=max_len).to(model.device)
    out = model(**enc, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0].float()  # [L+1, T, d]
    T = hs.shape[1]
    rows = []
    for s, e in token_windows:
        e = min(e, T)
        if e <= s:
            rows.append(torch.full((hs.shape[0], hs.shape[2]), float("nan")))
        else:
            rows.append(hs[:, s:e].mean(dim=1).cpu())
    return torch.stack(rows) if rows else torch.empty(0, hs.shape[0], hs.shape[2])


@torch.inference_mode()
def mean_projection(model, tok, text: str, hs_index: int, direction: torch.Tensor, token_window: tuple[int, int] | None = None, max_len: int = 3072) -> float:
    """Mean over tokens of ``h · v̂`` at ``hidden_states[hs_index]`` on the raw text (no template)."""
    enc = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len).to(model.device)
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[hs_index][0].float()
    if token_window is not None:
        s, e = token_window
        h = h[s:e]
    v = direction.to(h.device).float()
    v = v / v.norm()
    return float((h @ v).mean()) if h.shape[0] else float("nan")


class ResidualAdd:
    """Venhoff's steering: add ``vec`` to decoder layer ``layer_idx``'s output at every position on
    every forward pass (their ``model.model.layers[l].output[0][:, :] += coefficient * vector``).
    Use as a context manager."""

    def __init__(self, model, layer_idx: int, vec: torch.Tensor):
        self.layer = model.model.layers[layer_idx]
        self.vec = vec
        self.handle = None

    def _hook(self, module, args, output):
        v = self.vec.to(output[0].device if isinstance(output, tuple) else output.device)
        if isinstance(output, tuple):
            return (output[0] + v.to(output[0].dtype),) + tuple(output[1:])
        return output + v.to(output.dtype)

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
