"""Venhoff et al.'s steering-vector recipe, ported from ``train_vectors.py`` and ``utils.py``.

* Spans come from the annotated thinking via their regex; each span's text is located in the raw
  ``full_response`` with ``str.find`` and mapped to tokens through the tokenizer's offset mapping
  (their ``get_char_to_token_map``), tokenising the raw response without special tokens.
* The per-span vector is the mean of layer outputs over ``[start-1, min(end-1, start+10))`` — one
  token before the span through at most ten tokens into it — and the label's mean is a running mean
  over spans (``count`` per span).
* ``overall`` is, per trace, the mean over ``[min_start, max_end)`` of all spans, with a running mean
  over traces (``count`` per trace).
* Feature vectors are ``mean[b] − mean['overall']``, rescaled per layer to the norm of the overall
  mean (``normalize_features=True``).
* Layer index ``l`` in the saved tensors is decoder layer ``l``'s output, i.e. HF ``hidden_states[l+1]``.

A span-text filter lets the same recipe build a vector from a subset of spans (e.g. spans without
self-referential test language), which is how the geometry step separates test-talk from the rest.
"""

from __future__ import annotations

from typing import Callable

import torch

from .parse import LABELS, SPAN_RE

BEHAVIOURS = list(LABELS)


def label_positions(annotated: str, full_response: str, tok) -> dict[str, list[tuple[int, int]]]:
    """Their ``get_label_positions``: token spans per label, first occurrence, offsets-based."""
    enc = tok(full_response, return_offsets_mapping=True, add_special_tokens=False)
    char_to_token = {}
    for ti, (s, e) in enumerate(enc["offset_mapping"]):
        for c in range(s, e):
            char_to_token[c] = ti
    out: dict[str, list[tuple[int, int]]] = {}
    for m in SPAN_RE.finditer(annotated):
        label = m.group(1).strip()
        text = m.group(2).strip()
        if not text:
            continue
        pos = full_response.find(text)
        if pos < 0:
            continue
        ts = char_to_token.get(pos)
        te = char_to_token.get(pos + len(text) - 1)
        if ts is None or te is None:
            continue
        te += 1
        if ts >= te:
            continue
        out.setdefault(label, []).append((ts, te))
    return out


def span_windows(label_positions_: dict[str, list[tuple[int, int]]]) -> tuple[list[tuple[str, int, int]], tuple[int, int] | None]:
    """Their windows: per span ``[start-1, min(end-1, start+10))``; overall ``[min_start, max_end)``."""
    wins = []
    allpos = [p for ps in label_positions_.values() for p in ps]
    for label, ps in label_positions_.items():
        for s, e in ps:
            a, b = s - 1, min(e - 1, s + 10)
            if b > a >= 0:
                wins.append((label, a, b))
    overall = (min(s for s, _ in allpos), max(e for _, e in allpos)) if allpos else None
    return wins, overall


class MeanVectors:
    """Running means in their ``{label: {'mean': [n_layers, d], 'count': int}}`` layout."""

    def __init__(self):
        self.d: dict[str, dict] = {}

    def update(self, label: str, vec: torch.Tensor) -> None:
        if label not in self.d:
            self.d[label] = {"mean": torch.zeros_like(vec), "count": 0}
        cur = self.d[label]
        cur["mean"] = cur["mean"] + (vec - cur["mean"]) / (cur["count"] + 1)
        cur["count"] += 1

    def as_dict(self) -> dict:
        return {k: {"mean": v["mean"].clone(), "count": v["count"]} for k, v in self.d.items()}


def accumulate_trace(mv: MeanVectors, pooled_layers: Callable[[list[tuple[int, int]]], torch.Tensor], annotated: str, full_response: str, tok, span_filter: Callable[[str, str], bool] | None = None) -> int:
    """Add one trace to ``mv``. ``pooled_layers(windows)`` must return ``[n_windows, n_layers, d]``
    for decoder-layer outputs (HF hidden_states[1:]). ``span_filter(label, text)`` keeps a span when
    True. Returns the number of spans used."""
    lp = label_positions(annotated, full_response, tok)
    if span_filter is not None:
        enc_text = full_response
        kept: dict[str, list[tuple[int, int]]] = {}
        # re-derive span texts in reading order to apply the filter
        texts = [(m.group(1).strip(), m.group(2).strip()) for m in SPAN_RE.finditer(annotated)]
        it = {label: iter(ps) for label, ps in lp.items()}
        for label, text in texts:
            if label not in it or not text or enc_text.find(text) < 0:
                continue
            try:
                span = next(it[label])
            except StopIteration:
                continue
            if span_filter(label, text):
                kept.setdefault(label, []).append(span)
        lp = kept
    wins, overall = span_windows(lp)
    if not wins:
        return 0
    windows = [(a, b) for _, a, b in wins] + ([overall] if overall else [])
    pooled = pooled_layers(windows)  # [n, L, d]
    for (label, _, _), vec in zip(wins, pooled[: len(wins)]):
        if torch.isnan(vec).any():
            continue
        mv.update(label, vec)
    if overall is not None and not torch.isnan(pooled[-1]).any():
        mv.update("overall", pooled[-1])
    return len(wins)


def feature_vectors(mean_vectors: dict, normalize: bool = True) -> dict[str, torch.Tensor]:
    """``mean[b] − mean['overall']``, rescaled per layer to ``‖overall‖`` (their ``load_model_and_vectors``)."""
    overall = mean_vectors["overall"]["mean"]
    out = {}
    for b in BEHAVIOURS:
        if b not in mean_vectors:
            continue
        v = mean_vectors[b]["mean"] - overall
        if normalize:
            v = v * (overall.norm(dim=1, keepdim=True) / v.norm(dim=1, keepdim=True).clamp_min(1e-8))
        out[b] = v
    return out
