"""Abdelnabi & Salem's steering: a weight edit, ported from ``scripts/modify_model.py``.

``get_regions`` ranks every row of every layer's ``mlp.gate_proj.weight`` by cosine similarity to the
probe's positive-class weight row, takes the global top-800, and adds (``aware=True``) or subtracts
``alpha * vector`` to those rows in place. This module does the same, vectorised, and additionally

* records which rows were edited and the layer histogram of the 800 (Deliverable 2a in the plan),
* can undo the edit (so one loaded model serves every α and control), and
* provides the random-direction control: a Gaussian vector at the probe direction's norm, pushed
  through the identical selection-and-edit pipeline.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import torch
import torch.nn.functional as F

TOP_K = 800  # their setting


@dataclass
class Edit:
    layer_rows: list[tuple[int, int, float]]  # (layer, row, cosine) for every edited row
    alpha: float
    aware: bool
    originals: dict[tuple[int, int], torch.Tensor]

    @property
    def layer_histogram(self) -> dict[int, int]:
        return dict(sorted(collections.Counter(l for l, _, _ in self.layer_rows).items()))


def rank_rows(model, vector: torch.Tensor, k: int = TOP_K) -> list[tuple[int, int, float]]:
    """Global top-k ``gate_proj`` rows by cosine to ``vector``, exactly their ranking."""
    scores = []
    for layer_idx, layer in enumerate(model.model.layers):
        w = layer.mlp.gate_proj.weight  # [d_ff, d_model]
        cos = F.cosine_similarity(w.float(), vector.to(w.device).float().unsqueeze(0), dim=1)
        top = cos.topk(k=min(k, w.size(0)))
        scores.extend((layer_idx, int(i), float(v)) for v, i in zip(top.values.tolist(), top.indices.tolist()))
    scores.sort(key=lambda x: -x[2])
    return scores[:k]


@torch.no_grad()
def apply_edit(model, vector: torch.Tensor, alpha: float, aware: bool, k: int = TOP_K) -> Edit:
    """``w_row += alpha * vector`` (aware) or ``-=`` (unaware) on the top-k rows. Returns an undo handle."""
    rows = rank_rows(model, vector, k)
    originals = {}
    for layer_idx, row, _ in rows:
        w = model.model.layers[layer_idx].mlp.gate_proj.weight
        originals[(layer_idx, row)] = w[row].detach().clone()
        delta = (alpha * vector.to(w.device)).to(w.dtype)
        w[row] += delta if aware else -delta
    return Edit(layer_rows=rows, alpha=alpha, aware=aware, originals=originals)


@torch.no_grad()
def undo_edit(model, edit: Edit) -> None:
    for (layer_idx, row), orig in edit.originals.items():
        model.model.layers[layer_idx].mlp.gate_proj.weight[row] = orig


def random_direction_like(vector: torch.Tensor, seed: int) -> torch.Tensor:
    """Gaussian direction with the same norm as ``vector`` (the matched-norm random control)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(vector.shape, generator=g, dtype=torch.float32)
    return (r / r.norm() * vector.float().norm()).to(vector.dtype)


def load_probe_direction(path) -> torch.Tensor:
    """Their probe checkpoint: ``state_dict['fc.weight'][1]`` is the positive-class row."""
    sd = torch.load(path, map_location="cpu")
    return sd["fc.weight"][1].detach().clone()
