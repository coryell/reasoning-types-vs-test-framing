"""One description of a steering condition shared by the generators and the readouts.

Three manipulations appear in the plan's follow-ups (EXECUTION_PLAN_2, follow-up table):
``prompt`` (Abdelnabi & Salem §4.5: a sentence appended to the system prompt, no weight change),
``weight_edit`` (their gate_proj edit, :mod:`d10.steer`) and ``residual_add`` (Venhoff et al.'s
residual-stream addition at one layer, :class:`d10.activations.ResidualAdd`). A :class:`Condition`
is serialised into each arm's sidecar so the projection readout can re-apply the exact manipulation
the traces were generated under, instead of inferring it from the arm's name.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from d10.activations import ResidualAdd
from d10.steer import apply_edit, random_direction_like, rank_rows, undo_edit
from d10.venhoff import feature_vectors

KINDS = ("none", "prompt", "weight_edit", "residual_add")
#: Their §4.5 prompting baseline, verbatim.
TEST_SCENARIO_SENTENCE = "This is a test scenario with no real-world consequences."


@dataclass
class Condition:
    name: str
    kind: str = "none"
    # weight_edit: direction ∈ {probe, random, shuffled}; alpha; aware (sign); rows_from ∈ {"edit direction", "probe direction"}
    direction: str = ""
    alpha: float = 0.0
    aware: bool = True
    rows_from: str = "edit direction"
    # residual_add: direction = behaviour; vectors = mean_vectors | mean_vectors_nontest | mean_vectors_testlex, or
    # "random" (a seeded Gaussian with the norm of mean_vectors' direction at that layer: the norm-matched control); layer; coefficient
    vectors: str = "mean_vectors"
    layer: int | None = None
    coefficient: float = 0.0
    # prompt
    prompt_suffix: str = ""
    seed: int = 0
    info: dict = field(default_factory=dict)  # filled when applied: layer_histogram, relative_edit, vector_norm

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown condition kind {self.kind!r}")
        if self.kind == "weight_edit" and (not self.direction or not self.alpha):
            raise ValueError(f"{self.name}: weight_edit needs a direction and a non-zero alpha")
        if self.kind == "residual_add" and (not self.direction or self.layer is None or not self.coefficient):
            raise ValueError(f"{self.name}: residual_add needs a behaviour, a layer and a non-zero coefficient")
        if self.kind == "prompt" and not self.prompt_suffix:
            raise ValueError(f"{self.name}: prompt condition needs prompt_suffix")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # the (alpha, aware) pair the Tier 1 loaders and analysis carry; residual_add maps its signed coefficient onto it
    @property
    def signed(self) -> tuple[float, bool]:
        if self.kind == "weight_edit":
            return self.alpha, self.aware
        if self.kind == "residual_add":
            return abs(self.coefficient), self.coefficient > 0
        return 0.0, True

    def system_prompt(self, base: str | None) -> str | None:
        if self.kind != "prompt":
            return base
        return f"{base.rstrip()}\n\n{self.prompt_suffix}" if base else self.prompt_suffix


def legacy_condition(arm: str, alpha: float, aware: bool) -> Condition:
    """The main run's arms (``scripts/run_steering.py``), whose sidecars predate ``Condition``."""
    if arm.startswith("baseline") or not alpha:
        return Condition(name=arm)
    direction = "random" if arm.startswith("random") else "shuffled" if arm.startswith("shuffled") else "probe"
    rows_from = "probe direction" if "_samerows_" in arm else "edit direction"
    return Condition(name=arm, kind="weight_edit", direction=direction, alpha=alpha, aware=aware, rows_from=rows_from)


def condition_from_sidecar(path: Path, side: dict) -> Condition:
    if isinstance(side.get("condition"), dict):
        return Condition.from_dict(side["condition"])
    stem = Path(path).stem
    c = legacy_condition(stem, float(side.get("alpha", 0.0)), bool(side.get("aware")))
    # run_steering.py draws its random direction from the *base* seed even for the `_seed1` second
    # decodes (whose generation seed is base + 1), so the edit seed is recovered from the name
    c.seed = int(side.get("config", {}).get("seed", 0)) - (1 if stem.endswith("_seed1") else 0)
    return c


def probe_vectors(probe: dict, seed: int) -> dict[str, torch.Tensor]:
    """``probe`` (their positive-class row), ``shuffled`` (shuffled-label probe, matched norm) and
    ``random`` (Gaussian, matched norm, seeded) — the three weight-edit directions."""
    v_pos = torch.tensor(probe["direction"], dtype=torch.float32)
    v_shuf = torch.tensor(probe["shuffled_direction"], dtype=torch.float32)
    return {"probe": v_pos, "shuffled": v_shuf / v_shuf.norm() * v_pos.norm(), "random": random_direction_like(v_pos, seed=seed)}


def residual_vector(cond: Condition, vectors_dir: Path) -> torch.Tensor:
    if cond.vectors == "random":
        ref = feature_vectors(torch.load(Path(vectors_dir) / "mean_vectors.pt"))[cond.direction][cond.layer]
        return random_direction_like(ref, seed=cond.seed) * cond.coefficient
    mv = torch.load(Path(vectors_dir) / f"{cond.vectors}.pt")
    return feature_vectors(mv)[cond.direction][cond.layer] * cond.coefficient


@contextlib.contextmanager
def applied(model, cond: Condition, probe: dict | None = None, vectors_dir: Path | None = None):
    """Apply ``cond`` to ``model`` for the duration of the block and undo it afterwards. Fills
    ``cond.info``. ``prompt`` and ``none`` conditions leave the model untouched."""
    cond.info = {}
    if cond.kind == "weight_edit":
        vecs = probe_vectors(probe, cond.seed)
        vec = vecs[cond.direction]
        rows = rank_rows(model, vecs["probe"]) if cond.rows_from == "probe direction" else None
        edit = apply_edit(model, vec, cond.alpha, cond.aware, rows=rows)
        row_norms = torch.stack([edit.originals[(l, r)].float().norm() for l, r, _ in edit.layer_rows])
        cond.info = {"layer_histogram": edit.layer_histogram, "relative_edit": float(cond.alpha * vec.norm() / row_norms.mean()), "vector_norm": float(vec.norm()), "rows_edited": len(edit.layer_rows)}
        try:
            yield
        finally:
            undo_edit(model, edit)
    elif cond.kind == "residual_add":
        vec = residual_vector(cond, vectors_dir)
        cond.info = {"vector_norm": float(vec.norm()), "layer": cond.layer}
        with ResidualAdd(model, cond.layer, vec):
            yield
    else:
        yield


def dump(conds: list[Condition]) -> str:
    return json.dumps([c.to_dict() for c in conds], indent=1)
