#!/usr/bin/env python
"""F5 (EXECUTION_PLAN_2 follow-ups): what is in the shipped 32B awareness probes' steering direction?

Abdelnabi & Salem steer with ``fc.weight[1]`` — the positive-class row of a 2-output linear head
trained with cross-entropy from PyTorch's default initialisation (uniform ±1/√d; root-mean-square
row norm √(1/3) ≈ 0.577, independent of d) by plain SGD (their ``mlp_train.py``: seed 0, lr 0.008,
300 epochs, no momentum, no weight decay). For a two-class softmax the gradients on the two rows are
exactly opposite, so ``w1 + w0`` never changes during training: ``w1 = c + Δ`` and ``w0 = c − Δ``
with ``c`` the fixed initialisation mean (the *same* vector for every probe, since the seed and
width are shared) and ``2Δ = w1 − w0`` the learned discriminant. Per model and layer this script
reports the row norms, ``‖w1 + w0‖`` (must equal the seed-0 value everywhere: a check of the
assumption), the cosines between ``w1``, ``w0`` and ``w1 − w0``, and the fraction of ``‖w1‖²``
along the discriminant. The steering row is therefore the learned discriminant plus one fixed
random vector of norm ≈ 0.41 (``‖c‖``); the ratio ``‖c‖ / ‖w1‖`` says how much that matters per
layer. CPU only. Writes ``results/probe_geometry_shipped/``.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from d10.shipped import TA_DIR

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = TA_DIR / "scripts" / "output_models"
INIT_ROW_RMS = math.sqrt(1.0 / 3.0)  # √E‖row‖² for uniform(−1/√d, 1/√d), independent of d


def probe_stats(sd: dict) -> dict:
    w = sd["fc.weight"].float()
    w0, w1 = w[0], w[1]
    d = w1 - w0
    cos = lambda a, b: float(F.cosine_similarity(a, b, dim=0))
    return {
        "d_model": int(w.shape[1]), "norm_w0": float(w0.norm()), "norm_w1": float(w1.norm()), "norm_diff": float(d.norm()),
        "norm_w1_over_init": float(w1.norm()) / INIT_ROW_RMS,
        "norm_sum": float((w0 + w1).norm()),  # = ‖2c‖, invariant under their training; identical across probes if the init assumption holds
        "init_over_w1": float((w0 + w1).norm() / 2 / w1.norm()),  # ‖c‖ / ‖w1‖: relative size of the fixed init vector inside the steering row
        "cos_w1_diff": cos(w1, d), "cos_w0_w1": cos(w0, w1),
        # how much of w1 lies along the discriminant vs orthogonal to it
        "frac_w1_along_diff": float((w1 @ d / d.norm()) ** 2 / (w1.norm() ** 2)) if float(d.norm()) > 0 else math.nan,
        "bias_diff": float(sd["fc.bias"].float()[1] - sd["fc.bias"].float()[0]) if "fc.bias" in sd else math.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "probe_geometry_shipped")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mdir in sorted(args.models_dir.iterdir()):
        if not mdir.is_dir():
            continue
        model = mdir.name.split("_from_evidence")[0]
        files = [(int(m.group(1)), f) for f in mdir.glob("model_*.pth") if (m := re.fullmatch(r"model_(\d+)\.pth", f.name))]
        for layer, f in sorted(files):
            rows.append({"model": model, "layer": layer, **probe_stats(torch.load(f, map_location="cpu"))})
    if not rows:
        raise SystemExit(f"no model_<layer>.pth files under {args.models_dir}; is the Test_Awareness_Steering clone present?")
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_layer.csv", index=False)
    summ = df.groupby("model").agg(layers=("layer", "size"), d_model=("d_model", "first"), norm_w1_mean=("norm_w1", "mean"), norm_w1_min=("norm_w1", "min"), norm_w1_over_init_mean=("norm_w1_over_init", "mean"),
                                  norm_sum_min=("norm_sum", "min"), norm_sum_max=("norm_sum", "max"), init_over_w1_max=("init_over_w1", "max"),
                                  cos_w1_diff_mean=("cos_w1_diff", "mean"), cos_w1_diff_min=("cos_w1_diff", "min"), cos_w0_w1_mean=("cos_w0_w1", "mean"),
                                  frac_w1_along_diff_mean=("frac_w1_along_diff", "mean")).reset_index()
    summ.to_csv(args.out_dir / "summary.csv", index=False)
    lines = ["# Shipped awareness probes: how much of the steering row is the learned discriminant\n",
             f"PyTorch's default Linear init gives rows of root-mean-square norm {INIT_ROW_RMS:.3f} regardless of width; their SGD (no weight decay) moves `w1` and `w0` by exactly opposite steps, so `w1 + w0` is the initialisation mean `2c` and is the same vector in every probe (`norm_sum` must be constant; the seed-0 value at d = 5120 is 0.8166). "
             "`cos_w1_diff` is the cosine between the steering row `w1` and the discriminant `w1 − w0`; `frac_w1_along_diff` the fraction of `‖w1‖²` along it; `norm_w1_over_init` the row norm relative to its initialisation scale; `init_over_w1` = `‖c‖ / ‖w1‖`, the relative size of the fixed random vector inside the steering row (largest at layer 0, where the row barely trained).\n",
             summ.round(4).to_markdown(index=False), "\n## Per layer (every 8th)\n",
             df[df.layer % 8 == 0].round(4).to_markdown(index=False)]
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
