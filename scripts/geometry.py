#!/usr/bin/env python
"""Geometry (EXECUTION_PLAN_2 §M2): cosine of the awareness probe direction against the rebuilt
Venhoff directions, per layer, against chance and a random direction; plus the test-language
decomposition of the directions themselves — the uncertainty (and every other) direction rebuilt
from spans WITH and WITHOUT self-referential test language.

Layer alignment: the probe's best layer is an HF hidden_states index k (k = output of decoder layer
k−1); Venhoff's vectors are indexed by decoder layer l (= hidden_states[l+1]). The comparison uses the
probe direction at every k against the behaviour vector at l = k−1, i.e. the same residual stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from d10.venhoff import BEHAVIOURS, feature_vectors  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vectors-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "vectors")
    ap.add_argument("--probe-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "probe")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "qwen3_4b" / "geometry.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    probes = torch.load(args.probe_dir / "probe_directions.pt")
    best = json.loads((args.probe_dir / "probe_best.json").read_text())["layer"]
    fv = {}
    for name in ("mean_vectors", "mean_vectors_nontest", "mean_vectors_testlex"):
        f = args.vectors_dir / f"{name}.pt"
        if f.exists():
            mv = torch.load(f)
            if "overall" in mv:
                fv[name] = feature_vectors(mv)
    n_dec = fv["mean_vectors"]["deduction"].shape[0]
    d = fv["mean_vectors"]["deduction"].shape[1]
    g = torch.Generator().manual_seed(args.seed)
    rand = torch.randn(1000, d, generator=g)
    chance_abs_cos = float(torch.nn.functional.cosine_similarity(rand[:500], rand[500:], dim=1).abs().mean())

    per_layer = []
    for k in range(1, n_dec + 1):  # hidden_states index k ↔ decoder layer k-1
        if k not in probes:
            continue
        row = {"hs_index": k, "decoder_layer": k - 1}
        for name, vecs in fv.items():
            for b in BEHAVIOURS:
                if b in vecs:
                    row[f"{name}/{b}"] = cos(probes[k], vecs[b][k - 1])
        row["random"] = float(torch.nn.functional.cosine_similarity(probes[k].float().unsqueeze(0), rand[:200], dim=1).abs().mean())
        per_layer.append(row)
    at_best = next(r for r in per_layer if r["hs_index"] == best)
    # inter-behaviour cosines at the probe's layer (Venhoff Appendix C analogue)
    inter = {}
    for i, a in enumerate(BEHAVIOURS):
        for b in BEHAVIOURS[i + 1 :]:
            if a in fv["mean_vectors"] and b in fv["mean_vectors"]:
                inter[f"{a}~{b}"] = cos(fv["mean_vectors"][a][best - 1], fv["mean_vectors"][b][best - 1])
    shuffled = probes.get("shuffled")
    shuf_row = {b: cos(shuffled, fv["mean_vectors"][b][best - 1]) for b in BEHAVIOURS if shuffled is not None and b in fv["mean_vectors"]}
    out = {"probe_best_hs_index": best, "chance_abs_cos": chance_abs_cos, "at_best_layer": at_best, "shuffled_probe_at_best": shuf_row, "inter_behaviour_at_best": inter, "per_layer": per_layer}
    args.out.write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = [r["decoder_layer"] for r in per_layer]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for c, b in zip(colors, BEHAVIOURS):
        key = f"mean_vectors/{b}"
        if key in per_layer[0]:
            ax.plot(xs, [r[key] for r in per_layer], color=c, linewidth=2, label=b)
    ax.plot(xs, [r["random"] for r in per_layer], color="#c3c2b7", linestyle=":", linewidth=1.5, label="random |cos|")
    ax.axvline(best - 1, color="#c3c2b7", linewidth=0.8, linestyle="--")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("cos(awareness probe, behaviour direction)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=150)
    print(json.dumps({"best": best, "at_best": {k: round(v, 3) for k, v in at_best.items() if isinstance(v, float)}, "chance": round(chance_abs_cos, 4)}, indent=1))


if __name__ == "__main__":
    main()
