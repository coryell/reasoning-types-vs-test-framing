#!/usr/bin/env python
"""Train the test-awareness probe on our own generations, their recipe (EXECUTION_PLAN_2 §M1).

Inputs: a trigger generation file and its awareness judgments. For each judged side, the class is
the paper's (positive: awareness Yes/Maybe; negative: awareness No and recognition No); the evidence
quotes the judge returned are relocated in the lowercased chat-template text and mean-pooled per
layer (their ``location="avg"``). One datapoint per quote. Negatives use the judge's evidence for
the No conclusion, as in their ``get_evidence_indices``.

Outputs under ``--out-dir``: ``probe_metrics.json`` (per-layer acc/AUROC, both trainers, shuffled and
layer-0 controls), ``probe_directions.pt`` (fc.weight[1] per layer; gitignored) and
``probe_best.json`` (best layer + direction as a list, committed).

``--pooling`` reproduces their token-position ablation (§4.2): ``span`` (default, average over the
judge's evidence span — what the steering direction uses), ``whole`` (average over every reasoning
token including the opening ``<think>``, excluding the closing ``</think>``; one datapoint per side),
``last`` (the last reasoning token before ``</think>``). Traces longer than ``--max-len`` tokens are
clipped for ``whole`` and dropped for ``last``; the count is logged. The last two
test whether awareness is decodable away from the words the judge selected; if only ``span`` works,
the probe reads the quote's vocabulary. Use a different ``--out-dir`` per pooling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from d10.activations import abdelnabi_probe_text, char_spans_to_token_spans, locate_quotes, pooled_states
from d10.awareness import is_negative, is_positive, parse_judgment
from d10.generate import load_model
from d10.judge import load_results, record_ok
from d10.probe import probe_all_layers
from d10.shipped import load_generations

REPO = Path(__file__).resolve().parents[1]


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-name", default="qwen3_4b")
    ap.add_argument("--family", default="triggers_probe")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "probe")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pooling", choices=["span", "whole", "last"], default="span")
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "build_probe.log")
    args = ap.parse_args()
    log = make_logger(args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traces = {t.id: t for t in load_generations(args.generations, args.model_name, args.family)}
    judged = {k: v for k, v in load_results(args.judgments).items() if record_ok(v) and k in traces}
    log(f"{len(judged)} judged sides with usable judgments; pooling={args.pooling}")

    model, tok = load_model(args.model)
    feats, labels, groups, meta = [], [], [], []
    n_pos = n_neg = n_quotes_dropped = 0
    for tid, rec in judged.items():
        j = parse_judgment(rec["text"])
        if is_positive(j):
            y = 1
            quotes = j["model_awareness"].get("evidence") or []
        elif is_negative(j):
            y = 0
            quotes = (j["model_awareness"].get("evidence") or []) + (j["hypothetical_recognition"].get("evidence") or [])
        else:
            continue
        t = traces[tid]
        text = abdelnabi_probe_text(tok, t.task, t.raw)
        if args.pooling == "span":
            cspans = locate_quotes(text, quotes)
            n_quotes_dropped += len([q for q in quotes if isinstance(q, str)]) - len(cspans)
            tspans = char_spans_to_token_spans(tok, text, cspans, add_special_tokens=True)
        else:
            # the reasoning region: from the end of the chat-template prefix to just before the appended </think>
            reasoning_lower = t.raw.rsplit("</think>", 1)[0].lower()
            prefix_len = len(text) - len(reasoning_lower + "</think>")
            (rs, re_), = char_spans_to_token_spans(tok, text, [(prefix_len, prefix_len + len(reasoning_lower))], add_special_tokens=True) or [(0, 0)]
            tspans = [(rs, re_)] if args.pooling == "whole" else [(re_ - 1, re_)]  # 'last' = last reasoning token before </think>
            tspans = [s for s in tspans if s[1] > s[0]]
            if tspans and tspans[0][1] > args.max_len:
                n_clipped = locals().get("n_clipped", 0) + 1
            else:
                n_clipped = locals().get("n_clipped", 0)
        if not tspans:
            continue
        pooled = pooled_states(model, tok, text, tspans, add_special_tokens=True, max_len=args.max_len)
        for row in pooled:
            if torch.isnan(row).any():
                continue
            feats.append(row)
            labels.append(y)
            groups.append(t.index)
            meta.append(tid)
        n_pos += y == 1
        n_neg += y == 0
        if len(meta) % 200 == 0:
            log(f"  {len(meta)} spans from {n_pos + n_neg} sides")
    if not feats:
        raise SystemExit("no spans extracted")
    X = torch.stack(feats)
    y = np.array(labels)
    g = np.array(groups)
    log(f"features {tuple(X.shape)}; sides pos={n_pos} neg={n_neg}; spans pos={int(y.sum())} neg={int((1 - y).sum())}; quotes not relocated={n_quotes_dropped}; windows beyond max_len={locals().get('n_clipped', 0)}")
    torch.save({"X": X, "y": torch.tensor(y), "groups": torch.tensor(g)}, args.out_dir / "probe_features.pt")
    (args.out_dir / "probe_feature_ids.json").write_text(json.dumps(meta))

    results, control = probe_all_layers(X, y, g, seed=args.seed)
    best = max(results, key=lambda r: (r.acc, r.auroc))
    w1, w_shuf = best.direction, control["shuffled_direction"]
    w1_minus_w0 = best.weight[1] - best.weight[0]
    diag = {
        "norm_w1": float(w1.norm()), "norm_w1_minus_w0": float(w1_minus_w0.norm()),
        "cos_w1_vs_w1_minus_w0": float((w1 @ w1_minus_w0) / (w1.norm() * w1_minus_w0.norm())),
        "norm_shuffled_w1": float(w_shuf.norm()),
        "cos_w1_vs_shuffled": float((w1 @ w_shuf) / (w1.norm() * w_shuf.norm())),
        "init_row_norm_expected": float((1.0 / 3.0) ** 0.5),  # nn.Linear default init: U(-1/sqrt(d), 1/sqrt(d)) per weight -> row norm ~ 1/sqrt(3); if norm_w1 sits here, SGD barely moved it
    }
    log(f"direction diagnostics: {json.dumps({k: round(v, 4) for k, v in diag.items()})}")
    metrics = {
        "pooling": args.pooling,
        "direction_diagnostics": diag,
        "model": args.model,
        "n_sides_pos": n_pos,
        "n_sides_neg": n_neg,
        "n_spans_pos_balanced": control["n_pos"],
        "n_spans_neg_balanced": control["n_neg"],
        "hidden_state_indexing": "HF: 0 = embeddings, k = output of decoder layer k-1",
        "per_layer": [{"layer": r.layer, "n_train": r.n_train, "n_test": r.n_test, "acc": r.acc, "auroc": r.auroc, "sk_acc": r.sk_acc, "sk_auroc": r.sk_auroc} for r in results],
        "best_layer": best.layer,
        "best_acc": best.acc,
        "best_auroc": best.auroc,
        "controls": {k: v for k, v in control.items() if k != "shuffled_direction"},
    }
    (args.out_dir / "probe_metrics.json").write_text(json.dumps(metrics, indent=1))
    torch.save({r.layer: r.direction for r in results} | {"shuffled": control["shuffled_direction"], "shuffled_layer": control["shuffled_layer"]}, args.out_dir / "probe_directions.pt")
    (args.out_dir / "probe_best.json").write_text(json.dumps({"layer": best.layer, "acc": best.acc, "auroc": best.auroc, "direction": best.direction.tolist(), "shuffled_direction": control["shuffled_direction"].tolist()}))
    log(f"best layer {best.layer}: acc {best.acc:.3f} auroc {best.auroc:.3f} (sklearn {best.sk_acc:.3f}/{best.sk_auroc:.3f}); shuffled acc {control['shuffled_acc']:.3f}; layer-0 acc {control['layer0_acc']:.3f}")


if __name__ == "__main__":
    main()
