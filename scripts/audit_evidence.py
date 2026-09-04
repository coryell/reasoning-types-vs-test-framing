#!/usr/bin/env python
"""Deliverable 0b (EXECUTION_PLAN_2 §4): audit the evidence quotes behind Abdelnabi & Salem's published
probe labels.

Their judge is told to copy quotes "word-by-word, letter-by-letter, and character-by-character"; their
training code (``get_evidence_indices``) relocates each awareness/recognition quote as a lowercased
substring of chat-template(prompt) + reasoning and silently drops the rest. ``task_performance``
quotes are never read by the probe pipeline and are, by the prompt's instruction, drawn from the
answer — they are reported separately and excluded from every headline number.

Tiers per quote: exact substring of the reasoning; lowercased match (what their filter accepts);
verbatim in the answer only; found only after whitespace/punctuation normalisation; found only in the
user prompt (their search space includes the prompt, so such a span passes their filter but pools the
wrong tokens); not found.

Two class rules are reported, because they disagree: the paper's §3.3 (positive = awareness ∈
{Yes, Maybe}), which reproduces the published per-class counts exactly (742 / 563 / 322), and the
code's ``get_classes`` default (awareness OR recognition ∈ {Yes, Maybe}), which does not. Per quote,
``used_by_probe`` follows their code: for a positive side, quotes of the keys whose conclusion is
Yes/Maybe; for a negative side (both No), quotes of both keys. ``used_by_probe_paper`` takes only the
awareness key's quotes on a positive side (their code run with ``positive_keys=["model_awareness"]``,
which is what reproduces the published counts); ``used_by_probe_code`` follows their default, which
also takes recognition quotes when recognition is Yes/Maybe. Duplicate quotes within a side are
counted once, as their ``set()`` does. Writes ``results/evidence_audit/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from d10.parse import normalise
from d10.shipped import TA_DIR

REPO = Path(__file__).resolve().parents[1]
FILES = {
    "deepseek_qwen": "triggers_deepseek_qwen_with_GPT_labels_evidence.json",
    "qwen3": "triggers_qwen3_with_GPT_labels_evidence.json",
    "qwq": "triggers_qwq_with_GPT_labels_evidence_refined.json",
}
DIMS = ("model_awareness", "recognition", "task_performance")
YM = ("Yes", "Maybe")
TIERS = ["exact", "lowercased", "answer_only", "normalised_only", "prompt_only", "not_found"]


def tier(quote: str, reasoning: str, answer: str, prompt: str) -> str:
    q = quote.lower().strip()
    if quote in reasoning:
        return "exact"
    if q in reasoning.lower():
        return "lowercased"  # their get_evidence_indices accepts this
    if quote in answer or q in answer.lower():
        return "answer_only"
    if normalise(quote) and normalise(quote) in normalise(reasoning):
        return "normalised_only"
    if q in prompt.lower():
        return "prompt_only"  # passes their filter (prompt is in their search text) but pools prompt tokens
    return "not_found"


def audit(model: str, path: Path) -> tuple[list[dict], list[dict]]:
    rows, sides = [], []
    for i, e in enumerate(json.loads(path.read_text())):
        for side in ("real", "hypothetical"):
            raw = e.get(f"{side}_result")
            aw = e.get(f"{side}_model_awareness")
            rc = e.get(f"{side}_recognition")
            # their parse_json drops the whole entry on any bad field; we drop the bad side only (negligible: <20 sides per file)
            if not isinstance(raw, str) or "</think>" not in raw or not isinstance(aw, dict) or not isinstance(rc, dict):
                continue
            reasoning, answer = raw.split("</think>", 1)
            prompt = e.get(side, "") or ""
            a, r = aw.get("conclusion"), rc.get("conclusion")
            pos_paper = a in YM
            pos_code = a in YM or r in YM
            neg = a == "No" and r == "No"
            sides.append({"model": model, "side": side, "awareness": a, "recognition": r, "pos_paper": pos_paper, "pos_code": pos_code, "neg": neg})
            for dim in DIMS:
                d = e.get(f"{side}_{dim}")
                if not isinstance(d, dict):
                    continue
                seen = set()
                for q in d.get("evidence") or []:
                    if not isinstance(q, str) or not q.strip() or q in seen:
                        continue
                    seen.add(q)
                    key_positive = d.get("conclusion") in YM
                    used_code = dim != "task_performance" and ((pos_code and key_positive) or (neg and not pos_code))
                    used_paper = dim != "task_performance" and ((pos_paper and dim == "model_awareness" and key_positive) or (neg and not pos_paper))
                    rows.append(
                        {
                            "model": model, "item": i, "side": side, "dimension": dim, "conclusion": d.get("conclusion"),
                            "probe_class_paper": "positive" if pos_paper else ("negative" if neg else "unused"),
                            "probe_class_code": "positive" if pos_code else ("negative" if neg else "unused"),
                            "used_by_probe_paper": used_paper, "used_by_probe_code": used_code,
                            "tier": tier(q, reasoning, answer, prompt), "quote": q,
                        }
                    )
    return rows, sides


def table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    t = df.groupby(by).tier.value_counts(normalize=True).unstack().reindex(columns=TIERS).fillna(0.0)
    t["n_quotes"] = df.groupby(by).size()
    t["dropped_by_their_filter"] = t["answer_only"] + t["normalised_only"] + t["not_found"]
    t["wrong_tokens_prompt"] = t["prompt_only"]
    return t.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "evidence_audit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, sides = [], []
    for model, name in FILES.items():
        r, s = audit(model, TA_DIR / "data" / "triggers_annotated_answers" / name)
        rows += r
        sides += s
    df = pd.DataFrame(rows)
    sd = pd.DataFrame(sides)
    df.drop(columns=["quote"]).to_csv(args.out_dir / "quotes.csv", index=False)
    probe = df[df.dimension != "task_performance"]
    by_model = table(probe, ["model"])
    by_dim_class = table(probe, ["model", "dimension", "conclusion"])
    by_paper = table(probe[probe.used_by_probe_paper], ["model", "probe_class_paper"])
    by_code = table(probe[probe.used_by_probe_code], ["model", "probe_class_code"])
    task = table(df[df.dimension == "task_performance"], ["model"])
    for name, t in (("by_model", by_model), ("by_dimension_conclusion", by_dim_class), ("by_probe_class_paper", by_paper), ("by_probe_class_code", by_code), ("task_performance_quotes", task)):
        t.to_csv(args.out_dir / f"{name}.csv", index=False)
    classes = sd.groupby("model").agg(sides=("side", "size"), aware_yes=("awareness", lambda s: int((s == "Yes").sum())), aware_maybe=("awareness", lambda s: int((s == "Maybe").sum())), recog_yes=("recognition", lambda s: int((s == "Yes").sum())), recog_maybe=("recognition", lambda s: int((s == "Maybe").sum())), pos_paper=("pos_paper", "sum"), pos_code=("pos_code", "sum"), neg=("neg", "sum")).reset_index()
    classes.to_csv(args.out_dir / "class_sizes.csv", index=False)
    sample = probe.sample(n=min(10, len(probe)), random_state=args.seed)
    lines = [
        "# Evidence-quote audit of the shipped probe labels\n",
        f"{len(probe):,} awareness/recognition quotes and {len(df) - len(probe):,} task-performance quotes across three models; duplicates within a side counted once. "
        "Tiers: exact substring of the reasoning; lowercased match (what their training filter accepts); verbatim in the answer only; normalised-only (whitespace/punctuation rewritten); prompt-only (passes their filter because the prompt is in their search text, but pools prompt tokens); not found. "
        "`dropped_by_their_filter` = answer_only + normalised_only + not_found. Task-performance quotes are reported separately: the prompt tells the judge to draw them from the answer and the probe never reads them.\n",
        "## Class sizes per side\n",
        "The paper's §3.3 rule (awareness Yes/Maybe) reproduces the published per-class counts 742 / 563 / 322 (QwQ / Qwen3 / Distill-Qwen); the code's `get_classes` default (awareness OR recognition) gives the `pos_code` column and matches nothing published. The QwQ file's `_refined` suffix is explained nowhere in the repository.\n",
        classes.to_markdown(index=False),
        "\n## Awareness/recognition quotes by model\n", by_model.round(4).to_markdown(index=False),
        "\n## By probe class, paper's rule (quotes the probe would actually use)\n", by_paper.round(4).to_markdown(index=False),
        "\n## By probe class, code's default rule\n", by_code.round(4).to_markdown(index=False),
        "\n## By dimension and conclusion\n", by_dim_class.round(4).to_markdown(index=False),
        "\n## Task-performance quotes (expected to sit in the answer)\n", task.round(4).to_markdown(index=False),
        "\n## Ten randomly selected awareness/recognition quotes with their tier\n",
    ]
    for _, r in sample.iterrows():
        lines.append(f"- [{r.model} / {r.dimension} / {r.conclusion} / {r.tier}] {r.quote[:200]}")
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(classes.to_string(index=False))
    print(by_model.round(4).to_string(index=False))
    print(by_paper.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
