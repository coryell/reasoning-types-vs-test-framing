#!/usr/bin/env python
"""Choose the steering coefficients from the α sweep (EXECUTION_PLAN_2 §M3): verbalization is the
manipulation check, coherence the guard.

Rule, per side: among arms whose ``</think>`` closure rate ≥ ``--min-closure`` and mean 4-gram
repetition ≤ ``--max-repetition``, take the arm with the largest verbalization change in the intended
direction (aware: highest positive rate; unaware: lowest). Ties go to the smaller |α|. If nothing
qualifies, the smallest α is chosen and flagged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from d10 import parse
from d10.awareness import is_positive, parse_judgment
from d10.judge import load_results, record_ok
from d10.shipped import load_generations, parse_arm

REPO = Path(__file__).resolve().parents[1]


def arm_files(sweep_dir: Path) -> list[Path]:
    """Generation files only: skip the .gen.json and .stats.json sidecars and require a sidecar."""
    return [p for p in sorted(sweep_dir.glob("*.json")) if not p.name.endswith((".gen.json", ".stats.json")) and p.with_suffix(p.suffix + ".gen.json").exists()]


def arm_stats(path: Path, model_name: str) -> dict:
    arm = path.stem
    traces = load_generations(path, model_name, "sweep", arm)
    closed = sum(t.has_think_close for t in traces) / len(traces)
    rep = sum(parse.repetition_rate(t.reasoning) for t in traces) / len(traces)
    jud = path.with_suffix(path.suffix + ".awareness.jsonl")
    recs = [r for r in load_results(jud).values() if record_ok(r)] if jud.exists() else []
    pos = sum(is_positive(parse_judgment(r["text"])) for r in recs)
    a = parse_arm(arm + ".json") if arm != "baseline" else None
    return {"arm": arm, "alpha": a.alpha if a else 0.0, "aware": a.aware if a else None, "n": len(traces), "closure": closed, "repetition": rep, "n_judged": len(recs), "verbalization": pos / len(recs) if recs else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "sweep")
    ap.add_argument("--model-name", default="qwen3_4b")
    ap.add_argument("--min-closure", type=float, default=0.9)
    ap.add_argument("--max-repetition", type=float, default=0.03)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    stats = [arm_stats(p, args.model_name) for p in arm_files(args.sweep_dir)]
    base = next(s for s in stats if s["arm"] == "baseline")
    chosen = {}
    for aware in (True, False):
        cands = [s for s in stats if s["aware"] is aware and s["closure"] >= args.min_closure and s["repetition"] <= args.max_repetition and s["n_judged"] > 0]
        flagged = False
        if not cands:
            cands = [s for s in stats if s["aware"] is aware]
            flagged = True
        key = (lambda s: (s["verbalization"], -s["alpha"])) if aware else (lambda s: (-s["verbalization"], -s["alpha"]))
        best = max(cands, key=key)
        chosen["aware" if aware else "unaware"] = {**best, "flagged_no_qualifying_arm": flagged, "baseline_verbalization": base["verbalization"]}
    out = args.out or args.sweep_dir / "chosen.json"
    out.write_text(json.dumps({"arms": stats, "chosen": chosen, "rule": {"min_closure": args.min_closure, "max_repetition": args.max_repetition}}, indent=1))
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
