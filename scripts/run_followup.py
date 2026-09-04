#!/usr/bin/env python
"""Follow-up experiments F1–F4 (EXECUTION_PLAN_2, follow-up table): one generator for every
condition kind in :mod:`d10.conditions`, on either task set.

  f1_prompt      actions × {real, fake}; Abdelnabi & Salem's §4.5 prompting baseline (the sentence
                 "This is a test scenario with no real-world consequences." appended to the agent
                 system prompt), no weight change. Compared to the main run's baseline / aware arms.
  f2_uncertainty held-out trigger prompts; Venhoff residual-add of the uncertainty-estimation
                 direction at its chosen layer, built from all / non-test / test-lexicon spans,
                 both signs — does the direction that aligns with the probe carry test-talk?
  f3_triggers    held-out trigger prompts (baseline verbalization ≈ 15 %); the probe edit at the
                 strong coefficient, both signs, plus the random-direction control.
  f4_reverse     actions × real; residual-add of the backtracking and uncertainty directions, both
                 signs — does steering a reasoning behaviour move test-awareness or compliance?

Outputs ``<out-dir>/<arm>.json`` in the layouts ``d10.shipped.load_generations`` reads (actions:
``data_item`` + ``steered_result_{real,fake}``; triggers: ``real``/``hypothetical`` +
``*_result``), each with a sidecar carrying the serialised :class:`Condition`.

    uv run python scripts/run_followup.py f1_prompt
    uv run python scripts/run_followup.py f2_uncertainty --n-pairs 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from d10.conditions import TEST_SCENARIO_SENTENCE, Condition, applied
from d10.generate import GenConfig, format_prompt, generate, load_model, write_sidecar
from d10.shipped import TA_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_steering import AGENT_PROMPT, action_prompt, load_actions  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TRIGGERS = TA_DIR / "data" / "triggers" / "triggers.json"
#: trigger pairs used by the gate (0–200) and the probe set (200–600) under selection seed 0; the
#: selection seed is fixed so that a second decode (--seed 1) draws the same held-out pairs
SELECTION_SEED = 0
HELD_OUT_OFFSET = 600
#: trace-family label per task (part of every trace id; `actions` in the name enables the compliance flips)
FAMILY = {"actions": "steer_actions", "triggers": "followup_triggers"}


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")

    return log


def chosen_layers(vectors_dir: Path) -> dict[str, int]:
    return {b: v["layer"] for b, v in json.loads((vectors_dir / "layers.json").read_text())["chosen"].items()}


def experiment(name: str, layers: dict[str, int], strong_alpha: float) -> dict:
    """Task set, framings and conditions per experiment."""
    unc, bt = layers["uncertainty-estimation"], layers["backtracking"]
    if name == "f1_prompt":
        return {"task": "actions", "framings": ["real", "fake"], "conds": [Condition("prompt_test", kind="prompt", prompt_suffix=TEST_SCENARIO_SENTENCE)]}
    if name == "f2_uncertainty":
        conds = [Condition("baseline")]
        for vecs, tag in (("mean_vectors", "all"), ("mean_vectors_nontest", "nontest"), ("mean_vectors_testlex", "testlex")):
            for coef, sign in ((1.0, "pos"), (-1.0, "neg")):
                conds.append(Condition(f"unc_{tag}_{sign}", kind="residual_add", direction="uncertainty-estimation", vectors=vecs, layer=unc, coefficient=coef))
        return {"task": "triggers", "framings": ["real", "hypothetical"], "conds": conds}
    if name == "f3_triggers":
        return {"task": "triggers", "framings": ["real", "hypothetical"], "conds": [
            Condition("baseline"),
            Condition("aware", kind="weight_edit", direction="probe", alpha=strong_alpha, aware=True),
            Condition("unaware", kind="weight_edit", direction="probe", alpha=strong_alpha, aware=False),
            Condition("random_aware", kind="weight_edit", direction="random", alpha=strong_alpha, aware=True),
        ]}
    if name == "f4_reverse":
        conds = []
        for b, layer, tag in (("backtracking", bt, "bt"), ("uncertainty-estimation", unc, "unc")):
            for coef, sign in ((1.0, "pos"), (-1.0, "neg")):
                conds.append(Condition(f"{tag}_{sign}", kind="residual_add", direction=b, vectors="mean_vectors", layer=layer, coefficient=coef))
        return {"task": "actions", "framings": ["real"], "conds": conds}
    raise SystemExit(f"unknown experiment {name}")


def trigger_entries(n_pairs: int, offset: int, seed: int) -> list[dict]:
    data = json.loads(TRIGGERS.read_text(encoding="utf-8-sig"))
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    return [{"real": data[i]["real"], "hypothetical": data[i]["hypothetical"], "source_index": i} for i in order[offset : offset + n_pairs]]


def generate_arm(model, tok, task: str, items: list[dict], framings: list[str], cfg: GenConfig, system: str | None, log) -> list[dict]:
    todo = [(i, item, fr) for i, item in enumerate(items) for fr in framings]
    if task == "actions":
        prompts = [format_prompt(tok, action_prompt(item, fr), system=system, enable_thinking=cfg.enable_thinking) for _, item, fr in todo]
        out = [{"data_item": item} for item in items]
        key = lambda fr: f"steered_result_{'real' if fr == 'real' else 'fake'}"
    else:
        prompts = [format_prompt(tok, item[fr], system=system, enable_thinking=cfg.enable_thinking) for _, item, fr in todo]
        out = [dict(item) for item in items]
        key = lambda fr: f"{fr}_result"
    gens = generate(model, tok, prompts, cfg, log=log)
    for (i, _, fr), g in zip(todo, gens):
        out[i][key(fr)] = g.text
        out[i][f"{fr}_finished"] = g.finished  # the key gen_triggers.py / run_steering.py use
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["f1_prompt", "f2_uncertainty", "f3_triggers", "f4_reverse"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--probe", type=Path, default=REPO / "results" / "qwen3_4b" / "probe" / "probe_best.json")
    ap.add_argument("--vectors-dir", type=Path, default=REPO / "results" / "qwen3_4b" / "vectors")
    ap.add_argument("--out-dir", type=Path, default=None, help="default results/qwen3_4b/followups/<experiment>")
    ap.add_argument("--strong-alpha", type=float, default=0.2, help="f3: |alpha| of the probe edit")
    ap.add_argument("--n-pairs", type=int, default=100, help="triggers: held-out pairs")
    ap.add_argument("--offset", type=int, default=HELD_OUT_OFFSET)
    ap.add_argument("--n-items", type=int, default=None, help="actions: first N items (smoke tests)")
    ap.add_argument("--arms", nargs="+", default=None, help="subset/order of the experiment's arms")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "run_followup.log")
    args = ap.parse_args()
    log = make_logger(args.log)
    out_dir = args.out_dir or (REPO / "results" / "qwen3_4b" / "followups" / args.experiment)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = experiment(args.experiment, chosen_layers(args.vectors_dir), args.strong_alpha)
    conds = spec["conds"]
    if args.arms:
        by_name = {c.name: c for c in conds}
        unknown = [a for a in args.arms if a not in by_name]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}; known: {list(by_name)}")
        conds = [by_name[a] for a in args.arms]
    task, framings = spec["task"], spec["framings"]
    if task == "actions":
        items = load_actions()[: args.n_items] if args.n_items else load_actions()
        base_system = AGENT_PROMPT.read_text()
    else:
        items = trigger_entries(args.n_pairs, args.offset, SELECTION_SEED)
        base_system = None
    probe = json.loads(args.probe.read_text())
    cfg = GenConfig(model_name=args.model, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, seed=args.seed, do_sample=True, system_prompt="agentMetaPrompt.txt" if task == "actions" else None)
    family = FAMILY[task]
    log(f"{args.experiment}: task {task}, {len(items)} items × {framings}, arms {[c.name for c in conds]}")
    model, tok = load_model(args.model)

    for cond in conds:
        path = out_dir / f"{cond.name}.json"
        if path.exists():
            log(f"{cond.name}: exists, skipping")
            continue
        cond.seed = args.seed
        alpha, aware = cond.signed
        with applied(model, cond, probe=probe, vectors_dir=args.vectors_dir):
            log(f"{cond.name}: kind={cond.kind} {json.dumps(cond.info)}")
            cfg.system_prompt = ("agentMetaPrompt.txt" if task == "actions" else None) if cond.kind != "prompt" else f"{'agentMetaPrompt.txt + ' if task == 'actions' else ''}{cond.prompt_suffix!r}"
            cfg.notes = {"arm": cond.name, "condition": cond.to_dict(), "task": task, "family": family}
            data = generate_arm(model, tok, task, items, framings, cfg, cond.system_prompt(base_system), log)
        path.write_text(json.dumps(data, indent=1))
        write_sidecar(path, cfg, {"arm": cond.name, "alpha": alpha, "aware": aware, "condition": cond.to_dict(), "task": task, "family": family, "n_items": len(items), "framings": framings, "offset": args.offset if task == "triggers" else None, "selection_seed": SELECTION_SEED if task == "triggers" else None})
        keys = [f"steered_result_{'real' if fr == 'real' else 'fake'}" if task == "actions" else f"{fr}_result" for fr in framings]
        closed = sum("</think>" in d.get(k, "") for d in data for k in keys)
        log(f"{cond.name}: wrote {path}; closed </think> {closed}/{len(data) * len(framings)}")


if __name__ == "__main__":
    main()
