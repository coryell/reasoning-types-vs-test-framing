#!/usr/bin/env python
"""Generate reasoning traces on Abdelnabi & Salem's trigger prompts with a local model.

Mirrors their ``genOutputs.py``: one user message per prompt, no system prompt, both the real and the
hypothetical variant of each entry, thinking enabled for Qwen3. Sampling follows Qwen3's recommended
thinking-mode settings (their trigger generations were sampled through vLLM; their steered outputs
were greedy — see ``d10.generate``).

Output mirrors their ``triggers_*_with_GPT_labels_evidence.json`` layout minus the labels
(``real_result`` / ``hypothetical_result`` per entry) so the same loaders and judge script apply.
Resumable: entries that already have both results are skipped.

    uv run python scripts/gen_triggers.py --n-pairs 200 --out results/qwen3_4b/triggers_gate.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from d10.generate import GenConfig, format_prompt, generate, load_model, write_sidecar
from d10.shipped import TA_DIR

REPO = Path(__file__).resolve().parents[1]
TRIGGERS = TA_DIR / "data" / "triggers" / "triggers.json"


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
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N pairs of the shuffled order (to extend a sample)")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed for pair selection and torch seed")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=6, help="6 keeps a 4B model + KV cache under the 4070 Ti's ~10.8 GB free")
    ap.add_argument("--greedy", action="store_true", help="do_sample=False (their steered-output setting)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "gen_triggers.log")
    args = ap.parse_args()
    log = make_logger(args.log)

    data = json.loads(TRIGGERS.read_text(encoding="utf-8-sig"))
    order = list(range(len(data)))
    random.Random(args.seed).shuffle(order)
    chosen = order[args.offset : args.offset + args.n_pairs]
    entries = [{**data[i], "source_index": i} for i in chosen]
    log(f"triggers: {len(data)} pairs on disk; selected {len(entries)} (seed {args.seed}, offset {args.offset})")

    if args.out.exists():
        prev = {e["source_index"]: e for e in json.loads(args.out.read_text()) if "source_index" in e}
        for e in entries:
            if e["source_index"] in prev and all(isinstance(prev[e["source_index"]].get(k), str) for k in ("real_result", "hypothetical_result")):
                e.update(prev[e["source_index"]])
        log(f"resume: {sum('real_result' in e for e in entries)} entries already generated")

    cfg = GenConfig(model_name=args.model, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, seed=args.seed, do_sample=not args.greedy)
    todo = [(e, side) for e in entries for side in ("real", "hypothetical") if not isinstance(e.get(f"{side}_result"), str)]
    if todo:
        log(f"loading {args.model}")
        model, tok = load_model(args.model, cfg.dtype)
        prompts = [format_prompt(tok, e[side], system=None, enable_thinking=cfg.enable_thinking) for e, side in todo]
        log(f"generating {len(prompts)} sides, max_new_tokens={cfg.max_new_tokens}, batch={cfg.batch_size}, sample={cfg.do_sample}")
        gens = generate(model, tok, prompts, cfg, log=log)
        for (e, side), g in zip(todo, gens):
            e[f"{side}_result"] = g.text
            e[f"{side}_finished"] = g.finished
            e[f"{side}_n_new_tokens"] = g.n_new_tokens
        unfinished = sum(not g.finished for g in gens)
        log(f"done: {len(gens)} sides, {unfinished} hit max_new_tokens, {sum('</think>' in g.text for g in gens)} closed </think>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, indent=1))
    side = write_sidecar(args.out, cfg, {"n_pairs": len(entries), "offset": args.offset, "selection_seed": args.seed, "source": str(TRIGGERS)})
    log(f"wrote {args.out} and {side}")


if __name__ == "__main__":
    main()
