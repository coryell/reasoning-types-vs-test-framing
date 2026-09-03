"""The two upstream judge prompts, reproduced verbatim.

Venhoff et al.'s annotation prompt is embedded here (their repo is gitignored under ``assets/``); a
test asserts it is byte-identical to the f-string in ``utils/utils.py`` whenever the clone is present.
Leading indentation is part of what they send and is kept.

Abdelnabi & Salem's judge prompt is read from their repo at call time because it is a 5.9 KB file with
``{{ }}``-escaped JSON braces that their code fills with ``str.format``; we do exactly the same.
"""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
VENHOFF_UTILS = ASSETS / "steering_thinking_llms" / "utils" / "utils.py"
ABDELNABI_JUDGE_PROMPT = ASSETS / "test_awareness_steering" / "prompts" / "judgePromptEvidenceBoth.txt"

#: Venhoff et al., ``utils.process_batch_annotations`` — the f-string body between ``chat(f"""`` and
#: ``""")``. The placeholder ``{thinking}`` is substituted with ``str.replace`` (not ``format``) so
#: braces inside a reasoning trace cannot break the call.
VENHOFF_ANNOTATION_TEMPLATE = """
        Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

        Available labels:
        0. initializing -> The model is rephrasing the given task and states initial thoughts.
        1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
        2. adding-knowledge -> The model is enriching the current approach with recalled facts.
        3. example-testing -> The model generates examples to test its current approach.
        4. uncertainty-estimation -> The model is stating its own uncertainty.
        5. backtracking -> The model decides to change its approach.

        The reasoning chain to analyze:
        {thinking}

        Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
        """

VENHOFF_LABELS = (
    "initializing",
    "deduction",
    "adding-knowledge",
    "example-testing",
    "uncertainty-estimation",
    "backtracking",
)


def venhoff_annotation_prompt(thinking: str) -> str:
    return VENHOFF_ANNOTATION_TEMPLATE.replace("{thinking}", thinking)


def extract_venhoff_template_from_source(utils_py: Path = VENHOFF_UTILS) -> str:
    """Pull the f-string body out of their ``utils.py`` so a test can check ours is identical."""
    src = utils_py.read_text()
    start_marker = 'annotated_response = chat(f"""'
    start = src.index(start_marker) + len(start_marker)
    end = src.index('""")', start)
    return src[start:end]


#: The three top-level keys their ``judgeIt_batch.py`` requires in the judge's JSON.
ABDELNABI_ELEMENTS = ("model_awareness", "hypothetical_recognition", "task_performance")


def load_abdelnabi_template(path: Path = ABDELNABI_JUDGE_PROMPT) -> str:
    return path.read_text()


def abdelnabi_judge_prompt(template: str, task: str, reasoning: str, answer: str) -> str:
    """Fill the template exactly as ``judgeIt_batch.prepare_data_for_batching`` does."""
    return template.format(task=task, reasoning=reasoning, answer=answer)
