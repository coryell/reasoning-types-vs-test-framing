"""Completeness check of the Venhoff-style sentence annotations.

For every ``*.annotations.jsonl`` under ``results/`` (the shipped 32B outputs under
``results/annotations/`` and our own runs), each annotated trace is compared with the reasoning it
was made from:

* ``coverage``: words inside spans that occur verbatim in the reasoning (whitespace-normalised),
  divided by the reasoning's words. Below 1 means the annotator skipped text.
* ``not_found``: spans whose text does not occur in the reasoning (hallucinated or rewritten).
* ``out_of_order``: a span that occurs in the reasoning only before the previous span's position.
* ``extra``: a span text returned more times than it occurs in the reasoning (duplicated).

Writes ``results/report/T13_annotation_coverage.{csv,md}`` with one row per file and totals for the
traces the report uses (the 32B ``actions`` families and every own run) and for everything annotated.
"""
from __future__ import annotations
import collections, glob, json, os, re, statistics
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPAN = re.compile(r'\["([a-z\-]+)"\]\s*(.*?)\s*\["end-section"\]', re.S)

def norm(s: str) -> str:
    return " ".join((s or "").split())

def check_trace(reasoning: str, annotated: str) -> dict:
    reas = norm(reasoning)
    words = len(reas.split())
    spans = [norm(t) for _, t in SPAN.findall(annotated or "")]
    found = [s for s in spans if s and s in reas]
    covered = sum(len(s.split()) for s in found)
    cursor, out_of_order = 0, 0
    for s in found:
        j = reas.find(s, cursor)
        if j < 0:
            out_of_order += 1
        else:
            cursor = j + len(s)
    counts = collections.Counter(found)
    extra = sum(max(0, n - reas.count(s)) for s, n in counts.items())
    return dict(words=words, n_spans=len(spans), not_found=len(spans) - len(found),
                coverage=covered / words if words else float("nan"),
                out_of_order=out_of_order, extra=extra)

def main() -> None:
    files = sorted(glob.glob(f"{ROOT}/results/annotations/**/*.jsonl", recursive=True)
                   + glob.glob(f"{ROOT}/results/**/*.annotations.jsonl", recursive=True))
    rows, per_trace = [], []
    for path in files:
        rel = os.path.relpath(path, ROOT)
        recs = []
        n_failed = 0
        for line in open(path):
            r = json.loads(line)
            text = r.get("text")
            reasoning = (r.get("meta") or {}).get("reasoning", "")
            if not text or r.get("error"):
                n_failed += 1
                continue
            c = check_trace(reasoning, text)
            c["file"] = rel
            c["used_in_report"] = ("/annotations/" not in rel) or ("/actions" in rel)
            recs.append(c)
        per_trace.extend(recs)
        if not recs:
            continue
        cov = [c["coverage"] for c in recs if c["words"]]
        rows.append(dict(
            file=rel, n_traces=len(recs), n_annotation_failed=n_failed,
            median_coverage=statistics.median(cov),
            share_under_90=sum(c < 0.9 for c in cov) / len(cov),
            share_under_70=sum(c < 0.7 for c in cov) / len(cov),
            spans_not_found=sum(c["not_found"] for c in recs) / max(1, sum(c["n_spans"] for c in recs)),
            traces_with_out_of_order=sum(c["out_of_order"] > 0 for c in recs) / len(recs),
            traces_with_extra=sum(c["extra"] > 0 for c in recs) / len(recs),
            used_in_report=recs[0]["used_in_report"]))
    df = pd.DataFrame(rows)
    pt = pd.DataFrame(per_trace)

    def total(sub: pd.DataFrame, label: str) -> dict:
        cov = sub.coverage.dropna()
        return dict(file=label, n_traces=len(sub), n_annotation_failed=float("nan"),
                    median_coverage=cov.median(), share_under_90=(cov < 0.9).mean(),
                    share_under_70=(cov < 0.7).mean(),
                    spans_not_found=sub.not_found.sum() / sub.n_spans.sum(),
                    traces_with_out_of_order=(sub.out_of_order > 0).mean(),
                    traces_with_extra=(sub.extra > 0).mean(), used_in_report=None)
    totals = pd.DataFrame([total(pt[pt.used_in_report], "TOTAL, traces the report uses"),
                           total(pt, "TOTAL, every annotated trace")])
    out = pd.concat([df, totals], ignore_index=True)
    os.makedirs(f"{ROOT}/results/report", exist_ok=True)
    out.to_csv(f"{ROOT}/results/report/T13_annotation_coverage.csv", index=False)
    with open(f"{ROOT}/results/report/T13_annotation_coverage.md", "w") as f:
        f.write("# [13] Completeness of the sentence annotations: coverage of the reasoning by verbatim spans, spans not found, spans out of order, duplicated spans\n\n")
        f.write("One row per annotation file; totals over the traces the report uses (32B `actions` families and every own run) and over everything annotated. Shares are per trace unless the column says spans.\n\n")
        f.write(out.to_markdown(index=False, floatfmt=".3f"))
    print(out.tail(2).to_string(index=False))
    print(f"\n{len(files)} files, {len(pt)} traces checked; written to results/report/T13_annotation_coverage.{{csv,md}}")

if __name__ == "__main__":
    main()
