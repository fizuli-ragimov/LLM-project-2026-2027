from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from build_eval import DOCUMENTS_DIR, ROOT, parse_front_matter
from expectations import document_identity


SCORED_STEPS = {"sql", "calculate", "retrieve_documents"}


def normalized(value):
    if isinstance(value, list):
        return "array", tuple(normalized(item) for item in value)
    if isinstance(value, dict):
        return "object", tuple((key, normalized(item)) for key, item in sorted(value.items()))
    if type(value) in {int, float, Decimal}:
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("Trace contains a non-finite number")
        return "number", number
    return type(value).__name__, value


def row_multiset(rows: list[dict]) -> Counter:
    return Counter(normalized(row) for row in rows)


def same_value(expected, actual) -> bool:
    if isinstance(expected, list) and isinstance(actual, list):
        return row_multiset(expected) == row_multiset(actual)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            same_value(value, actual[key]) for key, value in expected.items()
        )
    return normalized(expected) == normalized(actual)


def key_tuple(key: dict) -> tuple[str, str]:
    return key["path"], key["version"]


def corpus_document_keys() -> dict[str, tuple[str, str]]:
    keys = {}
    for path in DOCUMENTS_DIR.rglob("*.md"):
        metadata, _ = parse_front_matter(path)
        key = document_identity({"path": path.relative_to(ROOT).as_posix(), "metadata": metadata})
        keys[metadata["doc_id"]] = key_tuple(key)
    return keys


def score_trace(record: dict, trace: dict, document_keys: dict[str, tuple[str, str]]) -> list[dict]:
    if trace["id"] != record["id"]:
        raise ValueError("Trace and eval record IDs differ")
    observed = {}
    for step in trace["steps"]:
        index = step["step_index"]
        if type(index) is not int or not 0 <= index < len(record["expected_steps"]) or index in observed:
            raise ValueError(f"Invalid or duplicate step index: {index}")
        if step["step"] != record["expected_steps"][index]["step"]:
            raise ValueError(f"Step type differs at index {index}")
        if step.get("scope") != record["expected_steps"][index].get("scope"):
            raise ValueError(f"Step scope differs at index {index}")
        observed[index] = step.get("result")

    results = []
    for index, step in enumerate(record["expected_steps"]):
        kind = step["step"]
        if kind not in SCORED_STEPS:
            continue
        expected = step["expected_result"]
        actual = observed.get(index)
        result = {"step_index": index, "step": kind}
        if kind == "sql":
            matches = (
                isinstance(actual, dict) and actual.get("period") == expected["period"]
                and isinstance(actual.get("rows"), list)
                and row_multiset(actual["rows"])
                == row_multiset(expected["rows"])
            )
            result["correct"] = matches
        elif kind == "calculate":
            matches = (
                isinstance(actual, dict) and actual.get("operation") == expected["operation"]
                and same_value(expected["inputs"], actual.get("inputs"))
                and same_value(expected["outputs"], actual.get("outputs"))
            )
            result["correct"] = matches
        else:
            ids = actual.get("ranked_doc_ids") if isinstance(actual, dict) else None
            required = {key_tuple(key) for key in expected["document_keys"]}
            if ids is None:
                ids = []
            if not isinstance(ids, list) or any(not isinstance(doc_id, str) for doc_id in ids):
                raise ValueError(f"ranked_doc_ids must be a list of strings at step {index}")
            # A repeated hit occupies its original rank but contributes at most once.
            found = {document_keys.get(doc_id) for doc_id in ids[:5]}
            result.update(hits=len(found & required), required=len(required))
            if not required:
                result["scope_logged"] = isinstance(actual, dict) and "ranked_doc_ids" in actual
        results.append(result)
    return results


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SQL, calculator and ranked document retrieval traces.")
    parser.add_argument("trace", type=Path, help="JSONL with one trace per evaluated question")
    parser.add_argument("--eval", type=Path, default=Path(__file__).with_name("visible.jsonl"))
    args = parser.parse_args()
    traces = read_records(args.trace)
    records = {record["id"]: record for record in read_records(args.eval)}
    trace_ids = [trace["id"] for trace in traces]
    if not traces or len(trace_ids) != len(set(trace_ids)) or set(trace_ids) - records.keys():
        parser.error("Provide a non-empty set of traces with unique, known eval IDs")

    document_keys = corpus_document_keys()
    scores = {trace["id"]: score_trace(records[trace["id"]], trace, document_keys) for trace in traces}
    overall = {}
    for kind in sorted(SCORED_STEPS):
        steps = [step for trace_scores in scores.values() for step in trace_scores if step["step"] == kind]
        if kind == "retrieve_documents":
            numerator = sum(step["hits"] for step in steps)
            denominator = sum(step["required"] for step in steps)
        else:
            numerator = sum(step["correct"] for step in steps)
            denominator = len(steps)
        overall[kind] = {"matched": numerator, "expected": denominator, "score": numerator / denominator if denominator else None}
    print(json.dumps({"questions_scored": len(scores), "metrics": overall, "by_question": scores}, indent=2))


if __name__ == "__main__":
    main()
