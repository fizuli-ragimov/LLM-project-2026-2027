from __future__ import annotations

import json
import sys
from collections import Counter

from build_eval import EVAL_VERSION, OUTPUT_DIR, REFERENCE_TIME, GroundTruth, build_records, section_text
from expectations import document_identity


VISIBLE_COUNTS = {
    "sql": 10, "sql_calculation": 8, "document": 8, "sql_document": 8,
    "multi_step": 4, "clarification": 3, "no_data": 3, "security": 1,
}
REQUIRED_FIELDS = {
    "id", "split", "eval_version", "category", "difficulty", "origin", "question",
    "reference_time", "expected_behavior", "required_sources", "expected_steps",
    "gold", "contract_checks", "tags",
}


def collect_citations(value):
    if isinstance(value, dict):
        if {"doc_id", "section", "quote"}.issubset(value):
            yield value
        for nested in value.values():
            yield from collect_citations(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from collect_citations(nested)


def resolve_pointer(value, pointer: str):
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid gold pointer {pointer!r}")
    for component in pointer[1:].split("/"):
        key = component.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def validate_step(step: dict, documents: dict, sources: set[str]) -> list[str]:
    kind = step["step"]
    expected = step["expected_result"]
    errors = []
    if kind == "sql":
        if not expected["sources"] or not set(expected["sources"]) <= sources:
            errors.append("unknown or missing SQL sources")
        columns = expected["columns"]
        if not columns or len(columns) != len(set(columns)) or not expected["rows"]:
            errors.append("SQL expectation must have unique columns and at least one row")
        elif any(set(row) != set(columns) for row in expected["rows"]):
            errors.append("SQL result row does not match the expected columns")
    elif kind == "calculate":
        if any(not expected[field] for field in ("operation", "formula", "inputs", "outputs")):
            errors.append("calculator expectation must have an operation, formula, inputs and outputs")
    elif kind == "retrieve_documents":
        ids = expected["doc_ids"]
        if len(ids) != len(set(ids)) or set(ids) - documents.keys():
            errors.append("unknown or duplicate required document")
        elif expected["document_keys"] != [document_identity(documents[doc_id]) for doc_id in ids]:
            errors.append("document keys do not match the corpus")
    return errors


def validate_record(record: dict, truth: GroundTruth) -> list[str]:
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        return [f"missing fields {sorted(missing)}"]
    errors = []
    if record["split"] != "visible" or record["eval_version"] != EVAL_VERSION:
        errors.append("unexpected split or eval version")
    if record["reference_time"] != REFERENCE_TIME:
        errors.append("unexpected reference time")
    if record["difficulty"] not in {"simple", "medium", "complex"}:
        errors.append("invalid difficulty")
    if record["expected_behavior"] not in {"answer", "clarify", "refuse", "partial_answer", "report_tool_error"}:
        errors.append("invalid expected behavior")
    if not record["question"].strip() or not record["contract_checks"] or not record["tags"]:
        errors.append("question, contract checks and tags must be non-empty")
    steps = record["expected_steps"]
    if not steps or not record["gold"]:
        errors.append("expected steps and gold must be non-empty")

    table_names = {"customers", "subscriptions", "payments", "support_tickets"}
    if set(record["required_sources"]) - (truth.documents.keys() | table_names | {"documents"}):
        errors.append("unknown required source")
    for index, step in enumerate(steps):
        errors.extend(f"step {index}: {error}" for error in validate_step(step, truth.documents, table_names))

    for citation in collect_citations(record["gold"]):
        try:
            document = truth.documents[citation["doc_id"]]
            section = section_text(document["body"], citation["section"])
            if citation["version"] != document["metadata"]["version"] or citation["quote"] not in section:
                errors.append(f"wrong citation version or quote: {citation['doc_id']}, {citation['section']}")
        except (KeyError, ValueError) as error:
            errors.append(f"invalid citation: {error}")

    claims = record.get("answer_checklist", [])
    if record["expected_behavior"] in {"answer", "partial_answer"} and not claims:
        errors.append("answer checklist is missing")
    if len({claim["id"] for claim in claims}) != len(claims):
        errors.append("duplicate claim IDs")
    for claim in claims:
        evidence = claim["evidence"]
        try:
            resolve_pointer(record["gold"], claim["gold_pointer"])
        except (IndexError, KeyError, TypeError, ValueError):
            errors.append(f"{claim['id']}: unresolved gold pointer")
        indices = evidence["step_indices"]
        if not indices or any(type(index) is not int or not 0 <= index < len(steps) for index in indices):
            errors.append(f"{claim['id']}: invalid evidence step indices")
        elif evidence["kind"] == "document":
            allowed = {
                doc_id for index in indices if steps[index]["step"] == "retrieve_documents"
                for doc_id in steps[index]["expected_result"]["doc_ids"]
            }
            if "doc_id" in evidence and evidence["doc_id"] not in allowed:
                errors.append(f"{claim['id']}: document is outside the expected retrieval step")
        elif evidence["kind"] not in {"data", "interpretation"}:
            errors.append(f"{claim['id']}: unknown evidence kind")
    return errors


def main() -> int:
    path = OUTPUT_DIR / "visible.jsonl"
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        truth = GroundTruth()
        generated = build_records(truth)
        errors = []
        if records != generated:
            errors.append("visible.jsonl differs from regenerated records")
        if len(records) != 45 or len({record["id"] for record in records}) != 45:
            errors.append("expected 45 unique visible cases")
        if Counter(record["category"] for record in records) != VISIBLE_COUNTS:
            errors.append("unexpected visible category counts")
        for record in records:
            errors.extend(f"{record['id']}: {error}" for error in validate_record(record, truth))
    except (OSError, IndexError, KeyError, TypeError, ValueError) as error:
        print(f"FAILED: {error}")
        return 1
    if errors:
        print("\n".join(errors))
        return 1
    print("PASSED: 45 visible cases match the corpus, step expectations and answer checklists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
