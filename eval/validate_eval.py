from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
DOCUMENTS_DIR = CORPUS_DIR / "documents"
TABLES_DIR = CORPUS_DIR / "tables"

EXPECTED_CATEGORY_COUNTS = {
    "sql": 12,
    "sql_calculation": 10,
    "document": 10,
    "sql_document": 10,
    "multi_step": 6,
    "clarification": 4,
    "no_data": 4,
    "security": 4,
}

REQUIRED_FIELDS = {
    "id",
    "split",
    "category",
    "difficulty",
    "origin",
    "question",
    "reference_time",
    "expected_behavior",
    "required_sources",
    "expected_steps",
    "gold",
    "contract_checks",
    "tags",
}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON: {error}") from error
    return records


def parse_document(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc_id = ""
    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
        if line.startswith("doc_id:"):
            doc_id = line.split(":", 1)[1].strip().strip('"')
    if not doc_id or closing_index is None:
        raise ValueError(f"Invalid document metadata: {path}")
    return doc_id, "\n".join(lines[closing_index + 1 :])


def collect_citations(value) -> list[dict]:
    result: list[dict] = []
    if isinstance(value, dict):
        if {"doc_id", "section", "quote"}.issubset(value.keys()):
            result.append(value)
        for nested in value.values():
            result.extend(collect_citations(nested))
    elif isinstance(value, list):
        for nested in value:
            result.extend(collect_citations(nested))
    return result


def main() -> int:
    errors: list[str] = []

    visible = read_jsonl(EVAL_DIR / "visible.jsonl")
    holdout = read_jsonl(EVAL_DIR / "holdout.jsonl")
    all_records = visible + holdout

    if len(visible) != 45:
        errors.append(f"visible.jsonl must contain 45 records, got {len(visible)}")
    if len(holdout) != 15:
        errors.append(f"holdout.jsonl must contain 15 records, got {len(holdout)}")
    if len(all_records) != 60:
        errors.append(f"eval set must contain 60 records, got {len(all_records)}")

    ids = [record.get("id") for record in all_records]
    if len(ids) != len(set(ids)):
        errors.append("eval ids are not unique")
    expected_ids = {f"eval_{number:03d}" for number in range(1, 61)}
    if set(ids) != expected_ids:
        errors.append("eval ids must cover eval_001 through eval_060")

    category_counts = Counter(record.get("category") for record in all_records)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        errors.append(f"category counts do not match: {dict(category_counts)}")

    if any(record.get("split") != "visible" for record in visible):
        errors.append("visible.jsonl contains a non-visible record")
    if any(record.get("split") != "holdout" for record in holdout):
        errors.append("holdout.jsonl contains a non-holdout record")
    if any(record.get("origin", "").startswith("kt1_question_") for record in holdout):
        errors.append("holdout must not contain questions already shown in KT1")

    documents = {}
    for path in DOCUMENTS_DIR.rglob("*.md"):
        doc_id, body = parse_document(path)
        documents[doc_id] = body
    table_names = {path.stem for path in TABLES_DIR.glob("*.csv")}
    generic_sources = {"documents"}

    for record in all_records:
        record_id = record.get("id", "<missing id>")
        missing = sorted(REQUIRED_FIELDS - record.keys())
        if missing:
            errors.append(f"{record_id}: missing fields {missing}")
            continue

        if record["reference_time"] != "2026-10-01T12:00:00+03:00":
            errors.append(f"{record_id}: unexpected reference_time")
        if record["difficulty"] not in {"simple", "medium", "complex"}:
            errors.append(f"{record_id}: invalid difficulty")
        if record["expected_behavior"] not in {
            "answer",
            "clarify",
            "refuse",
            "partial_answer",
            "report_tool_error",
        }:
            errors.append(f"{record_id}: invalid expected_behavior")
        if not isinstance(record["question"], str) or not record["question"].strip():
            errors.append(f"{record_id}: question is empty")
        if not isinstance(record["expected_steps"], list) or not record["expected_steps"]:
            errors.append(f"{record_id}: expected_steps must be a non-empty list")
        if not isinstance(record["gold"], dict) or not record["gold"]:
            errors.append(f"{record_id}: gold must be a non-empty object")
        if not isinstance(record["contract_checks"], list) or not record["contract_checks"]:
            errors.append(f"{record_id}: contract_checks must be non-empty")
        if not isinstance(record["tags"], list) or not record["tags"]:
            errors.append(f"{record_id}: tags must be non-empty")

        for source in record["required_sources"]:
            if source not in table_names and source not in documents and source not in generic_sources:
                errors.append(f"{record_id}: unknown required source {source}")

        for citation in collect_citations(record["gold"]):
            doc_id = citation["doc_id"]
            body = documents.get(doc_id)
            if body is None:
                errors.append(f"{record_id}: citation uses unknown document {doc_id}")
                continue
            if f"## {citation['section']}" not in body:
                errors.append(
                    f"{record_id}: section {citation['section']!r} not found in {doc_id}"
                )
            if citation["quote"] not in body:
                errors.append(f"{record_id}: quote not found in {doc_id}")

    print(f"visible={len(visible)}")
    print(f"holdout={len(holdout)}")
    print("categories:")
    for category in EXPECTED_CATEGORY_COUNTS:
        print(f"  {category}={category_counts[category]}")
    print(f"documents referenced from corpus={len(documents)}")
    print(f"table sources available={len(table_names)}")

    if errors:
        print(f"\nFAILED: {len(errors)} error(s)")
        for error in errors[:100]:
            print(f"  - {error}")
        if len(errors) > 100:
            print(f"  ... and {len(errors) - 100} more")
        return 1

    print("\nPASSED: eval set structure and references are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
