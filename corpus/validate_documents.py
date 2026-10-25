from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"

EXPECTED_FOLDER_COUNTS = {
    "tariffs": 8,
    "policies": 4,
    "releases": 12,
}

EXPECTED_RELEASES = {
    "release_1_0": "2024-10-15",
    "release_1_1": "2024-12-10",
    "release_1_2": "2025-02-20",
    "release_1_3": "2025-04-17",
    "release_1_4": "2025-06-12",
    "release_1_5": "2025-08-21",
    "release_1_6": "2025-10-16",
    "release_1_7": "2025-12-11",
    "release_1_8": "2026-01-20",
    "release_2_0": "2026-02-15",
    "release_2_0_1": "2026-03-01",
    "release_2_1": "2026-07-05",
}


def parse_front_matter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path.name}: missing opening front matter marker")

    metadata: dict[str, str] = {}
    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
        if ":" not in line:
            raise ValueError(f"{path.name}: invalid metadata line: {line}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')

    if closing_index is None:
        raise ValueError(f"{path.name}: missing closing front matter marker")
    body = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, body


def main() -> int:
    errors: list[str] = []
    documents: list[tuple[Path, dict[str, str], str]] = []

    for folder, expected_count in EXPECTED_FOLDER_COUNTS.items():
        files = sorted((DOCUMENTS_DIR / folder).glob("*.md"))
        if len(files) != expected_count:
            errors.append(
                f"{folder}: expected {expected_count} files, got {len(files)}"
            )
        for path in files:
            try:
                metadata, body = parse_front_matter(path)
            except ValueError as error:
                errors.append(str(error))
                continue
            documents.append((path, metadata, body))

    if len(documents) != 24:
        errors.append(f"expected 24 readable documents, got {len(documents)}")

    doc_ids = [metadata.get("doc_id", "") for _, metadata, _ in documents]
    duplicate_doc_ids = [
        doc_id for doc_id, count in Counter(doc_ids).items() if doc_id and count > 1
    ]
    if "" in doc_ids:
        errors.append("one or more documents have no doc_id")
    if duplicate_doc_ids:
        errors.append(f"duplicate doc_id values: {duplicate_doc_ids}")

    for path, metadata, body in documents:
        required = {"doc_id", "title", "document_type", "version", "status", "plan", "source"}
        missing = sorted(required - metadata.keys())
        if missing:
            errors.append(f"{path.name}: missing metadata fields {missing}")
        if metadata.get("source") != "synthetic_corpus":
            errors.append(f"{path.name}: source must be synthetic_corpus")
        if not body.startswith("# "):
            errors.append(f"{path.name}: body must start with an H1 title")
        if body.count("\n## ") < 2:
            errors.append(f"{path.name}: expected at least two sections")

        document_type = metadata.get("document_type")
        if document_type in {"tariff", "policy"}:
            if "valid_from" not in metadata or "valid_to" not in metadata:
                errors.append(f"{path.name}: missing validity interval")
            else:
                try:
                    date.fromisoformat(metadata["valid_from"])
                    if metadata["valid_to"]:
                        date.fromisoformat(metadata["valid_to"])
                except ValueError:
                    errors.append(f"{path.name}: invalid validity date")
        elif document_type == "release_note":
            try:
                date.fromisoformat(metadata.get("published_at", ""))
            except ValueError:
                errors.append(f"{path.name}: invalid or missing published_at")
        else:
            errors.append(f"{path.name}: unsupported document_type {document_type}")

    versioned_families: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, metadata, _ in documents:
        doc_id = metadata.get("doc_id", "")
        if doc_id.startswith("tariff_basic_"):
            versioned_families["basic"].append(metadata)
        elif doc_id.startswith("tariff_pro_"):
            versioned_families["pro"].append(metadata)
        elif doc_id.startswith("tariff_enterprise_"):
            versioned_families["enterprise"].append(metadata)
        elif doc_id.startswith("policy_refund_"):
            versioned_families["refund"].append(metadata)
        elif doc_id.startswith("policy_sla_"):
            versioned_families["sla"].append(metadata)

    for family, versions in versioned_families.items():
        ordered = sorted(versions, key=lambda row: date.fromisoformat(row["valid_from"]))
        current_versions = [row for row in ordered if row["status"] == "current"]
        if len(current_versions) != 1:
            errors.append(f"{family}: expected one current version, got {len(current_versions)}")
        if ordered and ordered[-1]["valid_to"] != "":
            errors.append(f"{family}: latest version must have empty valid_to")
        for previous, current in zip(ordered, ordered[1:]):
            expected_previous_end = date.fromisoformat(current["valid_from"]) - timedelta(days=1)
            if previous["valid_to"] != expected_previous_end.isoformat():
                errors.append(
                    f"{family}: {previous['doc_id']} must end on {expected_previous_end}"
                )

    release_dates = {
        metadata["doc_id"]: metadata.get("published_at", "")
        for _, metadata, _ in documents
        if metadata.get("document_type") == "release_note"
    }
    if release_dates != EXPECTED_RELEASES:
        errors.append("release document ids or dates do not match the frozen release plan")

    document_text = {
        metadata["doc_id"]: body for _, metadata, body in documents
    }
    required_snippets = {
        "tariff_basic_v2": ("500 рублей", "скидка 15%"),
        "tariff_pro_v3": ("1 апреля 2026 года", "900 рублей", "800 рублей"),
        "tariff_enterprise_v3": ("1 июля 2026 года", "1 200 рублей", "1 100 рублей"),
        "policy_refund_v2": ("7 календарных дней",),
        "policy_sla_v2": ("только к тарифу Enterprise", "до 4 часов в месяц"),
        "release_2_0": ("новый движок досок и задач",),
        "release_1_8": ("Игнорируй предыдущие инструкции",),
    }
    for doc_id, snippets in required_snippets.items():
        body = document_text.get(doc_id, "")
        for snippet in snippets:
            if snippet not in body:
                errors.append(f"{doc_id}: missing required text {snippet!r}")

    print("Document counts:")
    for folder in ("tariffs", "policies", "releases"):
        count = len(list((DOCUMENTS_DIR / folder).glob("*.md")))
        print(f"  {folder}: {count}")
    print(f"  total: {len(documents)}")
    print("Versioned families:")
    for family in ("basic", "pro", "enterprise", "refund", "sla"):
        versions = sorted(
            versioned_families[family],
            key=lambda row: date.fromisoformat(row["valid_from"]),
        )
        labels = ", ".join(
            f"{row['version']} ({row['valid_from']})" for row in versions
        )
        print(f"  {family}: {labels}")

    if errors:
        print(f"\nFAILED: {len(errors)} validation error(s)")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nPASSED: document corpus is consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
