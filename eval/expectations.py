from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal


def select(row: dict, *fields: str) -> dict:
    return {field: row[field] for field in fields}


def period_rows(before, after, periods: tuple[str, str], field: str = "value") -> list[dict]:
    return [
        {"period": period, **(value if isinstance(value, dict) else {field: value})}
        for period, value in zip(periods, (before, after), strict=True)
    ]


def event_rows(events: list[dict], days: int, *, include_churn: bool) -> list[dict]:
    rows = []
    for event in events:
        anchor = date.fromisoformat(event["event_date"])
        for side, offset in (("before", -days), ("after", 0)):
            start = anchor + timedelta(days=offset)
            end = start + timedelta(days=days - 1)
            values = (
                {"revenue_rub": event[f"revenue_{side}"], **event[f"churn_{side}"]}
                if include_churn else {"subscription_starts": event[side]}
            )
            rows.append({
                "plan": event["plan"], "window": side,
                "start": start.isoformat(), "end": end.isoformat(), **values,
            })
    return rows


def sql_rows(number: int, gold: dict, extras: dict[int, list[dict]], period: str) -> list[dict]:
    if number in extras:
        return extras[number]
    if number in {1, 2, 21, 22, 27}:
        return [select(gold, "value")]
    if number in {3, 24, 28}:
        dimension, field, values = {
            3: ("plan", "revenue_rub", "all_values"),
            24: ("plan", "average_licensed_users", "value"),
            28: ("category", "ticket_count", "value"),
        }[number]
        return [{dimension: key, field: value} for key, value in sorted(gold[values].items())]
    if number in {4, 32}:
        return [select(gold, "value", "numerator", "denominator")]
    if number == 31:
        return [{"average_resolution_hours": gold["value"], "closed_tickets": gold["denominator"]}]
    if number in {5, 29, 34, 41}:
        field = "new_paying_customers" if number == 34 else "revenue_rub"
        before, after = period_rows(gold["before"], gold["after"], tuple(period.split("/")), field)
        after["change_percent"] = gold["change_percent"]
        before["change_percent"] = None
        return [before, after]
    if number == 6:
        return [{"segment": segment, **gold[segment]} for segment in ("small_business", "enterprise")]
    if number == 7:
        return gold["all_values"]
    if number in {13, 44}:
        return period_rows(gold["before"], gold["after"], tuple(period.split("/")))
    if number in {14, 43, 46}:
        if number == 46:
            periods = ("2025-12-01/2026-02-28", "2026-03-01/2026-05-31")
            field = "refund_count"
        else:
            periods = ("2026-01-17/2026-02-14", "2026-02-15/2026-03-15")
            field = "ticket_count"
        return period_rows(gold["before"], gold["after"], periods, field)
    if number == 15:
        return event_rows(gold["events"], 30, include_churn=False)
    if number == 16:
        return [
            {"period": period, "segment": "enterprise", **gold["retention"][key]}
            for period, key in (("2026-Q2", "Q2_2026"), ("2026-Q3", "Q3_2026"))
        ]
    if number == 48:
        return [select(row, "segment", "tickets", "active_customers") for row in gold["ranking"]]
    if number == 50:
        return event_rows(gold["events"], 90, include_churn=True)
    raise ValueError(f"Missing SQL expectation for eval_{number:03d}")


def calculator(number: int, operation: str, gold: dict) -> dict:
    if number == 48:
        return {
            "operation": operation,
            "formula": "100 * tickets / active_customers, or 0 when active_customers is 0",
            "inputs": [select(row, "segment", "tickets", "active_customers") for row in gold["ranking"]],
            "outputs": [select(row, "segment", "tickets_per_100") for row in gold["ranking"]],
        }
    if number != 50:
        raise ValueError(f"Missing calculator expectation for eval_{number:03d}")
    inputs = [
        {
            **select(row, "plan", "revenue_before", "revenue_after"),
            "churn_before_percent": row["churn_before"]["rate_percent"],
            "churn_after_percent": row["churn_after"]["rate_percent"],
        }
        for row in gold["events"]
    ]
    outputs = []
    for row in inputs:
        revenue_delta = Decimal(str(row["revenue_after"])) - Decimal(str(row["revenue_before"]))
        churn_delta = Decimal(str(row["churn_after_percent"])) - Decimal(str(row["churn_before_percent"]))
        outputs.append({
            "plan": row["plan"], "revenue_delta_rub": float(revenue_delta),
            "churn_delta_pp": float(churn_delta),
            "absolute_revenue_delta_rub": float(abs(revenue_delta)),
            "absolute_churn_delta_pp": float(abs(churn_delta)),
        })
    leaders = {
        "largest_revenue_change": max(outputs, key=lambda row: row["absolute_revenue_delta_rub"])["plan"],
        "largest_churn_change": max(outputs, key=lambda row: row["absolute_churn_delta_pp"])["plan"],
    }
    gold["changes"] = outputs
    gold["comparison"] = leaders
    return {
        "operation": operation,
        "formula": "after - before; compare the absolute values for each metric separately",
        "inputs": inputs,
        "outputs": {"by_plan": outputs, **leaders},
    }


def document_identity(document: dict) -> dict:
    return {"path": document["path"], "version": document["metadata"]["version"]}


def retrieval_expectation(doc_ids: list[str], documents: dict) -> dict:
    return {
        "doc_ids": doc_ids,
        "document_keys": [document_identity(documents[doc_id]) for doc_id in doc_ids],
    }


def churn_event_windows(gold: dict, documents: dict) -> list[dict]:
    windows = []
    for change in gold["strongest_changes"]:
        month = change["month"]
        previous = (date.fromisoformat(month + "-01") - timedelta(days=1)).strftime("%Y-%m")
        events = []
        for doc_id, doc in sorted(documents.items()):
            metadata = doc["metadata"]
            if metadata["document_type"] not in {"release_note", "tariff"}:
                continue
            effective = metadata.get("published_at") or metadata["valid_from"]
            if effective[:7] in {previous, month}:
                section = EVENT_SECTIONS.get(
                    doc_id, "Стоимость" if metadata["document_type"] == "tariff" else "Основные изменения",
                )
                heading = f"## {section}\n"
                details = doc["body"].split(heading, 1)[1].split("\n## ", 1)[0].strip()
                events.append({
                    "doc_id": doc_id, "date": effective, "title": metadata["title"],
                    "section": section, "change": details,
                })
        windows.append({"previous_month": previous, "month": month, "events": events})
    return windows


# Only claims needed to answer the question. The remaining gold fields are diagnostic context.
CLAIM_FIELDS = {
    1: "value", 2: "value", 3: "top", 4: "value", 5: "before after change_percent",
    6: "small_business enterprise", 7: "top_three", 8: "matches", 9: "facts",
    10: "date value", 11: "added_restrictions", 12: "releases",
    13: "before after change_percentage_points citation causality",
    14: "before after change_absolute change_percent release_date causality",
    15: "events strongest causality", 16: "retention related_events causality",
    17: "top_three related_changes_in_q3 segment_interpretation pre_q3_context causality",
    18: "strongest_changes event_windows causality",
    21: "value", 22: "value", 24: "value", 25: "value", 27: "value", 28: "value",
    29: "before after change_percent", 31: "value", 32: "value", 34: "before after change_percent",
    35: "facts", 37: "changes", 38: "value", 40: "facts",
    41: "before after change_percent tariff_change causality",
    43: "before after change_percent causality",
    44: "before after related_events causality", 46: "before after change_percent causality",
    48: "top_segment top_rate release causality", 50: "events changes comparison causality",
}
DOCUMENT_FIELDS = {
    "facts", "date", "added_restrictions", "releases", "citation", "release_date", "related_events",
    "related_changes_in_q3", "segment_interpretation", "pre_q3_context", "event_windows",
    "tariff_change", "release",
}
EVENT_SECTIONS = {"release_2_0_1": "Исправления"}
EVENT_SUMMARIES = {
    "policy_sla_v2": "SLA действует только для Enterprise, изменены целевое время реакции и правила учёта доступности.",
    "tariff_enterprise_v3": "Цена выросла с 1 100 до 1 200 рублей за лицензию в месяц.",
    "release_2_1": "Обновлены роли, аудит и политики доступа Enterprise.",
}


def checklist(record: dict, number: int) -> list[dict]:
    gold = record["gold"]
    steps_by_type = {}
    for index, step in enumerate(record["expected_steps"]):
        steps_by_type.setdefault(step["step"], []).append(index)
    claims = []
    for field in CLAIM_FIELDS[number].split():
        value = gold[field]
        if field == "event_windows":
            entries = []
            for index, window in enumerate(value):
                window_pointer = f"/event_windows/{index}"
                if window["events"]:
                    entries.extend(
                        (f"{window_pointer}/events/{event_index}", event)
                        for event_index, event in enumerate(window["events"])
                    )
                else:
                    entries.append((window_pointer, window))
        elif isinstance(value, list) and value:
            entries = [(f"/{field}/{index}", item) for index, item in enumerate(value)]
        else:
            entries = [(f"/{field}", value)]
        for pointer, item in entries:
            if field == "causality":
                evidence = {"kind": "interpretation", "step_indices": steps_by_type.get("sql", []) + steps_by_type.get("retrieve_documents", [])}
            elif "sql" not in steps_by_type or field in DOCUMENT_FIELDS:
                evidence = {"kind": "document", "step_indices": steps_by_type["retrieve_documents"]}
                if isinstance(item, dict) and "doc_id" in item:
                    evidence.update(select(item, "doc_id"))
                    if "section" in item:
                        evidence["section"] = item["section"]
                if field == "event_windows":
                    window_index = int(pointer.split("/")[2])
                    evidence["step_indices"] = [steps_by_type["retrieve_documents"][window_index]]
            else:
                indices = steps_by_type["sql"]
                if field in {"top_segment", "top_rate", "changes", "comparison"}:
                    indices = steps_by_type["calculate"]
                if field == "events":
                    indices = indices + steps_by_type.get("retrieve_documents", [])
                evidence = {"kind": "data", "step_indices": indices}
            claims.append({"id": f"claim_{len(claims) + 1:02d}", "gold_pointer": pointer, "evidence": evidence})
    return claims


def enrich_records(records: list[dict], documents: dict, extras: dict[int, list[dict]]) -> None:
    sql_metadata = {
        17: ("2026-Q2/2026-Q3", ["customers", "subscriptions"]),
        18: ("2025-10/2026-09", ["customers", "subscriptions"]),
        48: ("2026-Q3", ["customers", "subscriptions", "support_tickets"]),
        50: ("90-day windows", ["customers", "subscriptions", "payments"]),
    }
    for record in records:
        number = int(record["id"][-3:])
        gold = record["gold"]
        if number in {16, 44}:
            gold["related_events"] = [
                {"doc_id": doc_id, "date": documents[doc_id]["metadata"].get("valid_from")
                 or documents[doc_id]["metadata"]["published_at"], "change": change}
                for doc_id, change in EVENT_SUMMARIES.items()
            ]
        if number == 48:
            gold["top_rate"] = gold["ranking"][0]
            gold["release"] = {
                "doc_id": "release_2_1", "date": documents["release_2_1"]["metadata"]["published_at"],
                "change": EVENT_SUMMARIES["release_2_1"],
            }
        if number == 18:
            gold.pop("candidate_documents")
            gold["event_windows"] = churn_event_windows(gold, documents)
        steps = []
        for step in record["expected_steps"]:
            step_type = step["step"]
            expected = step["expected_result"]
            if step_type == "sql":
                if isinstance(expected, str):
                    period, sources = sql_metadata[number]
                    expected = {"metric": expected, "sources": sources, "period": period}
                rows = sql_rows(number, gold, extras, expected["period"])
                expected = {**expected, "columns": list(rows[0]), "rows": rows}
                if number == 48:
                    expected["active_customers_as_of"] = "2026-07-01"
                if number in {14, 43}:
                    expected["category"] = "all"
            elif step_type == "calculate":
                expected = calculator(number, expected, gold)
            elif step_type == "retrieve_documents":
                if number == 18:
                    for window in gold["event_windows"]:
                        steps.append({
                            "step": step_type, "scope": window["month"],
                            "expected_result": retrieval_expectation(
                                [event["doc_id"] for event in window["events"]], documents,
                            ),
                        })
                    continue
                expected = retrieval_expectation(expected["doc_ids"], documents)
            steps.append({**step, "expected_result": expected})
        record["expected_steps"] = steps
        if record["expected_behavior"] in {"answer", "partial_answer"}:
            record["answer_checklist"] = checklist(record, number)
