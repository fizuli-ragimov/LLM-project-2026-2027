from __future__ import annotations

import csv
import re
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
DATA_START = date(2024, 10, 1)
DATA_END = date(2026, 9, 30)
TOKEN = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
LIMITS = {"sql": (4, 0.5), "retrieve_documents": (3, 0.2), "calculate": (2, 0.1)}

TABLES = {
    "customers": "customer_id INTEGER, segment TEXT, region TEXT",
    "subscriptions": """subscription_id INTEGER, customer_id INTEGER, plan TEXT,
        started_at TEXT, ended_at TEXT, licensed_users INTEGER, billing_cycle TEXT""",
    "payments": """subscription_id INTEGER, paid_at TEXT, amount_rub REAL,
        operation_type TEXT, status TEXT, billing_period_start TEXT""",
    "support_tickets": "customer_id INTEGER, created_at TEXT, closed_at TEXT, category TEXT, status TEXT",
}

NEW_PAYING = """WITH first_charge AS (
    SELECT s.customer_id, MIN(p.paid_at) AS paid_at
    FROM payments p JOIN subscriptions s USING (subscription_id)
    WHERE p.operation_type = 'charge' AND p.status = 'succeeded'
    GROUP BY s.customer_id
) SELECT COUNT(*) AS value FROM first_charge WHERE paid_at BETWEEN :start AND :end"""

REVENUE = """WITH RECURSIVE offset(n) AS (
    SELECT 0 UNION ALL SELECT n+1 FROM offset WHERE n < 11
), allocation AS (
    SELECT s.plan, c.region,
        date(
            p.billing_period_start, 'start of month', printf('+%d months', n),
            printf('+%d days', MIN(
                CAST(strftime('%d', p.billing_period_start) AS INTEGER),
                CAST(strftime('%d', date(p.billing_period_start, 'start of month',
                    printf('+%d months', n + 1), '-1 day')) AS INTEGER)
            ) - 1)
        ) AS month,
        (CASE p.operation_type WHEN 'refund' THEN -p.amount_rub ELSE p.amount_rub END)
            / CASE s.billing_cycle WHEN 'annual' THEN 12.0 ELSE 1 END AS amount
    FROM payments p JOIN subscriptions s USING (subscription_id) JOIN customers c USING(customer_id)
    CROSS JOIN offset WHERE p.status = 'succeeded' AND (s.billing_cycle = 'annual' OR n = 0)
)"""
CHURN = """WITH active AS (
    SELECT DISTINCT s.customer_id FROM subscriptions s JOIN customers c USING(customer_id)
    WHERE s.started_at <= :start AND (s.ended_at IS NULL OR s.ended_at >= :start)
    AND (:plan IS NULL OR s.plan = :plan) AND (:segment IS NULL OR c.segment = :segment)
), events AS (
    SELECT DISTINCT s.customer_id FROM subscriptions s JOIN customers c USING(customer_id)
    WHERE s.ended_at BETWEEN :start AND :end
    AND (:plan IS NULL OR s.plan = :plan) AND (:segment IS NULL OR c.segment = :segment)
)
SELECT COUNT(*) AS active_at_start, COUNT(events.customer_id) AS cancelled,
    COALESCE(ROUND(100.0 * COUNT(events.customer_id) / COUNT(*), 2), 0.0) AS rate_percent
FROM active LEFT JOIN events USING(customer_id)"""

SQL = {
    "new_paying": (NEW_PAYING, ["customers", "subscriptions", "payments"]),
    "revenue": (REVENUE + """ SELECT ROUND(COALESCE(SUM(amount), 0), 2) AS value
        FROM allocation WHERE month BETWEEN :start AND :end AND (:plan IS NULL OR plan = :plan)""",
        ["subscriptions", "payments"]),
    "revenue_by_plan": (REVENUE + """ SELECT plan, ROUND(SUM(amount), 2) AS revenue_rub
        FROM allocation WHERE month BETWEEN :start AND :end GROUP BY plan ORDER BY plan""",
        ["subscriptions", "payments"]),
    "revenue_by_region": (REVENUE + """ SELECT region, ROUND(SUM(amount), 2) AS revenue_rub
        FROM allocation WHERE month BETWEEN :start AND :end GROUP BY region ORDER BY region""",
        ["customers", "subscriptions", "payments"]),
    "active": ("""SELECT COUNT(DISTINCT customer_id) AS value FROM subscriptions
        WHERE started_at <= :end AND (ended_at IS NULL OR ended_at >= :end)""",
        ["customers", "subscriptions"]),
    "licenses": ("""SELECT plan, ROUND(AVG(licensed_users), 2) AS average_licensed_users
        FROM subscriptions GROUP BY plan ORDER BY plan""", ["subscriptions"]),
    "refund_amount": ("""SELECT ROUND(COALESCE(SUM(amount_rub), 0), 2) AS value FROM payments
        WHERE operation_type = 'refund' AND status = 'succeeded' AND paid_at BETWEEN :start AND :end""", ["payments"]),
    "tickets": ("""SELECT COUNT(*) AS value FROM support_tickets
        WHERE substr(created_at, 1, 10) BETWEEN :start AND :end""", ["support_tickets"]),
    "tickets_by_category": ("""SELECT category, COUNT(*) AS ticket_count FROM support_tickets
        WHERE substr(created_at, 1, 10) BETWEEN :start AND :end GROUP BY category ORDER BY category""", ["support_tickets"]),
    "resolution_hours": ("""SELECT ROUND(AVG(24 * (julianday(closed_at) - julianday(created_at))), 2)
        AS average_resolution_hours, COUNT(*) AS closed_tickets FROM support_tickets
        WHERE status = 'closed' AND substr(created_at, 1, 10) BETWEEN :start AND :end""", ["support_tickets"]),
    "annual_share": ("""SELECT ROUND(100.0 * SUM(billing_cycle = 'annual') / COUNT(*), 2) AS value,
        SUM(billing_cycle = 'annual') AS numerator, COUNT(*) AS denominator FROM subscriptions
        WHERE started_at <= :end AND (ended_at IS NULL OR ended_at >= :end)""", ["subscriptions"]),
    "churn": (CHURN, ["customers", "subscriptions"]),
    "retention": ("""WITH active AS (
        SELECT DISTINCT customer_id FROM subscriptions
        WHERE started_at <= :start AND (ended_at IS NULL OR ended_at >= :start)
    ), retained AS (
        SELECT DISTINCT customer_id FROM subscriptions
        WHERE started_at <= :end AND (ended_at IS NULL OR ended_at >= :end)
    )
    SELECT segment, COUNT(active.customer_id) AS active_at_start,
        COUNT(retained.customer_id) AS retained_at_end,
        COALESCE(ROUND(100.0 * COUNT(retained.customer_id) / COUNT(active.customer_id), 2), 0.0) AS rate_percent
    FROM customers JOIN active USING(customer_id) LEFT JOIN retained USING(customer_id)
    GROUP BY segment ORDER BY segment""", ["customers", "subscriptions"]),
    "segment_tickets": ("""WITH tickets AS (
        SELECT customer_id, COUNT(*) AS n FROM support_tickets
        WHERE substr(created_at, 1, 10) BETWEEN :start AND :end GROUP BY customer_id
    ), active AS (
        SELECT DISTINCT customer_id FROM subscriptions
        WHERE started_at <= :start AND (ended_at IS NULL OR ended_at >= :start)
    )
    SELECT segment, COALESCE(SUM(tickets.n), 0) AS tickets, COUNT(active.customer_id) AS active_customers
    FROM customers LEFT JOIN tickets USING(customer_id) LEFT JOIN active USING(customer_id)
    GROUP BY segment ORDER BY segment""", ["customers", "subscriptions", "support_tickets"]),
}


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def tokenize(text: str) -> Counter:
    return Counter(token[:5] for token in TOKEN.findall(text.lower().replace("ё", "е")) if len(token) >= 2)


def parse_document(path: Path) -> dict:
    _, header, body = path.read_text(encoding="utf-8").split("---", 2)
    metadata = {key.strip(): value.strip().strip('"') for key, value in
                (line.split(":", 1) for line in header.splitlines() if line.strip())}
    sections = dict(re.findall(r"^## ([^\n]+)\n+(.+?)(?=^## |\Z)", body, flags=re.MULTILINE | re.DOTALL))
    sections = {section: text.strip() for section, text in sections.items()}
    return {
        "metadata": metadata, "sections": sections, "path": path.relative_to(ROOT).as_posix(),
        "title_index": tokenize(metadata["title"]), "section_index": tokenize(" ".join(sections)),
        "body_index": tokenize(body),
    }


class Corpus:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        for table, schema in TABLES.items():
            self.db.execute(f"CREATE TABLE {table} ({schema})")
            columns = [column.split()[0] for column in schema.split(",")]
            with (CORPUS / "tables" / f"{table}.csv").open(encoding="utf-8", newline="") as file:
                rows = csv.DictReader(file)
                self.db.executemany(
                    f"INSERT INTO {table} VALUES ({','.join('?' for _ in columns)})",
                    ([row[key] or None for key in columns] for row in rows),
                )
        self.db.commit()
        self.db.execute("PRAGMA query_only = ON")
        self.documents = [
            parse_document(path) for path in sorted((CORPUS / "documents").rglob("*.md"))
        ]

    def query(self, template: str, windows: list[dict]) -> list[list[dict]]:
        if template not in SQL or not 1 <= len(windows) <= 8:
            raise ToolError("invalid_input", "Неизвестный шаблон или число окон вне лимита.")
        deadline = perf_counter() + LIMITS["sql"][1]
        self.db.set_progress_handler(lambda: int(perf_counter() > deadline), 1000)
        try:
            results = []
            for window in windows:
                start, end = date.fromisoformat(window["start"]), date.fromisoformat(window["end"])
                if end < start:
                    raise ToolError("invalid_input", "Конец периода раньше начала.")
                if not DATA_START <= start <= end <= DATA_END:
                    raise ToolError("outside_snapshot", "Запрошенный период выходит за границы снимка данных.")
                params = {"plan": None, "segment": None, **window}
                rows = self.db.execute(SQL[template][0], params).fetchmany(101)
                if len(rows) > 100:
                    raise ToolError("row_limit", "Результат превысил 100 строк.")
                results.append([dict(row) for row in rows])
            return results
        except sqlite3.OperationalError as error:
            if "interrupted" in str(error):
                raise ToolError("timeout", "SQL превысил лимит времени.") from error
            raise ToolError("tool_error", "Ошибка чтения снимка данных.") from error
        except (KeyError, TypeError, ValueError) as error:
            raise ToolError("invalid_input", "Нужны корректные даты начала и конца.") from error
        finally:
            self.db.set_progress_handler(None, 0)

    def search(self, query: str, *, kind: str | tuple[str, ...], as_of: str, plan: str | None = None,
               current: bool = False, version: str | None = None,
               start: str | None = None, end: str | None = None, recent: bool = False) -> list[dict]:
        deadline = perf_counter() + LIMITS["retrieve_documents"][1]
        words = tokenize(query)
        kinds = (kind,) if isinstance(kind, str) else kind
        ranked = []
        for doc in self.documents:
            if perf_counter() > deadline:
                raise ToolError("timeout", "Поиск превысил лимит времени.")
            m = doc["metadata"]
            if m["document_type"] not in kinds or (plan and m.get("plan") != plan):
                continue
            effective = m.get("valid_from") or m["published_at"]
            if effective > as_of or (version and m["version"] != version):
                continue
            if current and m.get("valid_to") and m["valid_to"] < as_of:
                continue
            if (start and effective < start) or (end and effective > end):
                continue
            score = sum(
                5 * bool(doc["title_index"][word]) + 3 * bool(doc["section_index"][word])
                + min(doc["body_index"][word], 2) for word in words
            )
            ranked.append((effective if recent else "", score, effective, m["doc_id"], doc))
        ranked.sort(key=lambda item: item[:4], reverse=True)
        return [item[-1] for item in ranked[:5]]


def calculate(operation: str, rows: list[dict]) -> list[dict]:
    if operation != "tickets_per_100_active_customers" or not 1 <= len(rows) <= 100:
        raise ToolError("invalid_input", "Неизвестная операция или слишком много строк.")
    deadline = perf_counter() + LIMITS["calculate"][1]
    try:
        result = []
        for row in rows:
            if perf_counter() > deadline:
                raise ToolError("timeout", "Расчёт превысил лимит времени.")
            tickets, customers = row["tickets"], row["active_customers"]
            if type(tickets) is not int or type(customers) is not int or min(tickets, customers) < 0:
                raise ValueError("nonnegative integer required")
            result.append({"segment": row["segment"],
                           "tickets_per_100": round(100 * tickets / customers, 2) if customers else 0.0})
        return result
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError("invalid_input", "Расчёт требует сегмента и двух целых неотрицательных количеств.") from error
