from __future__ import annotations

import calendar
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from time import perf_counter

from kt3.tools import DATA_END, DATA_START, LIMITS, SQL, Corpus, ToolError, calculate

MONTHS = ("январ", "феврал", "март", "апрел", "ма", "июн", "июл",
          "август", "сентябр", "октябр", "ноябр", "декабр")
MONTH_PATTERN = re.compile(r"\b(" + "|".join(MONTHS) + r")[а-я]*\b")
PLANS = ("basic", "pro", "enterprise")
MAX_INPUT = 2_000
MAX_HISTORY = 6
ACCESS_DENIED = re.compile(r"нет доступа|чуж(?:ого|ие|ой)|удали|сотри|измени записи", re.IGNORECASE)
SESSION_TTL = timedelta(minutes=90)
MAX_STEPS = 6
RUN_TIMEOUT = 2.0


def month_window(year: int, month: int) -> dict:
    return {
        "start": date(year, month, 1).isoformat(),
        "end": date(year, month, calendar.monthrange(year, month)[1]).isoformat(),
    }


def date_window(start: date, end: date) -> dict:
    return {"start": start.isoformat(), "end": end.isoformat()}


def parse_period(question: str, as_of: date) -> tuple[str, dict] | None:
    year_match = re.search(r"\b20\d\d\b", question)
    year = int(year_match[0]) if year_match else as_of.year
    exact = re.search(r"\b(20\d\d-\d\d-\d\d)\b", question)
    if exact:
        day = date.fromisoformat(exact[1])
        return day.isoformat(), date_window(day, day)
    month_matches = MONTH_PATTERN.findall(question)
    if month_matches:
        month = MONTHS.index(month_matches[-1]) + 1
        window = month_window(year, month)
        day_match = re.search(r"(?<!\d)([1-9]|[12]\d|3[01])\s+" + month_matches[-1], question)
        if day_match:
            day = date(year, month, int(day_match[1]))
            return day.isoformat(), date_window(day, day)
        return f"{year}-{month:02}", window
    if "последний полный месяц" in question:
        previous = as_of.replace(day=1) - timedelta(days=1)
        return previous.strftime("%Y-%m"), month_window(previous.year, previous.month)
    quarter = re.search(r"\b(перв\w*|втор\w*|трет\w*|четверт\w*|[1-4])\s+квартал", question)
    if quarter:
        part = quarter[1]
        number = int(part) if part.isdigit() else next(
            i for i, stem in enumerate(("перв", "втор", "трет", "четверт"), 1) if part.startswith(stem)
        )
        end_month = 3 * number
        return f"{year}-Q{number}", {
            "start": date(year, end_month - 2, 1).isoformat(),
            "end": month_window(year, end_month)["end"],
        }
    return None


@dataclass
class Session:
    updated_at: datetime | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))

    def turn(self, question: str, at: datetime) -> tuple[str, bool]:
        if self.updated_at is not None and (at < self.updated_at or at - self.updated_at >= SESSION_TTL):
            self.history.clear()
        self.updated_at = at
        lowered = question.lower().replace("ё", "е").strip()
        corrected = bool(re.match(r"^(?:а|нет,?|уточнение:)\s+за\s+", lowered))
        if corrected and self.history:
            previous = self.history[-1]
            previous_without_period = re.split(r"\s+за\s+|\s+в\s+(?=[а-я]+\s+квартал)", previous, maxsplit=1)[0]
            lowered = f"{previous_without_period} {lowered.split(' за ', 1)[1]}"
        self.history.append(lowered)
        return lowered, corrected


class Execution:
    def __init__(self, corpus: Corpus, limit: int, fault: dict[str, int]):
        self.corpus = corpus
        self.limit = limit
        self.fault = fault
        self.deadline = perf_counter() + RUN_TIMEOUT
        self.steps = []
        self.facts = []
        self.counts = Counter()

    def call(self, kind: str, inputs: dict, run, serialize):
        if len(self.steps) >= self.limit or perf_counter() >= self.deadline:
            raise ToolError("budget_exhausted", "Исчерпан бюджет инструментов или времени.")
        if self.counts[kind] >= LIMITS[kind][0]:
            raise ToolError("budget_exhausted", "Исчерпан лимит вызовов инструмента.")
        self.counts[kind] += 1
        start = perf_counter()
        event = {"step_index": len(self.steps), "step": kind, "input": inputs}
        try:
            if self.counts[kind] <= self.fault.get(kind, 0):
                raise ToolError("timeout", "Воспроизведён таймаут инструмента в сценарии.")
            value = run()
            event["result"] = serialize(value)
            return value
        except ToolError as error:
            event["error"] = {"code": error.code, "message": str(error)}
            raise
        finally:
            event["duration_ms"] = round((perf_counter() - start) * 1000, 3)
            self.steps.append(event)

    def sql(self, template: str, period: str, windows: list[dict], shape=lambda rows: rows[0]):
        inputs = {"template": template, "period": period, "windows": windows}
        sources = SQL[template][1]
        result = self.call("sql", inputs, lambda: shape(self.corpus.query(template, windows)),
                           lambda rows: {"template": template, "period": period, "rows": rows, "sources": sources})
        return result

    def search(self, question: str, **filters):
        inputs = {"query": question, **filters}
        return self.call("retrieve_documents", inputs, lambda: self.corpus.search(question, **filters),
                         lambda docs: {"ranked_doc_ids": [d["metadata"]["doc_id"] for d in docs]})

    def calc(self, operation: str, rows: list[dict]):
        return self.call("calculate", {"operation": operation, "inputs": rows},
                         lambda: calculate(operation, rows),
                         lambda outputs: {"operation": operation, "inputs": rows, "outputs": outputs})

    def fact(self, step: int, kind: str, **details) -> str:
        fact_id = f"f{len(self.facts) + 1}"
        self.facts.append({"fact_id": fact_id, "step_index": step, "kind": kind, **details})
        return fact_id


def response(status: str, text: str, *, reason: str | None = None, answer=None) -> dict:
    return {"status": status, "text": text, "reason": reason, "answer": answer}


def cite(state: Execution, doc: dict, section: str | None = None, *, step: int | None = None) -> str:
    sections = doc["sections"]
    section = section if section in sections else next(iter(sections))
    meta = doc["metadata"]
    quote = sections[section][:500].strip()
    state.fact(
        len(state.steps) - 1 if step is None else step, "document",
        doc_id=meta["doc_id"], version=meta["version"], section=section, quote=quote, path=doc["path"],
    )
    return f"[{meta['doc_id']}, {section}], версия {meta['version']}. «{quote}»"


UNITS = {
    "new_paying": "компании", "revenue": "₽", "revenue_by_plan": "₽", "revenue_by_region": "₽",
    "active": "компании", "licenses": "лицензий", "refund_amount": "₽",
    "tickets": "обращений", "tickets_by_category": "обращений",
    "resolution_hours": "часов", "annual_share": "%", "churn": "%", "retention": "%",
}


def record_rows(state: Execution, template: str, period: str, rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        state.fact(len(state.steps) - 1, "sql", metric=template, period=row.get("period", period),
                   unit=UNITS.get(template), table_sources=SQL[template][1], row_index=index, values=row)


def scalar_or_group(state: Execution, template: str, period: str, window: dict, plan: str | None = None) -> dict:
    rows = state.sql(template, period, [{**window, "plan": plan}])
    record_rows(state, template, period, rows)
    unit = UNITS[template]
    sources = ", ".join(SQL[template][1])
    if template in {"revenue_by_plan", "revenue_by_region"}:
        value = max(rows, key=lambda row: row["revenue_rub"], default=None)
    else:
        value = rows[0].get("value", rows[0]) if len(rows) == 1 else rows
    return response("answer", f"{period}. {value} {unit}. Источники: {sources}.", answer=value)


def compare(state: Execution, template: str, labels: tuple[str, str], windows: list[dict],
            amount_field: str, *, trace_period: str | None = None) -> tuple[list[dict], float | None]:
    period = trace_period or "/".join(labels)

    def shape(groups):
        before, after = [
            {"period": label, **(rows[0] if template == "churn" else {amount_field: rows[0]["value"]})}
            for label, rows in zip(labels, groups, strict=True)
        ]
        a, b = before[amount_field], after[amount_field]
        if template in {"new_paying", "revenue"}:
            before["change_percent"] = None
            after["change_percent"] = round((b - a) * 100 / a, 2) if a else None
        return [before, after]

    rows = state.sql(template, period, windows, shape)
    record_rows(state, template, period, rows)
    before, after = (row[amount_field] for row in rows)
    delta = round(after - before, 2) if template == "churn" else (
        round((after - before) * 100 / before, 2) if before else None
    )
    return rows, delta


def search_with_retry(state: Execution, question: str, **filters):
    try:
        return state.search(question, **filters)
    except ToolError as error:
        if error.code != "timeout":
            raise
    return state.search(question, **filters)


def ticket_release(state: Execution, question: str, as_of: date) -> dict:
    version = re.search(r"\b\d+\.\d+(?:\.\d+)?\b", question)
    if version is None:
        return response("clarify", "Уточните версию релиза и окно сравнения.", reason="missing_release")
    docs = search_with_retry(state, question, kind="release_note", as_of=as_of.isoformat(), version=version[0])
    if not docs:
        return response("refuse", "Описание указанного релиза отсутствует.", reason="missing_document")
    reference = cite(state, docs[0])
    event_day = date.fromisoformat(docs[0]["metadata"]["published_at"])
    requested_days = re.search(r"(?<!\d)(\d{1,3})\s+дн", question)
    days = int(requested_days[1]) if requested_days else 29
    if not 1 <= days <= 180:
        return response("clarify", "Укажите окно сравнения от 1 до 180 дней.", reason="invalid_window")
    windows = [date_window(event_day - timedelta(days=days), event_day - timedelta(days=1)),
               date_window(event_day, event_day + timedelta(days=days - 1))]
    labels = tuple(f"{w['start']}/{w['end']}" for w in windows)
    period = (f"{days} days before/after {event_day}" if requested_days else
              f"{days} days before/{days} days after {event_day}")
    try:
        rows, delta = compare(state, "tickets", labels, windows, "ticket_count", trace_period=period)
    except ToolError as error:
        return response("partial_answer", f"Релиз опубликован {event_day}. {reference} Сравнение обращений недоступно.",
                        reason=error.code, answer={"release_date": event_day.isoformat()})
    before, after = (row["ticket_count"] for row in rows)
    return response(
        "answer", f"{labels[0]} — {before} обращений, {labels[1]} — {after}. "
        f"Источник количества — support_tickets. {reference} Совпадение по времени не доказывает причинность.",
        answer={"before": before, "after": after, "change_percent": delta, "release_date": event_day.isoformat()},
    )


def segment_release(state: Execution, question: str, as_of: date, parsed: tuple[str, dict]) -> dict:
    period, window = parsed
    rows = state.sql("segment_tickets", period, [window])
    sql_index = len(state.steps) - 1
    record_rows(state, "segment_tickets", period, rows)
    try:
        rates = state.calc("tickets_per_100_active_customers", rows)
    except ToolError as error:
        return response("partial_answer", f"{period}. Подтверждённые количества по сегментам: {rows}. "
                        "Частоту рассчитать не удалось.", reason=error.code, answer={"rows": rows})
    for index, rate in enumerate(rates, start=1):
        state.fact(len(state.steps) - 1, "calculation", metric="tickets_per_100_active_customers",
                   period=period, values=rate, unit="на 100 активных клиентов", input_fact_ids=[f"f{index}"],
                   input_step=sql_index)
    top = max(rates, key=lambda row: row["tickets_per_100"])
    summary = f"{period}. {top['segment']}: {top['tickets_per_100']} обращения на 100 активных клиентов. " \
              "Источники: support_tickets, subscriptions, customers."
    try:
        docs = search_with_retry(state, question, kind="release_note", as_of=as_of.isoformat(), **window)
    except ToolError as error:
        return response("partial_answer", summary + " Сведения о релизах недоступны.",
                        reason=error.code, answer={"top": top})
    citations = " ".join(cite(state, doc) for doc in docs)
    return response("answer", summary + " " + (citations or "В этом периоде релизов в корпусе нет.") +
                    " Совпадение по времени не доказывает причинность.",
                    answer={"top": top, "doc_ids": [doc["metadata"]["doc_id"] for doc in docs]})


def completed_quarter(as_of: date) -> tuple[str, dict]:
    first_month = ((as_of.month - 1) // 3) * 3 + 1
    previous_end = date(as_of.year, first_month, 1) - timedelta(days=1)
    previous_start = date(previous_end.year, previous_end.month - 2, 1)
    return f"{previous_end.year}-Q{previous_end.month // 3}", date_window(previous_start, previous_end)


def churn_after_price(state: Execution, q: str, as_of: date, plan: str) -> dict:
    docs = search_with_retry(state, q, kind="tariff", plan=plan, as_of=as_of.isoformat(), recent=True)
    if not docs:
        return response("refuse", "Не найдено действующее изменение цены.", reason="missing_document")
    doc = docs[0]
    citation = cite(state, doc, "Стоимость")
    effective = date.fromisoformat(doc["metadata"]["valid_from"])
    if effective.day != 1 or effective.month not in {1, 4, 7, 10}:
        return response("clarify", f"Тариф вступил в силу {effective}. {citation} Уточните окно сравнения.",
                        reason="missing_inputs")
    before_label, before_window = completed_quarter(effective)
    after_quarter = (effective.month - 1) // 3 + 1
    after_window = {"start": effective.isoformat(), "end": month_window(effective.year, after_quarter * 3)["end"]}
    if after_window["end"] >= as_of.isoformat():
        return response("partial_answer", f"Изменение цены от {effective}. {citation} Квартал после изменения ещё не завершён.",
                        reason="missing_data", answer={"change_date": effective.isoformat()})
    labels = (before_label, f"{effective.year}-Q{after_quarter}")
    try:
        rows, delta = compare(state, "churn", labels, [
            {**before_window, "plan": plan}, {**after_window, "plan": plan},
        ], "rate_percent")
    except ToolError as error:
        return response("partial_answer", f"Изменение цены от {effective}. {citation} Расчёт оттока недоступен.",
                        reason=error.code, answer={"change_date": effective.isoformat()})
    return response("answer", f"{labels[0]}: {rows[0]['rate_percent']}%, {labels[1]}: {rows[1]['rate_percent']}%. "
                    f"Изменение: {delta} п.п. Источники: customers, subscriptions. {citation} "
                    "Совпадение по времени не доказывает причинность.",
                    answer={"change_date": effective.isoformat(), "change_percentage_points": delta})


def retention_and_events(state: Execution, q: str, as_of: date) -> dict:
    after_label, after = completed_quarter(as_of)
    before_label, before = completed_quarter(date.fromisoformat(after["start"]))
    period = f"{before_label}/{after_label}"
    rows = state.sql("retention", period, [before, after], lambda groups: [
        {**row, "period": label} for label, group in zip((before_label, after_label), groups, strict=True)
        for row in group
    ])
    record_rows(state, "retention", period, rows)
    baselines = {row["segment"]: row["rate_percent"] for row in rows if row["period"] == before_label}
    drops = sorted((
        {"segment": row["segment"], "previous_percent": baselines[row["segment"]],
         "last_percent": row["rate_percent"], "delta_pp": round(row["rate_percent"] - baselines[row["segment"]], 2)}
        for row in rows if row["period"] == after_label and row["segment"] in baselines
    ), key=lambda row: row["delta_pp"])[:3]
    text = f"Удержание, {period}: {drops}. Источники: customers, subscriptions."
    product_scope = "enterprise" if any(row["segment"] == "enterprise" for row in drops) else None
    try:
        found = search_with_retry(
            state, q, kind=("release_note", "tariff", "policy"), as_of=as_of.isoformat(), plan=product_scope,
            start=before["start"], end=after["end"],
        )
    except ToolError as error:
        return response("partial_answer", text + " Поиск контекста недоступен.",
                        reason=error.code, answer={"top": drops})
    sections = {"tariff": "Стоимость", "policy": "Область действия", "release_note": "Основные изменения"}
    citations = [cite(state, doc, sections[doc["metadata"]["document_type"]]) for doc in found]
    candidates = [doc["metadata"]["doc_id"] for doc in found]
    return response("answer", text + " " + " ".join(citations) +
                    " Тариф и клиентский сегмент — разные признаки. События предыдущего квартала даны для контекста. "
                    "Связь по времени не доказывает причинность.",
                    answer={"top": drops, "doc_ids": candidates})


def documents(state: Execution, question: str, as_of: date, plan: str | None) -> dict | None:
    filters = {"as_of": as_of.isoformat()}
    limit = 1
    if "sla" in question and "отток" not in question:
        kind, requested_sections = "policy", ("Область действия", "Приоритеты инцидентов", "Расчёт доступности", "Каналы обращения")
        # The policy index contains both SLA and refund families.
        limit = 2
    elif "возврат" in question and not re.search(r"сумм|число|изменил", question):
        kind, requested_sections = "policy", ("Срок подачи запроса", "Размер возврата", "Исключения", "Порядок обращения")
        filters.update(current=True)
        if re.search(r"срок|в течение", question):
            requested_sections = requested_sections[:1]
    elif plan and ("тариф" in question or "версиями" in question):
        if not re.search(r"услов|ограничен|стоимост|верси", question):
            return None
        kind = "tariff"
        filters.update(plan=plan, recent=True)
        limit = 2 if "между версиями" in question else 1
        requested_sections = ("Ограничения",) if "ограничен" in question else (
            ("Стоимость", "Ограничения") if limit == 2 else ("Стоимость",)
        )
    elif "релиз" in question and ("последних" in question or "функци" in question):
        kind, requested_sections = "release_note", ("Основные изменения",)
        filters.update(recent=True)
        limit = 3
    else:
        return None

    docs = state.search(question, kind=kind, **filters)
    if kind == "policy":
        family = "policy_sla_" if "sla" in question else "policy_refund_"
        docs = [doc for doc in docs if doc["metadata"]["doc_id"].startswith(family)]
    docs = docs[:limit]
    if not docs:
        return response("refuse", "В корпусе отсутствует подходящий документ.", reason="missing_document")
    citations = " ".join(cite(state, doc, section) for doc in docs for section in requested_sections
                         if section in doc["sections"])
    if not citations:
        citations = " ".join(cite(state, doc) for doc in docs)
    return response("answer", citations, answer={"doc_ids": [doc["metadata"]["doc_id"] for doc in docs]})


def route(state: Execution, mode: str, q: str, as_of: date, corrected: bool) -> dict:
    if re.search(r"рекомендовать|удовлетворенн|дол[яю].*рынк|конкурент", q):
        return response("refuse", "Подключённые источники не содержат нужных данных.", reason="missing_data")
    parsed = parse_period(q, as_of)
    plan = next((plan for plan in PLANS if re.search(rf"\b{plan}\b", q)), None)
    if corrected and re.match(r"^(?:а|нет,?|уточнение:)\s+за\s+", q):
        return response("clarify", "Укажите показатель и объект для этого периода.", reason="no_session_context")
    if "обращен" in q and "релиз" in q:
        if mode == "baseline":
            return response("refuse", "Для этой комбинации источников требуется аналитик.", reason="unsupported")
        if "на 100" in q and parsed:
            return segment_release(state, q, as_of, parsed)
        return ticket_release(state, q, as_of)
    if mode == "system":
        if "удержани" in q and "три сегмент" in q and "последний квартал" in q:
            return retention_and_events(state, q, as_of)
        if plan and "отток" in q and "стоимост" in q and "тариф" in q:
            return churn_after_price(state, q, as_of, plan)
    if re.search(r"удержани|отток|сильнее всего|есть ли месяцы|после .*изменени", q):
        return response("refuse", "Для этой последовательности расчётов требуется аналитик.", reason="unsupported")
    if "новых платящ" in q or "выручк" in q:
        template = "new_paying" if "новых платящ" in q else "revenue"
        if "какой тариф" in q:
            template = "revenue_by_plan"
        if "какой регион" in q:
            template = "revenue_by_region"
        months = MONTH_PATTERN.findall(q)
        if len(months) == 2 and re.search(r"процент|изменил|вырос|снизил", q) and template in {"new_paying", "revenue"}:
            years = [int(year) for year in re.findall(r"\b20\d\d\b", q)]
            first_year, last_year = (years[0], years[-1]) if years else (as_of.year, as_of.year)
            dated_months = list(zip((first_year, last_year), (MONTHS.index(month) + 1 for month in months), strict=True))
            labels = tuple(f"{year}-{month:02}" for year, month in dated_months)
            windows = [{**month_window(year, month), "plan": plan} for year, month in dated_months]
            field = "revenue_rub" if template == "revenue" else "new_paying_customers"
            tariff_comparison = template == "revenue" and plan is not None and "изменение тарифа" in q
            citation = ""
            if tariff_comparison and mode == "system":
                docs = search_with_retry(
                    state, q, kind="tariff", plan=plan, as_of=as_of.isoformat(),
                    start=windows[1]["start"], end=windows[1]["end"],
                )
                if docs:
                    citation = cite(state, docs[0], "Стоимость")
            rows, delta = compare(state, template, labels, windows, field)
            measured = {"before": rows[0][field], "after": rows[1][field], "change_percent": delta}
            text = f"{rows}. Изменение: {delta}%. Источники: {', '.join(SQL[template][1])}."
            if tariff_comparison and not citation:
                return response("partial_answer", text + " Изменение тарифа не подтверждено.",
                                reason="missing_document", answer=measured)
            if citation:
                text += " " + citation + " Совпадение по времени не доказывает причинность."
            return response("answer", text, answer=measured)
        if parsed:
            return scalar_or_group(state, template, *parsed, plan)
    if "лиценз" in q and "средн" in q:
        return scalar_or_group(state, "licenses", "all_corpus", date_window(DATA_START, DATA_END))
    if parsed:
        for needle, template in (
            ("доля годовых", "annual_share"), ("активную подписку", "active"), ("сумма успешных возврат", "refund_amount"),
            ("обращений каждой категории", "tickets_by_category"), ("время закрытия обращений", "resolution_hours"),
        ):
            if needle in q:
                return scalar_or_group(state, template, *parsed)
    doc_answer = documents(state, q, as_of, plan)
    if doc_answer:
        return doc_answer
    if not parsed and re.search(r"недавно|последнее время|после релиза|выручк|заработал", q):
        return response("clarify", "Уточните показатель, объект и период сравнения.", reason="missing_inputs")
    return response("refuse", "Для этого вопроса потребуется проверка аналитиком.", reason="unsupported")


def run_turn(corpus: Corpus, mode: str, session: Session, question: str, as_of: datetime, *,
             step_limit: int = MAX_STEPS, fault: dict | None = None, restricted: bool = False) -> dict:
    state = Execution(corpus, step_limit, fault or {})
    start = perf_counter()
    state_read = len(session.history)
    corrected = False
    if not 0 < len(question.strip()) <= MAX_INPUT:
        result = response("clarify", "Введите вопрос длиной от 1 до 2000 символов.", reason="invalid_question")
    elif restricted or ACCESS_DENIED.search(question):
        result = response("refuse", "Запрос требует доступа или действий вне разрешённого чтения.",
                          reason="access_denied")
    else:
        normalized, corrected = session.turn(question, as_of)
        try:
            result = route(state, mode, normalized, as_of.date(), corrected)
        except ToolError as error:
            status = "refuse" if error.code == "outside_snapshot" else "report_tool_error"
            result = response(status, str(error), reason=error.code)
        except ValueError:
            result = response("clarify", "Укажите корректную дату или период.", reason="invalid_period")
    return {
        **result, "steps": state.steps, "facts": state.facts,
        "duration_ms": round((perf_counter() - start) * 1000, 3),
        "state": {"before": state_read, "after": len(session.history), "user_corrected": corrected},
        "step_budget": step_limit, "model_calls": 0,
    }
