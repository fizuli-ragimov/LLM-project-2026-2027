from __future__ import annotations

import calendar
import csv
import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path


SEED = 20261001
DATA_START = date(2024, 10, 1)
DATA_END = date(2026, 9, 30)

CUSTOMERS_COUNT = 3_000
SUBSCRIPTIONS_COUNT = 4_500
PAYMENTS_COUNT = 24_000
TICKETS_COUNT = 5_000

ROOT = Path(__file__).resolve().parent
TABLES_DIR = ROOT / "tables"
MOSCOW_TZ = timezone(timedelta(hours=3))

SEGMENT_SPECS = (
    ("micro_business", 1_050, 1, 10),
    ("small_business", 1_050, 11, 50),
    ("mid_market", 600, 51, 250),
    ("enterprise", 300, 251, 2_000),
)

REGIONS = (
    "moscow",
    "saint_petersburg",
    "central",
    "volga",
    "south",
    "ural",
    "siberia",
    "far_east",
)
REGION_WEIGHTS = (24, 12, 16, 14, 10, 10, 9, 5)

ACQUISITION_CHANNELS = ("organic", "paid_ads", "partner", "referral", "outbound")
ACQUISITION_WEIGHTS = (30, 20, 18, 20, 12)

COMPANY_FIRST_WORDS = (
    "Вектор",
    "Орбита",
    "Север",
    "Маяк",
    "Гранит",
    "Пульс",
    "Линия",
    "Спектр",
    "Альянс",
    "Горизонт",
)
COMPANY_SECOND_WORDS = (
    "Проект",
    "Системы",
    "Технологии",
    "Решения",
    "Групп",
    "Софт",
    "Команда",
    "Диджитал",
)

PLAN_WEIGHTS = {
    "micro_business": (("basic", "pro", "enterprise"), (70, 27, 3)),
    "small_business": (("basic", "pro", "enterprise"), (35, 55, 10)),
    "mid_market": (("basic", "pro", "enterprise"), (10, 65, 25)),
    "enterprise": (("basic", "pro", "enterprise"), (2, 28, 70)),
}

ANNUAL_PROBABILITY = {
    "basic": 0.22,
    "pro": 0.38,
    "enterprise": 0.65,
}

PLAN_LICENSE_LIMIT = {
    "basic": 10,
    "pro": 100,
    "enterprise": 2_000,
}

CANCELLATION_REASONS = (
    "too_expensive",
    "missing_features",
    "technical_issues",
    "switched_to_competitor",
    "no_longer_needed",
)


def weighted_choice(rng: random.Random, values, weights):
    return rng.choices(values, weights=weights, k=1)[0]


def random_date(rng: random.Random, start: date, end: date) -> date:
    if end < start:
        return start
    return start + timedelta(days=rng.randint(0, (end - start).days))


def add_months(value: date, months: int) -> date:
    absolute_month = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(absolute_month, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def segment_for_employee_count(employee_count: int) -> str:
    if employee_count <= 10:
        return "micro_business"
    if employee_count <= 50:
        return "small_business"
    if employee_count <= 250:
        return "mid_market"
    return "enterprise"


def generate_customers(rng: random.Random) -> list[dict]:
    customers: list[dict] = []
    customer_id = 1

    for segment, count, minimum, maximum in SEGMENT_SPECS:
        august_event_count = 90 if segment in {"micro_business", "small_business"} else 0
        september_event_count = 30 if segment in {"micro_business", "small_business"} else 0

        for position in range(count):
            employee_count = rng.randint(minimum, maximum)
            if position >= count - september_event_count:
                registered_at = random_date(rng, date(2026, 9, 1), date(2026, 9, 15))
            elif position >= count - september_event_count - august_event_count:
                registered_at = random_date(rng, date(2026, 8, 1), date(2026, 8, 20))
            else:
                registered_at = random_date(rng, DATA_START, date(2026, 7, 31))

            first_word = COMPANY_FIRST_WORDS[(customer_id - 1) % len(COMPANY_FIRST_WORDS)]
            second_word = COMPANY_SECOND_WORDS[
                ((customer_id - 1) // len(COMPANY_FIRST_WORDS)) % len(COMPANY_SECOND_WORDS)
            ]

            customers.append(
                {
                    "customer_id": customer_id,
                    "company_name": f"{first_word} {second_word} {customer_id:04d}",
                    "registered_at": registered_at,
                    "employee_count": employee_count,
                    "segment": segment_for_employee_count(employee_count),
                    "region": weighted_choice(rng, REGIONS, REGION_WEIGHTS),
                    "acquisition_channel": weighted_choice(
                        rng, ACQUISITION_CHANNELS, ACQUISITION_WEIGHTS
                    ),
                }
            )
            customer_id += 1

    assert len(customers) == CUSTOMERS_COUNT
    return customers


def choose_plan(rng: random.Random, segment: str) -> str:
    values, weights = PLAN_WEIGHTS[segment]
    return weighted_choice(rng, values, weights)


def choose_licensed_users(rng: random.Random, employee_count: int, plan: str) -> int:
    maximum = min(employee_count, PLAN_LICENSE_LIMIT[plan])
    minimum = max(1, int(maximum * 0.35))
    return rng.randint(minimum, maximum)


def choose_cancellation_reason(
    rng: random.Random, plan: str, ended_at: date
) -> str:
    weights = [22, 22, 20, 18, 18]
    if (plan == "pro" and ended_at >= date(2026, 4, 1)) or (
        plan == "enterprise" and ended_at >= date(2026, 7, 1)
    ):
        weights = [38, 17, 17, 14, 14]
    return weighted_choice(rng, CANCELLATION_REASONS, weights)


def build_subscription(
    rng: random.Random,
    subscription_id: int,
    customer: dict,
    started_at: date,
    ended_at: date | None,
    forced_plan: str | None = None,
) -> dict:
    plan = forced_plan or choose_plan(rng, customer["segment"])
    status = "cancelled" if ended_at is not None else "active"
    cancellation_reason = (
        choose_cancellation_reason(rng, plan, ended_at) if ended_at is not None else None
    )
    billing_cycle = (
        "annual" if rng.random() < ANNUAL_PROBABILITY[plan] else "monthly"
    )

    return {
        "subscription_id": subscription_id,
        "customer_id": customer["customer_id"],
        "plan": plan,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": status,
        "licensed_users": choose_licensed_users(
            rng, customer["employee_count"], plan
        ),
        "billing_cycle": billing_cycle,
        "cancellation_reason": cancellation_reason,
    }


def generate_subscriptions(
    rng: random.Random, customers: list[dict]
) -> list[dict]:
    customer_by_id = {row["customer_id"]: row for row in customers}
    eligible_for_second = [
        row["customer_id"]
        for row in customers
        if row["registered_at"] <= date(2026, 1, 31)
    ]
    second_subscription_ids = set(rng.sample(eligible_for_second, 1_500))

    enterprise_event_candidates = [
        row["customer_id"]
        for row in customers
        if row["segment"] == "enterprise"
        and row["registered_at"] <= date(2026, 5, 31)
        and row["customer_id"] not in second_subscription_ids
    ]
    enterprise_event_ids = set(
        rng.sample(enterprise_event_candidates, min(90, len(enterprise_event_candidates)))
    )

    subscriptions: list[dict] = []
    subscription_id = 1

    for customer in customers:
        first_start = min(
            customer["registered_at"] + timedelta(days=rng.randint(0, 14)), DATA_END
        )
        customer_id = customer["customer_id"]

        if customer_id in second_subscription_ids:
            minimum_end = first_start + timedelta(days=60)
            maximum_end = min(first_start + timedelta(days=300), date(2026, 7, 31))
            first_end = random_date(rng, minimum_end, maximum_end)
            first_subscription = build_subscription(
                rng, subscription_id, customer, first_start, first_end
            )
            subscriptions.append(first_subscription)
            subscription_id += 1

            second_start = min(
                first_end + timedelta(days=rng.randint(7, 30)), DATA_END
            )
            previous_plan = first_subscription["plan"]
            possible_plans = [plan for plan in ("basic", "pro", "enterprise") if plan != previous_plan]
            forced_plan = rng.choice(possible_plans) if rng.random() < 0.60 else None

            second_end = None
            if second_start <= DATA_END - timedelta(days=45) and rng.random() < 0.18:
                second_end = random_date(
                    rng, second_start + timedelta(days=30), DATA_END
                )

            subscriptions.append(
                build_subscription(
                    rng,
                    subscription_id,
                    customer,
                    second_start,
                    second_end,
                    forced_plan=forced_plan,
                )
            )
            subscription_id += 1
            continue

        ended_at = None
        if customer_id in enterprise_event_ids:
            earliest = max(first_start + timedelta(days=30), date(2026, 7, 1))
            ended_at = random_date(rng, earliest, date(2026, 9, 25))
        else:
            cancellation_probability = {
                "micro_business": 0.32,
                "small_business": 0.28,
                "mid_market": 0.22,
                "enterprise": 0.18,
            }[customer["segment"]]
            if first_start <= DATA_END - timedelta(days=45) and rng.random() < cancellation_probability:
                ended_at = random_date(
                    rng, first_start + timedelta(days=30), DATA_END
                )

        subscriptions.append(
            build_subscription(
                rng, subscription_id, customer_by_id[customer_id], first_start, ended_at
            )
        )
        subscription_id += 1

    assert len(subscriptions) == SUBSCRIPTIONS_COUNT
    return subscriptions


def monthly_price(plan: str, period_start: date) -> Decimal:
    if plan == "basic":
        return Decimal("500")
    if plan == "pro":
        return Decimal("800") if period_start < date(2026, 4, 1) else Decimal("900")
    return Decimal("1100") if period_start < date(2026, 7, 1) else Decimal("1200")


def money_string(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):.2f}"


def generate_payments(rng: random.Random, subscriptions: list[dict]) -> list[dict]:
    payments: list[dict] = []

    for subscription in subscriptions:
        current_period = subscription["started_at"]
        service_end = subscription["ended_at"] or DATA_END
        months = 12 if subscription["billing_cycle"] == "annual" else 1

        while current_period <= service_end and current_period <= DATA_END:
            period_end = add_months(current_period, months) - timedelta(days=1)
            price = monthly_price(subscription["plan"], current_period)
            amount = price * subscription["licensed_users"]
            if months == 12:
                amount *= Decimal("12") * Decimal("0.85")

            payments.append(
                {
                    "payment_id": 0,
                    "subscription_id": subscription["subscription_id"],
                    "paid_at": current_period,
                    "amount_rub": money_string(amount),
                    "operation_type": "charge",
                    "status": "succeeded",
                    "billing_period_start": current_period,
                    "billing_period_end": period_end,
                }
            )
            current_period = add_months(current_period, months)

    if len(payments) > PAYMENTS_COUNT:
        raise RuntimeError(
            f"Scheduled charges ({len(payments)}) exceed target ({PAYMENTS_COUNT})."
        )

    remaining = PAYMENTS_COUNT - len(payments)
    refundable = [row for row in payments if row["paid_at"] <= DATA_END - timedelta(days=7)]
    refund_count = min(300, remaining // 4, len(refundable))

    for source in rng.sample(refundable, refund_count):
        source_amount = Decimal(source["amount_rub"])
        refund_share = weighted_choice(
            rng,
            (Decimal("0.25"), Decimal("0.50"), Decimal("1.00")),
            (20, 45, 35),
        )
        refund_date = min(
            source["paid_at"] + timedelta(days=rng.randint(1, 30)), DATA_END
        )
        payments.append(
            {
                "payment_id": 0,
                "subscription_id": source["subscription_id"],
                "paid_at": refund_date,
                "amount_rub": money_string(source_amount * refund_share),
                "operation_type": "refund",
                "status": "succeeded",
                "billing_period_start": source["billing_period_start"],
                "billing_period_end": source["billing_period_end"],
            }
        )

    while len(payments) < PAYMENTS_COUNT:
        source = rng.choice(refundable)
        failed_at = max(source["billing_period_start"], source["paid_at"] - timedelta(days=1))
        payments.append(
            {
                "payment_id": 0,
                "subscription_id": source["subscription_id"],
                "paid_at": failed_at,
                "amount_rub": source["amount_rub"],
                "operation_type": "charge",
                "status": "failed",
                "billing_period_start": source["billing_period_start"],
                "billing_period_end": source["billing_period_end"],
            }
        )

    payments.sort(
        key=lambda row: (
            row["paid_at"],
            row["subscription_id"],
            row["operation_type"],
            row["status"],
        )
    )
    for payment_id, row in enumerate(payments, start=1):
        row["payment_id"] = payment_id

    assert len(payments) == PAYMENTS_COUNT
    return payments


def generate_support_tickets(
    rng: random.Random, customers: list[dict]
) -> list[dict]:
    tickets: list[dict] = []
    release_date = date(2026, 2, 15)
    spike_end = date(2026, 3, 15)
    spike_customers = [row for row in customers if row["registered_at"] <= release_date]

    for ticket_id in range(1, TICKETS_COUNT + 1):
        in_release_spike = ticket_id <= 700
        if in_release_spike:
            customer = rng.choice(spike_customers)
            created_date = random_date(rng, release_date, spike_end)
            category = weighted_choice(
                rng,
                ("technical", "access", "billing", "plan", "other"),
                (78, 10, 5, 4, 3),
            )
        else:
            customer = rng.choice(customers)
            created_date = random_date(rng, customer["registered_at"], DATA_END)
            category = weighted_choice(
                rng,
                ("technical", "access", "billing", "plan", "other"),
                (38, 20, 18, 14, 10),
            )

        created_at = datetime(
            created_date.year,
            created_date.month,
            created_date.day,
            rng.randint(8, 20),
            rng.randint(0, 59),
            tzinfo=MOSCOW_TZ,
        )
        remains_open = created_date >= date(2026, 9, 1) and rng.random() < 0.35
        if remains_open:
            status = "open"
            closed_at = None
        else:
            status = "closed"
            duration_hours = rng.randint(1, 24 * 14)
            closed_at = min(
                created_at + timedelta(hours=duration_hours),
                datetime(2026, 9, 30, 23, 59, tzinfo=MOSCOW_TZ),
            )

        tickets.append(
            {
                "ticket_id": ticket_id,
                "customer_id": customer["customer_id"],
                "created_at": created_at,
                "closed_at": closed_at,
                "category": category,
                "priority": weighted_choice(
                    rng, ("low", "normal", "high"), (25, 60, 15)
                ),
                "status": status,
                "channel": weighted_choice(
                    rng, ("chat", "email", "web"), (45, 30, 25)
                ),
            }
        )

    assert len(tickets) == TICKETS_COUNT
    return tickets


def serialize_value(value):
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def write_csv(filename: str, rows: list[dict], fieldnames: list[str]) -> None:
    path = TABLES_DIR / filename
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row[key]) for key in fieldnames})


def main() -> None:
    rng = random.Random(SEED)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers(rng)
    subscriptions = generate_subscriptions(rng, customers)
    payments = generate_payments(rng, subscriptions)
    support_tickets = generate_support_tickets(rng, customers)

    write_csv(
        "customers.csv",
        customers,
        [
            "customer_id",
            "company_name",
            "registered_at",
            "employee_count",
            "segment",
            "region",
            "acquisition_channel",
        ],
    )
    write_csv(
        "subscriptions.csv",
        subscriptions,
        [
            "subscription_id",
            "customer_id",
            "plan",
            "started_at",
            "ended_at",
            "status",
            "licensed_users",
            "billing_cycle",
            "cancellation_reason",
        ],
    )
    write_csv(
        "payments.csv",
        payments,
        [
            "payment_id",
            "subscription_id",
            "paid_at",
            "amount_rub",
            "operation_type",
            "status",
            "billing_period_start",
            "billing_period_end",
        ],
    )
    write_csv(
        "support_tickets.csv",
        support_tickets,
        [
            "ticket_id",
            "customer_id",
            "created_at",
            "closed_at",
            "category",
            "priority",
            "status",
            "channel",
        ],
    )

    print(f"customers={len(customers)}")
    print(f"subscriptions={len(subscriptions)}")
    print(f"payments={len(payments)}")
    print(f"support_tickets={len(support_tickets)}")


if __name__ == "__main__":
    main()
