from __future__ import annotations

import calendar
import csv
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path


DATA_START = date(2024, 10, 1)
DATA_END = date(2026, 9, 30)

EXPECTED_COUNTS = {
    "customers": 3_000,
    "subscriptions": 4_500,
    "payments": 24_000,
    "support_tickets": 5_000,
}

EXPECTED_SEGMENTS = {
    "micro_business": 1_050,
    "small_business": 1_050,
    "mid_market": 600,
    "enterprise": 300,
}

ROOT = Path(__file__).resolve().parent
TABLES_DIR = ROOT / "tables"


def read_csv(name: str) -> list[dict[str, str]]:
    path = TABLES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def add_months(value: date, months: int) -> date:
    absolute_month = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(absolute_month, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def segment_for_employee_count(employee_count: int) -> str:
    if employee_count <= 10:
        return "micro_business"
    if employee_count <= 50:
        return "small_business"
    if employee_count <= 250:
        return "mid_market"
    return "enterprise"


def monthly_price(plan: str, period_start: date) -> Decimal:
    if plan == "basic":
        return Decimal("500")
    if plan == "pro":
        return Decimal("800") if period_start < date(2026, 4, 1) else Decimal("900")
    return Decimal("1100") if period_start < date(2026, 7, 1) else Decimal("1200")


class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.passed: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.passed.append(message)
        else:
            self.errors.append(message)


def validate() -> int:
    validator = Validator()

    customers = read_csv("customers.csv")
    subscriptions = read_csv("subscriptions.csv")
    payments = read_csv("payments.csv")
    tickets = read_csv("support_tickets.csv")

    datasets = {
        "customers": customers,
        "subscriptions": subscriptions,
        "payments": payments,
        "support_tickets": tickets,
    }
    for name, rows in datasets.items():
        validator.check(
            len(rows) == EXPECTED_COUNTS[name],
            f"{name}: expected {EXPECTED_COUNTS[name]} rows, got {len(rows)}",
        )

    customer_ids = [int(row["customer_id"]) for row in customers]
    subscription_ids = [int(row["subscription_id"]) for row in subscriptions]
    payment_ids = [int(row["payment_id"]) for row in payments]
    ticket_ids = [int(row["ticket_id"]) for row in tickets]

    validator.check(len(customer_ids) == len(set(customer_ids)), "customer_id values are unique")
    validator.check(
        len(subscription_ids) == len(set(subscription_ids)),
        "subscription_id values are unique",
    )
    validator.check(len(payment_ids) == len(set(payment_ids)), "payment_id values are unique")
    validator.check(len(ticket_ids) == len(set(ticket_ids)), "ticket_id values are unique")

    customer_by_id = {int(row["customer_id"]): row for row in customers}
    subscription_by_id = {
        int(row["subscription_id"]): row for row in subscriptions
    }

    segment_counts = Counter(row["segment"] for row in customers)
    validator.check(
        dict(segment_counts) == EXPECTED_SEGMENTS,
        f"segment distribution matches expected values: {dict(segment_counts)}",
    )
    validator.check(
        len({row["company_name"] for row in customers}) == len(customers),
        "company names are unique",
    )

    for row in customers:
        employee_count = int(row["employee_count"])
        registered_at = parse_date(row["registered_at"])
        validator.check(
            row["segment"] == segment_for_employee_count(employee_count),
            f"customer {row['customer_id']}: segment matches employee_count",
        )
        validator.check(
            DATA_START <= registered_at <= DATA_END,
            f"customer {row['customer_id']}: registered_at is inside corpus period",
        )

    subscriptions_by_customer: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in subscriptions:
        subscription_id = int(row["subscription_id"])
        customer_id = int(row["customer_id"])
        started_at = parse_date(row["started_at"])
        ended_at = parse_date(row["ended_at"]) if row["ended_at"] else None
        customer = customer_by_id.get(customer_id)

        validator.check(
            customer is not None,
            f"subscription {subscription_id}: customer foreign key exists",
        )
        if customer is None:
            continue

        validator.check(
            started_at >= parse_date(customer["registered_at"]),
            f"subscription {subscription_id}: starts after customer registration",
        )
        validator.check(
            DATA_START <= started_at <= DATA_END,
            f"subscription {subscription_id}: started_at is inside corpus period",
        )
        validator.check(
            int(row["licensed_users"]) <= int(customer["employee_count"]),
            f"subscription {subscription_id}: licensed_users does not exceed employee_count",
        )
        validator.check(
            row["plan"] in {"basic", "pro", "enterprise"},
            f"subscription {subscription_id}: plan is valid",
        )
        validator.check(
            row["billing_cycle"] in {"monthly", "annual"},
            f"subscription {subscription_id}: billing_cycle is valid",
        )

        if row["status"] == "active":
            validator.check(
                ended_at is None and row["cancellation_reason"] == "",
                f"subscription {subscription_id}: active status is consistent",
            )
        elif row["status"] == "cancelled":
            validator.check(
                ended_at is not None
                and ended_at >= started_at
                and row["cancellation_reason"] != "",
                f"subscription {subscription_id}: cancelled status is consistent",
            )
        else:
            validator.check(False, f"subscription {subscription_id}: status is valid")

        subscriptions_by_customer[customer_id].append(row)

    validator.check(
        len(subscriptions_by_customer) == len(customers),
        "every customer has at least one subscription",
    )

    for customer_id, rows in subscriptions_by_customer.items():
        ordered = sorted(rows, key=lambda row: parse_date(row["started_at"]))
        for previous, current in zip(ordered, ordered[1:]):
            previous_end = parse_date(previous["ended_at"])
            current_start = parse_date(current["started_at"])
            validator.check(
                previous_end < current_start,
                f"customer {customer_id}: subscription periods do not overlap",
            )

    successful_charge_count_by_subscription = Counter()
    successful_amount_by_period: dict[tuple[int, str, str], Decimal] = defaultdict(Decimal)
    refund_amount_by_period: dict[tuple[int, str, str], Decimal] = defaultdict(Decimal)

    for row in payments:
        payment_id = int(row["payment_id"])
        subscription_id = int(row["subscription_id"])
        paid_at = parse_date(row["paid_at"])
        period_start = parse_date(row["billing_period_start"])
        period_end = parse_date(row["billing_period_end"])
        amount = Decimal(row["amount_rub"])
        subscription = subscription_by_id.get(subscription_id)

        validator.check(
            subscription is not None,
            f"payment {payment_id}: subscription foreign key exists",
        )
        validator.check(
            DATA_START <= paid_at <= DATA_END,
            f"payment {payment_id}: paid_at is inside corpus period",
        )
        validator.check(amount > 0, f"payment {payment_id}: amount is positive")
        validator.check(
            period_end >= period_start,
            f"payment {payment_id}: billing period is valid",
        )
        validator.check(
            row["operation_type"] in {"charge", "refund"},
            f"payment {payment_id}: operation_type is valid",
        )
        validator.check(
            row["status"] in {"succeeded", "failed"},
            f"payment {payment_id}: status is valid",
        )

        if subscription is None:
            continue

        if row["operation_type"] == "charge":
            months = 12 if subscription["billing_cycle"] == "annual" else 1
            expected_end = add_months(period_start, months) - timedelta(days=1)
            expected_amount = (
                monthly_price(subscription["plan"], period_start)
                * int(subscription["licensed_users"])
            )
            if months == 12:
                expected_amount *= Decimal("12") * Decimal("0.85")
            validator.check(
                period_end == expected_end,
                f"payment {payment_id}: billing period matches billing_cycle",
            )
            validator.check(
                amount == expected_amount.quantize(Decimal("0.01")),
                f"payment {payment_id}: amount matches tariff version and licenses",
            )

        period_key = (subscription_id, row["billing_period_start"], row["billing_period_end"])
        if row["status"] == "succeeded" and row["operation_type"] == "charge":
            successful_charge_count_by_subscription[subscription_id] += 1
            successful_amount_by_period[period_key] += amount
        elif row["status"] == "succeeded" and row["operation_type"] == "refund":
            refund_amount_by_period[period_key] += amount

    validator.check(
        len(successful_charge_count_by_subscription) == len(subscriptions),
        "every subscription has at least one successful charge",
    )
    for period_key, refund_amount in refund_amount_by_period.items():
        validator.check(
            refund_amount <= successful_amount_by_period[period_key],
            f"refund does not exceed successful charge for period {period_key}",
        )

    for row in tickets:
        ticket_id = int(row["ticket_id"])
        customer_id = int(row["customer_id"])
        created_at = parse_datetime(row["created_at"])
        closed_at = parse_datetime(row["closed_at"]) if row["closed_at"] else None
        customer = customer_by_id.get(customer_id)

        validator.check(
            customer is not None,
            f"ticket {ticket_id}: customer foreign key exists",
        )
        if customer is not None:
            validator.check(
                created_at.date() >= parse_date(customer["registered_at"]),
                f"ticket {ticket_id}: created after customer registration",
            )
        validator.check(
            DATA_START <= created_at.date() <= DATA_END,
            f"ticket {ticket_id}: created_at is inside corpus period",
        )
        if row["status"] == "open":
            validator.check(
                closed_at is None,
                f"ticket {ticket_id}: open status has no closed_at",
            )
        elif row["status"] == "closed":
            validator.check(
                closed_at is not None and closed_at >= created_at,
                f"ticket {ticket_id}: closed status has a valid closed_at",
            )
        else:
            validator.check(False, f"ticket {ticket_id}: status is valid")

    release_date = date(2026, 2, 15)
    before_start = release_date - timedelta(days=29)
    before_end = release_date - timedelta(days=1)
    after_start = release_date
    after_end = release_date + timedelta(days=28)
    technical_before = sum(
        1
        for row in tickets
        if row["category"] == "technical"
        and before_start <= parse_datetime(row["created_at"]).date() <= before_end
    )
    technical_after = sum(
        1
        for row in tickets
        if row["category"] == "technical"
        and after_start <= parse_datetime(row["created_at"]).date() <= after_end
    )
    validator.check(
        technical_after > technical_before * 2,
        f"release 2.0 ticket spike exists: before={technical_before}, after={technical_after}",
    )

    enterprise_cancellations_q2 = sum(
        1
        for row in subscriptions
        if customer_by_id[int(row["customer_id"])]["segment"] == "enterprise"
        and row["ended_at"]
        and date(2026, 4, 1) <= parse_date(row["ended_at"]) <= date(2026, 6, 30)
    )
    enterprise_cancellations_q3 = sum(
        1
        for row in subscriptions
        if customer_by_id[int(row["customer_id"])]["segment"] == "enterprise"
        and row["ended_at"]
        and date(2026, 7, 1) <= parse_date(row["ended_at"]) <= date(2026, 9, 30)
    )
    validator.check(
        enterprise_cancellations_q3 > enterprise_cancellations_q2 * 2,
        "enterprise cancellation increase exists: "
        f"Q2={enterprise_cancellations_q2}, Q3={enterprise_cancellations_q3}",
    )

    first_successful_charge: dict[int, date] = {}
    recognized_revenue: dict[str, Decimal] = defaultdict(Decimal)

    for row in payments:
        if row["status"] != "succeeded":
            continue
        subscription = subscription_by_id[int(row["subscription_id"])]
        customer_id = int(subscription["customer_id"])
        paid_at = parse_date(row["paid_at"])

        if row["operation_type"] == "charge":
            current_first = first_successful_charge.get(customer_id)
            if current_first is None or paid_at < current_first:
                first_successful_charge[customer_id] = paid_at

        months = 12 if subscription["billing_cycle"] == "annual" else 1
        signed_amount = Decimal(row["amount_rub"])
        if row["operation_type"] == "refund":
            signed_amount *= Decimal("-1")
        monthly_amount = signed_amount / months
        period_start = parse_date(row["billing_period_start"])
        for offset in range(months):
            allocation_date = add_months(period_start, offset)
            if allocation_date <= DATA_END:
                recognized_revenue[month_key(allocation_date)] += monthly_amount

    new_paying_by_month = Counter(month_key(value) for value in first_successful_charge.values())
    july_new = new_paying_by_month["2026-07"]
    august_new = new_paying_by_month["2026-08"]
    july_revenue = recognized_revenue["2026-07"]
    august_revenue = recognized_revenue["2026-08"]
    validator.check(
        august_new > july_new and august_revenue < july_revenue,
        "controlled August event exists: "
        f"new clients {july_new}->{august_new}, "
        f"revenue {july_revenue:.2f}->{august_revenue:.2f}",
    )

    operation_counts = Counter(
        (row["operation_type"], row["status"]) for row in payments
    )
    billing_cycle_counts = Counter(row["billing_cycle"] for row in subscriptions)

    print("Row counts:")
    for name, rows in datasets.items():
        print(f"  {name}: {len(rows)}")
    print("Segment counts:")
    for name in ("micro_business", "small_business", "mid_market", "enterprise"):
        print(f"  {name}: {segment_counts[name]}")
    print("Billing cycles:")
    for name in ("monthly", "annual"):
        print(f"  {name}: {billing_cycle_counts[name]}")
    print("Payment operations:")
    for key in sorted(operation_counts):
        print(f"  {key[0]}/{key[1]}: {operation_counts[key]}")
    print("Controlled events:")
    print(f"  technical tickets before/after release 2.0: {technical_before}/{technical_after}")
    print(
        "  enterprise cancellations Q2/Q3 2026: "
        f"{enterprise_cancellations_q2}/{enterprise_cancellations_q3}"
    )
    print(
        "  July/August new paying clients: "
        f"{july_new}/{august_new}"
    )
    print(
        "  July/August recognized revenue: "
        f"{july_revenue:.2f}/{august_revenue:.2f} RUB"
    )

    if validator.errors:
        print(f"\nFAILED: {len(validator.errors)} validation error(s)")
        for error in validator.errors[:50]:
            print(f"  - {error}")
        if len(validator.errors) > 50:
            print(f"  ... and {len(validator.errors) - 50} more")
        return 1

    print(f"\nPASSED: {len(validator.passed)} checks")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
