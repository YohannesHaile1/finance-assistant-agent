"""Generate a FAKE bank transaction CSV for the finance assistant agent.

All merchants and amounts are made up. A fixed random seed means every run
produces exactly the same file, so tests and evals stay reproducible.

Run from the project root:
    python scripts/generate_sample_data.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
START_DATE = date(2026, 4, 1)
END_DATE = date(2026, 9, 30)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"

# Everyday purchases: for each category, the chance it happens on any given day,
# the merchants it can come from (merchant name, bank-statement description),
# and the range of amounts in dollars.
DAILY_SPENDING = {
    "Groceries": {
        "chance_per_day": 0.30,
        "merchants": [
            ("FreshMart", "FRESHMART #0231"),
            ("Green Basket Grocery", "GREEN BASKET GROCERY"),
            ("Corner Market", "CORNER MARKET 14TH ST"),
        ],
        "amount_range": (25, 140),
    },
    "Dining": {
        "chance_per_day": 0.55,
        "merchants": [
            ("Luigi's Pizzeria", "LUIGIS PIZZERIA"),
            ("Sunrise Cafe", "SUNRISE CAFE"),
            ("Taco Town", "TACO TOWN #12"),
            ("Noodle House", "NOODLE HOUSE"),
            ("Burger Barn", "BURGER BARN #3"),
        ],
        "amount_range": (8, 60),
    },
    "Transport": {
        "chance_per_day": 0.45,
        "merchants": [
            ("QuickRide", "QUICKRIDE TRIP"),
            ("FuelUp Gas", "FUELUP GAS #887"),
            ("CityPark Parking", "CITYPARK PARKING GARAGE"),
        ],
        "amount_range": (8, 60),
    },
    "Shopping": {
        "chance_per_day": 0.15,
        "merchants": [
            ("MegaMart Online", "MEGAMART ONLINE ORDER"),
            ("Urban Threads", "URBAN THREADS CLOTHING"),
            ("HomeGoods Depot", "HOMEGOODS DEPOT"),
        ],
        "amount_range": (20, 150),
    },
    "Entertainment": {
        "chance_per_day": 0.08,
        "merchants": [
            ("CineMax Theaters", "CINEMAX THEATERS"),
            ("Bowl-O-Rama", "BOWL-O-RAMA LANES"),
        ],
        "amount_range": (14, 45),
    },
}

# Bills and paychecks that arrive on the same day every month:
# (day of month, merchant, description, category, amount range, type).
# A range like (1650, 1650) means a fixed amount.
MONTHLY_ITEMS = [
    (1, "Oakview Apartments", "OAKVIEW APARTMENTS RENT", "Rent", (1650, 1650), "debit"),
    (1, "Acme Corp", "ACME CORP PAYROLL DIRECT DEP", "Income", (2450, 2450), "credit"),
    (15, "Acme Corp", "ACME CORP PAYROLL DIRECT DEP", "Income", (2450, 2450), "credit"),
    (5, "City Power & Light", "CITY POWER & LIGHT AUTOPAY", "Utilities", (80, 130), "debit"),
    (10, "AquaWorks Water", "AQUAWORKS WATER UTILITY", "Utilities", (35, 55), "debit"),
    (15, "StreamNet Internet", "STREAMNET INTERNET SVC", "Utilities", (65, 65), "debit"),
    (20, "CellOne Mobile", "CELLONE MOBILE BILL", "Utilities", (45, 45), "debit"),
    (8, "StreamFlix", "STREAMFLIX SUBSCRIPTION", "Entertainment", (15.49, 15.49), "debit"),
    (22, "Tunely Music", "TUNELY MUSIC PREMIUM", "Entertainment", (10.99, 10.99), "debit"),
]

# Deliberately planted anomalies, so detect_outliers and the tests have
# something real to find. Each is (date, description, merchant, category, amount).
UNUSUALLY_LARGE_PURCHASE = (
    date(2026, 7, 18), "MEGAMART ONLINE ORDER - LAPTOP", "MegaMart Online", "Shopping", 2399.99,
)
DUPLICATE_CHARGE = (
    date(2026, 8, 12), "URBAN THREADS CLOTHING", "Urban Threads", "Shopping", 89.50,
)
DUPLICATE_CHARGE_REPEAT_DATE = date(2026, 8, 13)  # same merchant + amount, one day later
NEW_UNUSUAL_MERCHANT = (
    date(2026, 9, 6), "LUCKY STAR CASINO ATLANTIC", "Lucky Star Casino", "Entertainment", 450.00,
)

DUPLICATE_WINDOW_DAYS = 3
COLUMNS = ["date", "description", "merchant", "category", "amount", "type"]


# craftsman-ignore: WARN-PY001 - the six parameters are the six CSV columns
def make_row(
    day: date, description: str, merchant: str, category: str, amount: float, txn_type: str = "debit"
) -> dict:
    return {
        "date": day.isoformat(),
        "description": description,
        "merchant": merchant,
        "category": category,
        "amount": round(amount, 2),
        "type": txn_type,
    }


def add_monthly_items(rows: list[dict], day: date, rng: random.Random) -> None:
    """Add any bills or paychecks that fall on this day of the month."""
    for item_day, merchant, description, category, (low, high), txn_type in MONTHLY_ITEMS:
        if day.day == item_day:
            amount = rng.uniform(low, high)
            rows.append(make_row(day, description, merchant, category, amount, txn_type))


def add_daily_purchases(rows: list[dict], day: date, rng: random.Random) -> None:
    """Roll the dice for each everyday category and add any purchases that happen."""
    for category, spec in DAILY_SPENDING.items():
        if rng.random() < spec["chance_per_day"]:
            merchant, description = rng.choice(spec["merchants"])
            amount = rng.uniform(*spec["amount_range"])
            rows.append(make_row(day, description, merchant, category, amount))


def add_planted_anomalies(rows: list[dict]) -> None:
    rows.append(make_row(*UNUSUALLY_LARGE_PURCHASE))
    rows.append(make_row(*DUPLICATE_CHARGE))
    rows.append(make_row(DUPLICATE_CHARGE_REPEAT_DATE, *DUPLICATE_CHARGE[1:]))
    rows.append(make_row(*NEW_UNUSUAL_MERCHANT))


def generate_transactions(rng: random.Random) -> list[dict]:
    """Build the full list of transactions (as dicts), sorted by date."""
    rows = []
    day = START_DATE
    while day <= END_DATE:
        add_monthly_items(rows, day, rng)
        add_daily_purchases(rows, day, rng)
        day += timedelta(days=1)

    add_planted_anomalies(rows)

    # Stable sort: rows on the same date keep the order they were added in.
    rows.sort(key=lambda row: row["date"])
    return rows


def count_duplicate_pairs(rows: list[dict]) -> int:
    """Count debits that repeat the same merchant and amount within a few days.

    Expects rows sorted by date. Remembers the last date each (merchant, amount)
    was seen, so a repeat is a single dictionary lookup instead of a nested loop.
    """
    last_seen = {}
    pairs = 0
    for row in rows:
        if row["type"] != "debit":
            continue
        key = (row["merchant"], row["amount"])
        day = date.fromisoformat(row["date"])
        if key in last_seen and (day - last_seen[key]).days <= DUPLICATE_WINDOW_DAYS:
            pairs += 1
        last_seen[key] = day
    return pairs


def check_planted_anomalies(rows: list[dict]) -> None:
    """Fail loudly if random data accidentally creates extra anomalies or hides ours."""
    duplicate_pairs = count_duplicate_pairs(rows)
    assert duplicate_pairs == 1, f"expected exactly 1 duplicate pair, found {duplicate_pairs}"

    unusual_merchant = NEW_UNUSUAL_MERCHANT[2]
    unusual_rows = [row for row in rows if row["merchant"] == unusual_merchant]
    assert len(unusual_rows) == 1, "the unusual merchant should appear exactly once"


def main() -> None:
    rng = random.Random(SEED)
    rows = generate_transactions(rng)
    check_planted_anomalies(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} fake transactions to {OUTPUT_PATH}")
    print(f"Date range: {rows[0]['date']} to {rows[-1]['date']}")


if __name__ == "__main__":
    main()
