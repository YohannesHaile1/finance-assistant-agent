"""Read-only data tools the agent can call.

Every tool:
  * takes the transactions DataFrame as its first argument (so tests can pass in
    small hand-made data), followed by the arguments the LLM supplies;
  * validates its inputs and raises ToolInputError with a message the LLM can
    read and act on;
  * returns plain dicts/lists/strings/numbers, so the result can go straight
    into JSON for the LLM.

There is no LLM code in this file.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"
MAX_LIMIT = 50
Z_SCORE_THRESHOLD = 3.0
DUPLICATE_WINDOW_DAYS = 3
MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")


class ToolInputError(ValueError):
    """Bad input from the LLM. The message is sent back so the LLM can fix its call."""


# ---------- loading ----------

def load_transactions(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


# ---------- small helpers ----------

def _money(value: float) -> float:
    """Round to cents and convert numpy numbers to plain Python floats (JSON-friendly)."""
    return round(float(value), 2)


def _data_range_text(df: pd.DataFrame) -> str:
    first, last = df["date"].min(), df["date"].max()
    return f"Data covers {first:%Y-%m-%d} to {last:%Y-%m-%d}."


def _debits(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["type"] == "debit"]


def _rows_between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows whose date is between start and end, inclusive on both ends."""
    return df[df["date"].between(start, end)]


def _transaction_to_dict(row: pd.Series) -> dict:
    return {
        "date": f"{row['date']:%Y-%m-%d}",
        "description": row["description"],
        "merchant": row["merchant"],
        "category": row["category"],
        "amount": _money(row["amount"]),
        "type": row["type"],
    }


# ---------- input validation ----------

def parse_date(df: pd.DataFrame, value: str, field_name: str) -> pd.Timestamp:
    """Turn a 'YYYY-MM-DD' string into a Timestamp, checking it falls inside the data."""
    if not isinstance(value, str):
        raise ToolInputError(f"{field_name} must be a string in YYYY-MM-DD format, got {value!r}.")
    try:
        parsed = pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))
    except ValueError:
        raise ToolInputError(f"{field_name} '{value}' is not a valid YYYY-MM-DD date.") from None

    if not df["date"].min() <= parsed <= df["date"].max():
        raise ToolInputError(f"{field_name} {value} is outside the available data. {_data_range_text(df)}")
    return parsed


def parse_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = parse_date(df, start_date, "start_date")
    end = parse_date(df, end_date, "end_date")
    if start > end:
        raise ToolInputError(f"start_date {start_date} is after end_date {end_date}.")
    return start, end


def parse_month(df: pd.DataFrame, value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Turn 'YYYY-MM' into (first day, last day) of that month, checking the data covers it."""
    if not isinstance(value, str) or not MONTH_PATTERN.match(value):
        raise ToolInputError(f"month {value!r} must be in YYYY-MM format, for example 2026-07.")
    try:
        period = pd.Period(value, freq="M")
    except ValueError:
        raise ToolInputError(f"month '{value}' is not a real month.") from None

    start, end = period.start_time.normalize(), period.end_time.normalize()
    if end < df["date"].min() or start > df["date"].max():
        raise ToolInputError(f"There is no data for {value}. {_data_range_text(df)}")
    return start, end


def validate_limit(limit: int) -> int:
    # bool is a subclass of int in Python, so reject it explicitly.
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ToolInputError(f"limit must be a whole number, got {limit!r}.")
    if not 1 <= limit <= MAX_LIMIT:
        raise ToolInputError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}.")
    return limit


# ---------- tools ----------

def spending_by_category(df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    """Total spending (debits only) per category between two dates, inclusive."""
    start, end = parse_date_range(df, start_date, end_date)
    spending = _debits(_rows_between(df, start, end))

    grouped = spending.groupby("category")["amount"].agg(["sum", "count"])
    grouped = grouped.sort_values("sum", ascending=False)
    categories = [
        {"category": category, "total": _money(row["sum"]), "transaction_count": int(row["count"])}
        for category, row in grouped.iterrows()
    ]
    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_spending": _money(spending["amount"].sum()),
        "categories": categories,
    }


def monthly_summary(df: pd.DataFrame, month: str) -> dict:
    """Income, spending, net, and the top 3 spending categories for one month (YYYY-MM)."""
    start, end = parse_month(df, month)
    rows = _rows_between(df, start, end)
    income = rows.loc[rows["type"] == "credit", "amount"].sum()
    spending = _debits(rows)

    top = spending.groupby("category")["amount"].sum().sort_values(ascending=False).head(3)
    return {
        "month": month,
        "income": _money(income),
        "spending": _money(spending["amount"].sum()),
        "net": _money(income - spending["amount"].sum()),
        "transaction_count": len(rows),
        "top_categories": [{"category": c, "total": _money(t)} for c, t in top.items()],
    }


def top_merchants(df: pd.DataFrame, start_date: str, end_date: str, limit: int = 5) -> dict:
    """The merchants with the most spending between two dates, largest first."""
    start, end = parse_date_range(df, start_date, end_date)
    limit = validate_limit(limit)
    spending = _debits(_rows_between(df, start, end))

    grouped = spending.groupby("merchant")["amount"].agg(["sum", "count"])
    grouped = grouped.sort_values("sum", ascending=False).head(limit)
    merchants = [
        {"merchant": merchant, "total": _money(row["sum"]), "transaction_count": int(row["count"])}
        for merchant, row in grouped.iterrows()
    ]
    return {"start_date": start_date, "end_date": end_date, "merchants": merchants}


def search_transactions(
    df: pd.DataFrame,
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> dict:
    """Find transactions whose merchant or description contains the query (case-insensitive).

    Dates are optional; if left out, the whole data range is searched.
    Results are newest first. match_count is the total number of matches,
    even when only `limit` of them are returned.
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolInputError("query must be a non-empty string, for example 'taco' or 'rent'.")
    limit = validate_limit(limit)
    first_day = f"{df['date'].min():%Y-%m-%d}"
    last_day = f"{df['date'].max():%Y-%m-%d}"
    start, end = parse_date_range(df, start_date or first_day, end_date or last_day)

    rows = _rows_between(df, start, end)
    needle = query.strip().lower()
    # regex=False: treat the query as plain text, so characters like "." or "(" are safe.
    matches_text = (
        rows["merchant"].str.lower().str.contains(needle, regex=False)
        | rows["description"].str.lower().str.contains(needle, regex=False)
    )
    matches = rows[matches_text].sort_values("date", ascending=False)

    return {
        "query": query,
        "match_count": len(matches),
        "returned": min(len(matches), limit),
        "transactions": [_transaction_to_dict(row) for _, row in matches.head(limit).iterrows()],
    }


def _spending_per_category(df: pd.DataFrame, month: str) -> pd.Series:
    start, end = parse_month(df, month)
    return _debits(_rows_between(df, start, end)).groupby("category")["amount"].sum()


def compare_months(df: pd.DataFrame, month_a: str, month_b: str) -> dict:
    """Compare spending (debits only) per category between two months (YYYY-MM).

    Categories are sorted by the size of the change, largest first (ties broken
    alphabetically). percent_change is None when a category had no spending in month_a.
    """
    totals_a = _spending_per_category(df, month_a)
    totals_b = _spending_per_category(df, month_b)

    categories = []
    for category in sorted(set(totals_a.index) | set(totals_b.index)):
        amount_a = float(totals_a.get(category, 0.0))
        amount_b = float(totals_b.get(category, 0.0))
        change = amount_b - amount_a
        categories.append({
            "category": category,
            "month_a": _money(amount_a),
            "month_b": _money(amount_b),
            "change": _money(change),
            "percent_change": round(change / amount_a * 100, 1) if amount_a else None,
        })
    categories.sort(key=lambda c: (-abs(c["change"]), c["category"]))

    return {
        "month_a": month_a,
        "month_b": month_b,
        "total_a": _money(totals_a.sum()),
        "total_b": _money(totals_b.sum()),
        "total_change": _money(totals_b.sum() - totals_a.sum()),
        "categories": categories,
    }


def _find_large_transactions(all_debits: pd.DataFrame, candidates: pd.DataFrame) -> list[dict]:
    """Flag candidates more than Z_SCORE_THRESHOLD standard deviations above their category's mean."""
    stats = all_debits.groupby("category")["amount"].agg(["mean", "std"])
    flagged = []
    for _, row in candidates.iterrows():
        mean, std = stats.loc[row["category"], "mean"], stats.loc[row["category"], "std"]
        if pd.isna(std) or std == 0:
            continue  # every amount is identical (or there is only one), so nothing can stand out
        z_score = (row["amount"] - mean) / std
        if z_score > Z_SCORE_THRESHOLD:
            flagged.append({
                "date": f"{row['date']:%Y-%m-%d}",
                "merchant": row["merchant"],
                "category": row["category"],
                "amount": _money(row["amount"]),
                "z_score": round(float(z_score), 1),
                "reason": (
                    f"${row['amount']:,.2f} is {z_score:.1f} standard deviations above the average "
                    f"{row['category']} transaction (${mean:,.2f})."
                ),
            })
    return flagged


def _find_duplicates(debits: pd.DataFrame) -> list[dict]:
    """Find repeat charges: same merchant and amount within DUPLICATE_WINDOW_DAYS. Expects rows sorted by date."""
    last_seen = {}
    duplicates = []
    for _, row in debits.iterrows():
        key = (row["merchant"], row["amount"])
        previous_date = last_seen.get(key)
        last_seen[key] = row["date"]
        if previous_date is None:
            continue
        days_apart = (row["date"] - previous_date).days
        if days_apart > DUPLICATE_WINDOW_DAYS:
            continue
        duplicates.append({
            "merchant": row["merchant"],
            "amount": _money(row["amount"]),
            "dates": [f"{previous_date:%Y-%m-%d}", f"{row['date']:%Y-%m-%d}"],
            "days_apart": days_apart,
            "reason": f"Same merchant and amount charged twice within {days_apart} day(s).",
        })
    return duplicates


def detect_outliers(df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    """Flag unusually large transactions and possible duplicate charges between two dates.

    Large: a debit whose z-score within its category is above Z_SCORE_THRESHOLD. The
    category's mean and standard deviation come from ALL data, so short ranges still
    have a meaningful baseline. Duplicates: same merchant and amount within
    DUPLICATE_WINDOW_DAYS days.
    """
    start, end = parse_date_range(df, start_date, end_date)
    all_debits = _debits(df)
    debits_in_range = _rows_between(all_debits, start, end).sort_values("date", kind="stable")

    return {
        "start_date": start_date,
        "end_date": end_date,
        "large_transactions": _find_large_transactions(all_debits, debits_in_range),
        "possible_duplicates": _find_duplicates(debits_in_range),
    }
