"""Unit tests for the data tools.

Most tests use `tiny_df`: 15 hand-made transactions small enough that every
expected value can be checked with mental math. A few tests use the generated
sample CSV to confirm the planted anomalies are found.
"""

import json

import pandas as pd
import pytest

from src.tools import (
    ToolInputError,
    compare_months,
    detect_outliers,
    load_transactions,
    monthly_summary,
    search_transactions,
    spending_by_category,
    top_merchants,
)

TINY_ROWS = [
    # date,        description,               merchant,             category,        amount, type
    ("2026-01-01", "ACME CORP PAYROLL",       "Acme Corp",          "Income",        3000.0, "credit"),
    ("2026-01-01", "OAKVIEW APARTMENTS RENT", "Oakview Apartments", "Rent",          1000.0, "debit"),
    ("2026-01-05", "FRESHMART #0231",         "FreshMart",          "Groceries",      100.0, "debit"),
    ("2026-01-10", "CORNER MARKET 14TH ST",   "Corner Market",      "Groceries",       50.0, "debit"),
    ("2026-01-12", "TACO TOWN #12",           "Taco Town",          "Dining",          30.0, "debit"),
    ("2026-01-20", "TACO TOWN #12",           "Taco Town",          "Dining",          20.0, "debit"),
    ("2026-01-25", "URBAN THREADS CLOTHING",  "Urban Threads",      "Shopping",       200.0, "debit"),
    ("2026-02-01", "ACME CORP PAYROLL",       "Acme Corp",          "Income",        3000.0, "credit"),
    ("2026-02-01", "OAKVIEW APARTMENTS RENT", "Oakview Apartments", "Rent",          1000.0, "debit"),
    ("2026-02-03", "FRESHMART #0231",         "FreshMart",          "Groceries",      120.0, "debit"),
    ("2026-02-14", "LUIGIS PIZZERIA",         "Luigi's Pizzeria",   "Dining",          80.0, "debit"),
    ("2026-02-15", "QUICKRIDE TRIP",          "QuickRide",          "Transport",       40.0, "debit"),
    ("2026-02-20", "URBAN THREADS CLOTHING",  "Urban Threads",      "Shopping",        55.0, "debit"),
    ("2026-02-21", "URBAN THREADS CLOTHING",  "Urban Threads",      "Shopping",        55.0, "debit"),
    ("2026-02-28", "STREAMFLIX SUBSCRIPTION", "StreamFlix",         "Entertainment",   15.0, "debit"),
]
# Mental-math cheat sheet:
#   January:  income 3000, spending 1000+100+50+30+20+200 = 1400, net 1600
#   February: income 3000, spending 1000+120+80+40+55+55+15 = 1365, net 1635


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    df = pd.DataFrame(TINY_ROWS, columns=["date", "description", "merchant", "category", "amount", "type"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return load_transactions()


def by_key(items: list[dict], key: str) -> dict:
    """Turn a list of dicts into a dict keyed by one field, for easy lookups in asserts."""
    return {item[key]: item for item in items}


# ---------- spending_by_category ----------

def test_spending_by_category_january(tiny_df):
    result = spending_by_category(tiny_df, "2026-01-01", "2026-01-31")

    assert result["total_spending"] == 1400.0
    assert [c["category"] for c in result["categories"]] == ["Rent", "Shopping", "Groceries", "Dining"]
    groceries = by_key(result["categories"], "category")["Groceries"]
    assert groceries == {"category": "Groceries", "total": 150.0, "transaction_count": 2}


def test_spending_by_category_excludes_income(tiny_df):
    result = spending_by_category(tiny_df, "2026-01-01", "2026-02-28")
    assert "Income" not in by_key(result["categories"], "category")


def test_spending_by_category_end_date_is_inclusive(tiny_df):
    result = spending_by_category(tiny_df, "2026-02-28", "2026-02-28")
    assert result["total_spending"] == 15.0


def test_spending_by_category_empty_range(tiny_df):
    # No transactions between Jan 26 and Jan 31.
    result = spending_by_category(tiny_df, "2026-01-26", "2026-01-31")
    assert result["total_spending"] == 0.0
    assert result["categories"] == []


def test_spending_by_category_on_sample_data(sample_df):
    result = spending_by_category(sample_df, "2026-04-01", "2026-09-30")
    totals = {c["category"]: c["total"] for c in result["categories"]}
    assert totals["Rent"] == 9900.0
    assert totals["Shopping"] == 4973.95
    assert result["total_spending"] == 28554.71


# ---------- input validation (shared by all tools) ----------

@pytest.mark.parametrize("bad_date", ["2026-13-01", "01/05/2026", "yesterday", "", "2026-02-30"])
def test_invalid_date_format_is_rejected(tiny_df, bad_date):
    with pytest.raises(ToolInputError, match="not a valid YYYY-MM-DD date"):
        spending_by_category(tiny_df, bad_date, "2026-01-31")


def test_non_string_date_is_rejected(tiny_df):
    with pytest.raises(ToolInputError, match="must be a string"):
        spending_by_category(tiny_df, 20260101, "2026-01-31")


def test_start_after_end_is_rejected(tiny_df):
    with pytest.raises(ToolInputError, match="is after end_date"):
        spending_by_category(tiny_df, "2026-02-01", "2026-01-01")


def test_date_outside_data_is_rejected_with_the_data_range(tiny_df):
    with pytest.raises(ToolInputError, match="Data covers 2026-01-01 to 2026-02-28"):
        spending_by_category(tiny_df, "2025-12-01", "2026-01-31")


@pytest.mark.parametrize("bad_limit", [0, -1, 51, 2.5, "5", True])
def test_invalid_limit_is_rejected(tiny_df, bad_limit):
    with pytest.raises(ToolInputError, match="limit"):
        top_merchants(tiny_df, "2026-01-01", "2026-02-28", limit=bad_limit)


# ---------- monthly_summary ----------

def test_monthly_summary_february(tiny_df):
    result = monthly_summary(tiny_df, "2026-02")
    assert result["income"] == 3000.0
    assert result["spending"] == 1365.0
    assert result["net"] == 1635.0
    assert result["transaction_count"] == 8
    assert result["top_categories"] == [
        {"category": "Rent", "total": 1000.0},
        {"category": "Groceries", "total": 120.0},
        {"category": "Shopping", "total": 110.0},
    ]


@pytest.mark.parametrize("bad_month", ["2026-1", "Feb 2026", "2026-02-01", "2026-13"])
def test_monthly_summary_rejects_bad_month_format(tiny_df, bad_month):
    with pytest.raises(ToolInputError):
        monthly_summary(tiny_df, bad_month)


def test_monthly_summary_rejects_month_without_data(tiny_df):
    with pytest.raises(ToolInputError, match="There is no data for 2025-12"):
        monthly_summary(tiny_df, "2025-12")


# ---------- top_merchants ----------

def test_top_merchants(tiny_df):
    result = top_merchants(tiny_df, "2026-01-01", "2026-02-28", limit=3)
    assert result["merchants"] == [
        {"merchant": "Oakview Apartments", "total": 2000.0, "transaction_count": 2},
        {"merchant": "Urban Threads", "total": 310.0, "transaction_count": 3},
        {"merchant": "FreshMart", "total": 220.0, "transaction_count": 2},
    ]


def test_top_merchants_ignores_income(tiny_df):
    result = top_merchants(tiny_df, "2026-01-01", "2026-02-28", limit=50)
    assert "Acme Corp" not in by_key(result["merchants"], "merchant")


# ---------- search_transactions ----------

def test_search_is_case_insensitive_and_newest_first(tiny_df):
    result = search_transactions(tiny_df, "taco")
    assert result["match_count"] == 2
    assert [t["date"] for t in result["transactions"]] == ["2026-01-20", "2026-01-12"]


def test_search_matches_description_text(tiny_df):
    # "PAYROLL" only appears in the description, not the merchant name.
    result = search_transactions(tiny_df, "payroll")
    assert result["match_count"] == 2


def test_search_respects_date_range(tiny_df):
    result = search_transactions(tiny_df, "urban", start_date="2026-02-01", end_date="2026-02-28")
    assert result["match_count"] == 2


def test_search_limit_caps_results_but_reports_total(tiny_df):
    result = search_transactions(tiny_df, "urban", limit=1)
    assert result["match_count"] == 3
    assert result["returned"] == 1
    assert len(result["transactions"]) == 1


def test_search_no_matches(tiny_df):
    result = search_transactions(tiny_df, "spaceship")
    assert result["match_count"] == 0
    assert result["transactions"] == []


def test_search_treats_special_characters_as_plain_text(tiny_df):
    # "." would match any character if treated as a regex.
    assert search_transactions(tiny_df, ".")["match_count"] == 0


def test_search_rejects_empty_query(tiny_df):
    with pytest.raises(ToolInputError, match="non-empty"):
        search_transactions(tiny_df, "   ")


# ---------- results must be JSON-serializable ----------

def test_all_results_are_json_serializable(sample_df):
    results = [
        spending_by_category(sample_df, "2026-04-01", "2026-09-30"),
        monthly_summary(sample_df, "2026-07"),
        top_merchants(sample_df, "2026-04-01", "2026-09-30", limit=5),
        search_transactions(sample_df, "megamart", limit=5),
    ]
    for result in results:
        json.dumps(result)  # raises TypeError if a numpy type slipped through


# ---------- compare_months ----------

def test_compare_months_totals(tiny_df):
    result = compare_months(tiny_df, "2026-01", "2026-02")
    assert result["month_a"] == "2026-01"
    assert result["month_b"] == "2026-02"
    assert result["total_a"] == 1400.0
    assert result["total_b"] == 1365.0
    assert result["total_change"] == -35.0


def test_compare_months_category_changes(tiny_df):
    categories = by_key(compare_months(tiny_df, "2026-01", "2026-02")["categories"], "category")
    assert categories["Shopping"] == {
        "category": "Shopping", "month_a": 200.0, "month_b": 110.0, "change": -90.0, "percent_change": -45.0,
    }
    assert categories["Dining"]["percent_change"] == 60.0
    assert categories["Rent"]["change"] == 0.0


def test_compare_months_category_only_in_one_month(tiny_df):
    categories = by_key(compare_months(tiny_df, "2026-01", "2026-02")["categories"], "category")
    # Transport only appears in February, so there is no percentage to compute.
    assert categories["Transport"] == {
        "category": "Transport", "month_a": 0.0, "month_b": 40.0, "change": 40.0, "percent_change": None,
    }


def test_compare_months_excludes_income(tiny_df):
    categories = by_key(compare_months(tiny_df, "2026-01", "2026-02")["categories"], "category")
    assert "Income" not in categories


def test_compare_months_sorted_by_size_of_change(tiny_df):
    names = [c["category"] for c in compare_months(tiny_df, "2026-01", "2026-02")["categories"]]
    # |change|: Shopping 90, Transport 40, Dining 30, Groceries 30 (tie -> alphabetical), Entertainment 15, Rent 0
    assert names == ["Shopping", "Transport", "Dining", "Groceries", "Entertainment", "Rent"]


def test_compare_months_rejects_month_without_data(tiny_df):
    with pytest.raises(ToolInputError, match="There is no data"):
        compare_months(tiny_df, "2026-01", "2026-05")


def test_compare_months_is_json_serializable(tiny_df):
    json.dumps(compare_months(tiny_df, "2026-01", "2026-02"))


# ---------- detect_outliers ----------

def test_detect_outliers_finds_duplicate_charge(tiny_df):
    result = detect_outliers(tiny_df, "2026-01-01", "2026-02-28")
    assert len(result["possible_duplicates"]) == 1
    duplicate = result["possible_duplicates"][0]
    assert duplicate["merchant"] == "Urban Threads"
    assert duplicate["amount"] == 55.0
    assert duplicate["dates"] == ["2026-02-20", "2026-02-21"]
    assert duplicate["days_apart"] == 1
    assert duplicate["reason"]  # must explain why


def test_detect_outliers_ignores_monthly_rent(tiny_df):
    # Rent: same merchant and amount, but 31 days apart, so NOT a duplicate.
    # Its standard deviation is 0, which must not crash the z-score step.
    result = detect_outliers(tiny_df, "2026-01-01", "2026-02-28")
    flagged_merchants = [d["merchant"] for d in result["possible_duplicates"]]
    flagged_merchants += [t["merchant"] for t in result["large_transactions"]]
    assert "Oakview Apartments" not in flagged_merchants


def test_detect_outliers_duplicate_outside_range_is_not_reported(tiny_df):
    result = detect_outliers(tiny_df, "2026-01-01", "2026-01-31")
    assert result["possible_duplicates"] == []


def test_detect_outliers_empty_range(tiny_df):
    result = detect_outliers(tiny_df, "2026-01-26", "2026-01-31")
    assert result == {
        "start_date": "2026-01-26",
        "end_date": "2026-01-31",
        "large_transactions": [],
        "possible_duplicates": [],
    }


def test_detect_outliers_rejects_invalid_dates(tiny_df):
    with pytest.raises(ToolInputError):
        detect_outliers(tiny_df, "2026-02-01", "2026-01-01")


def test_detect_outliers_finds_planted_large_purchases(sample_df):
    result = detect_outliers(sample_df, "2026-04-01", "2026-09-30")
    flagged = {(t["date"], t["merchant"], t["amount"]) for t in result["large_transactions"]}
    assert flagged == {
        ("2026-07-18", "MegaMart Online", 2399.99),
        ("2026-09-06", "Lucky Star Casino", 450.0),
    }
    for transaction in result["large_transactions"]:
        assert transaction["z_score"] > 3.0
        assert transaction["reason"]


def test_detect_outliers_finds_planted_duplicate(sample_df):
    result = detect_outliers(sample_df, "2026-04-01", "2026-09-30")
    assert [(d["merchant"], d["amount"], d["dates"]) for d in result["possible_duplicates"]] == [
        ("Urban Threads", 89.5, ["2026-08-12", "2026-08-13"]),
    ]


def test_detect_outliers_baseline_uses_all_data(sample_df):
    # A one-day range around the laptop purchase is still flagged, because the
    # "normal" Shopping amount is measured over the whole dataset.
    result = detect_outliers(sample_df, "2026-07-18", "2026-07-18")
    assert [t["merchant"] for t in result["large_transactions"]] == ["MegaMart Online"]


def test_detect_outliers_is_json_serializable(sample_df):
    json.dumps(detect_outliers(sample_df, "2026-04-01", "2026-09-30"))
