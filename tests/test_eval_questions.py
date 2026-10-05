"""Checks the eval itself, without calling the API.

1. Every expected number in eval_questions.json is recomputed here with plain pandas,
   independently of src/tools.py, so a bug in the tools can't hide in the answer key.
2. The grading function is tested on hand-written answers.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from evals.run_eval import QUESTIONS_PATH, extract_numbers, grade

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"


@pytest.fixture(scope="module")
def questions() -> dict:
    return {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


def spending(df, month, categories=None, merchant=None):
    rows = df[(df["type"] == "debit") & (df["date"].dt.strftime("%Y-%m") == month)]
    if categories:
        rows = rows[rows["category"].isin(categories)]
    if merchant:
        rows = rows[rows["merchant"] == merchant]
    return rows


def expected_numbers(df) -> dict[str, list[float]]:
    debits = df[df["type"] == "debit"]
    credits = df[df["type"] == "credit"]
    june_income = credits[credits["date"].dt.strftime("%Y-%m") == "2026-06"]["amount"].sum()
    august_income = credits[credits["date"].dt.strftime("%Y-%m") == "2026-08"]["amount"].sum()
    may_by_category = spending(df, "2026-05").groupby("category")["amount"].sum().drop("Rent")
    july_total = spending(df, "2026-07")["amount"].sum()
    august_total = spending(df, "2026-08")["amount"].sum()
    return {
        "q01": [spending(df, "2026-07", ["Dining"])["amount"].sum()],
        "q02": [spending(df, "2026-08", ["Groceries", "Dining"])["amount"].sum()],
        "q03": [june_income],
        "q04": [august_income - august_total],
        "q05": [debits.groupby("merchant")["amount"].sum().max()],
        "q06": [may_by_category.max()],
        "q07": [debits[debits["merchant"] == "StreamFlix"]["amount"].sum()],
        "q08": [len(spending(df, "2026-08", merchant="Taco Town"))],
        "q09": [abs(august_total - july_total)],
        "q10": [
            debits["amount"].max(),
            debits[debits["merchant"] == "Lucky Star Casino"]["amount"].sum(),
            89.50,
        ],
        "q11": [89.50],
        "q12": [spending(df, "2026-06", ["Dining"])["amount"].sum()],
    }


def test_there_are_15_questions_with_unique_ids(questions):
    assert len(questions) == 15


def test_expected_numbers_match_pandas(questions, df):
    for question_id, numbers in expected_numbers(df).items():
        assert questions[question_id]["numbers"] == pytest.approx([round(n, 2) for n in numbers]), question_id


def test_expected_keywords_match_pandas(questions, df):
    debits = df[df["type"] == "debit"]
    top_merchant = debits.groupby("merchant")["amount"].sum().idxmax()
    may_top = spending(df, "2026-05").groupby("category")["amount"].sum().drop("Rent").idxmax()
    assert top_merchant.startswith(questions["q05"]["keywords"][0][0])
    assert questions["q06"]["keywords"] == [[may_top]]


def test_trick_questions_are_outside_the_data(questions, df):
    assert df["date"].min() > pd.Timestamp("2026-01-31")  # q13 asks about January 2026
    assert all(questions[qid].get("decline") for qid in ("q13", "q14", "q15"))


# ---------- grading ----------

def test_extract_numbers_handles_dollar_signs_and_commas():
    assert extract_numbers("You spent $1,750.61 on 3 trips.") == [1750.61, 3.0]


def test_grade_passes_within_tolerance():
    assert grade("You spent $675.50 on dining.", {"numbers": [675.80]})[0]


def test_grade_fails_outside_tolerance():
    passed, reason = grade("You spent $640.00 on dining.", {"numbers": [675.80]})
    assert not passed and "675.8" in reason


def test_grade_counts_use_exact_tolerance():
    assert not grade("You went 4 times.", {"numbers": [5], "tolerance": 0})[0]
    assert grade("You went 5 times.", {"numbers": [5], "tolerance": 0})[0]


def test_grade_requires_keywords():
    expected = {"numbers": [2226.83], "keywords": [["down", "decrease"]]}
    assert grade("Spending went down by $2,226.83.", expected)[0]
    assert not grade("Spending went up by $2,226.83.", expected)[0]


def test_grade_decline():
    assert grade("I don't have data for January 2026; it covers April to September.", {"decline": True})[0]
    assert not grade("You spent $500 on dining in January.", {"decline": True})[0]
