"""Build a banking-app-style spending dashboard from the transaction CSV.

The output is a single HTML file (no server needed) with a spending donut chart
by category, income/spending/net totals, and a sortable, searchable transaction
table. It is a visual aid for humans only; nothing here is sent to the LLM.

Run from the project root:
    python scripts/build_dashboard.py
"""

import json
import webbrowser
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "sample_transactions.csv"
TEMPLATE_PATH = Path(__file__).resolve().parent / "dashboard_template.html"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "spending_dashboard.html"
DATA_PLACEHOLDER = "__TRANSACTIONS_JSON__"


def load_transactions(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def build_html(transactions: list[dict], template: str) -> str:
    """Embed the transactions into the HTML template as a JSON array."""
    data_json = json.dumps(transactions)
    # A "</" inside a <script> block could end the script early, so escape it.
    data_json = data_json.replace("</", "<\\/")
    return template.replace(DATA_PLACEHOLDER, data_json)


def main() -> None:
    transactions = load_transactions(CSV_PATH)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = build_html(transactions, template)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Built dashboard with {len(transactions)} transactions: {OUTPUT_PATH}")

    webbrowser.open(OUTPUT_PATH.as_uri())


if __name__ == "__main__":
    main()
