"""Tool definitions sent to the LLM.

The LLM never sees the Python functions, only these names, descriptions, and
input schemas. It decides which tool to call (and with what arguments) purely
from this text, so the descriptions say what each tool returns and when to use it.

The `df` argument of each function in tools.py is supplied by the agent, so it
is not part of any schema.
"""

DATE = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}
MONTH = {"type": "string", "pattern": r"^\d{4}-\d{2}$"}

TOOL_SCHEMAS = [
    {
        "name": "spending_by_category",
        "description": (
            "Total spending per category between two dates (inclusive), largest first, plus the "
            "overall total and a transaction count per category. Spending means money out (debits); "
            "income is never included. Use this for questions like 'how much did I spend on dining "
            "in July' or 'where does my money go'. Categories: Groceries, Dining, Transport, "
            "Utilities, Rent, Entertainment, Shopping."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {**DATE, "description": "First day to include, YYYY-MM-DD."},
                "end_date": {**DATE, "description": "Last day to include, YYYY-MM-DD."},
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "monthly_summary",
        "description": (
            "Income, spending, net (income minus spending), transaction count, and the top 3 "
            "spending categories for one calendar month. Use this for income questions, "
            "'did I save money in August', or a quick overview of a single month."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "month": {**MONTH, "description": "The month, YYYY-MM (for example 2026-07)."},
            },
            "required": ["month"],
            "additionalProperties": False,
        },
    },
    {
        "name": "top_merchants",
        "description": (
            "The merchants with the most spending between two dates (inclusive), largest first, "
            "with each merchant's total and number of transactions. Use this for 'where do I "
            "spend the most' or 'my biggest merchants'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {**DATE, "description": "First day to include, YYYY-MM-DD."},
                "end_date": {**DATE, "description": "Last day to include, YYYY-MM-DD."},
                "limit": {
                    "type": "integer", "minimum": 1, "maximum": 50,
                    "description": "How many merchants to return. Defaults to 5.",
                },
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_transactions",
        "description": (
            "Find individual transactions whose merchant name or bank description contains the "
            "query text (case-insensitive, plain text, not a regex). Returns matching transactions "
            "newest first, plus match_count, the total number of matches even when only `limit` are "
            "returned. Use this to look up a specific merchant ('how much have I paid StreamFlix'), "
            "list transactions, or check details of a single charge. Dates are optional and default "
            "to all available data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to look for, for example 'taco' or 'rent'."},
                "start_date": {**DATE, "description": "Optional first day to include, YYYY-MM-DD."},
                "end_date": {**DATE, "description": "Optional last day to include, YYYY-MM-DD."},
                "limit": {
                    "type": "integer", "minimum": 1, "maximum": 50,
                    "description": "Maximum transactions to return. Defaults to 20.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_months",
        "description": (
            "Compare spending per category between two calendar months: each category's amount in "
            "both months, the dollar change (month_b minus month_a), and the percent change, sorted "
            "by the biggest change. Also returns both months' totals. Use this for 'how did my "
            "spending change', 'did I spend more in August than July', or 'what went up'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "month_a": {**MONTH, "description": "The earlier ('before') month, YYYY-MM."},
                "month_b": {**MONTH, "description": "The later ('after') month, YYYY-MM."},
            },
            "required": ["month_a", "month_b"],
            "additionalProperties": False,
        },
    },
    {
        "name": "detect_outliers",
        "description": (
            "Flag unusual activity between two dates (inclusive): (1) unusually large transactions, "
            "meaning more than 3 standard deviations above the normal amount for their category, and "
            "(2) possible duplicate charges, meaning the same merchant and amount within 3 days. Each "
            "flag includes a 'reason' explaining why it was flagged. Use this for 'anything "
            "unusual?', 'any suspicious charges?', or 'was I double charged?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {**DATE, "description": "First day to check, YYYY-MM-DD."},
                "end_date": {**DATE, "description": "Last day to check, YYYY-MM-DD."},
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
]
