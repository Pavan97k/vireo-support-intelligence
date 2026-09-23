from pathlib import Path
import pandas as pd


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def find_file(keyword):
    """
    Finds a CSV file containing the given keyword.
    This allows us to work even if the original downloaded
    files have long/random filenames.
    """
    matches = list(DATA_DIR.glob(f"*{keyword}*.csv"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find a CSV containing '{keyword}' inside {DATA_DIR}"
        )

    return matches[0]


def load_data():
    """Load all Vireo support datasets."""

    files = {
        "tickets": find_file("tickets"),
        "customers": find_file("customers"),
        "orders": find_file("orders"),
        "products": find_file("products"),
        "agents": find_file("agents"),
    }

    data = {}

    for name, path in files.items():
        data[name] = pd.read_csv(path)

    return data


def validate_data(data):
    """Run basic checks before analysis."""

    required_ticket_columns = [
        "ticket_id",
        "created_at",
        "first_response_at",
        "resolved_at",
        "status",
        "channel",
        "customer_id",
        "category",
        "assigned_team",
        "agent_id",
        "transfers",
        "csat_score",
        "refund_amount_inr",
        "source_system",
    ]

    missing = [
        col
        for col in required_ticket_columns
        if col not in data["tickets"].columns
    ]

    if missing:
        raise ValueError(
            f"Missing required ticket columns: {missing}"
        )

    print("\nVIREO DATA VALIDATION")
    print("=" * 45)

    for name, df in data.items():
        print(
            f"{name:<12} "
            f"Rows: {len(df):>6} | "
            f"Columns: {len(df.columns):>2}"
        )

    tickets = data["tickets"]

    duplicate_rows = tickets.duplicated(
        subset=["ticket_id"], keep=False
    ).sum()

    duplicate_ids = (
        tickets.loc[
            tickets.duplicated("ticket_id", keep=False),
            "ticket_id",
        ]
        .nunique()
    )

    print("\nTicket quality checks")
    print("-" * 45)
    print(f"Total exported rows: {len(tickets):,}")
    print(f"Unique ticket IDs:   {tickets['ticket_id'].nunique():,}")
    print(f"Duplicate IDs:       {duplicate_ids:,}")
    print(f"Rows involved:       {duplicate_rows:,}")


if __name__ == "__main__":

    datasets = load_data()

    validate_data(datasets)

    print("\nSUCCESS: All datasets loaded.")