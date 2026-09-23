from pathlib import Path
import pandas as pd

from data_loader import load_data


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

CUTOFF_DATE = pd.Timestamp("2025-09-14")


def clean_tickets(tickets):
    """
    Clean and deduplicate the ticket dataset.
    """

    df = tickets.copy()

    print("\nStarting ticket cleaning...")
    print(f"Original rows: {len(df):,}")

    # ---------------------------------------------------------
    # 1. Convert timestamps
    # ---------------------------------------------------------

    date_columns = [
        "created_at",
        "first_response_at",
        "resolved_at",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # 2. Convert numeric columns
    # ---------------------------------------------------------

    numeric_columns = [
        "transfers",
        "csat_score",
        "refund_amount_inr",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # 3. Normalize CSAT
    # ---------------------------------------------------------

    df.loc[
        df["csat_score"] == 0,
        "csat_score"
    ] = pd.NA

    # ---------------------------------------------------------
    # 4. Sort records
    # ---------------------------------------------------------

    df = df.sort_values(
        ["ticket_id", "created_at"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. Identify duplicate ticket IDs
    # ---------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=["ticket_id"],
        keep=False
    )

    duplicate_ids = (
        df.loc[
            duplicate_mask,
            "ticket_id"
        ].nunique()
    )

    print(
        f"Duplicate ticket IDs found: "
        f"{duplicate_ids:,}"
    )

    # ---------------------------------------------------------
    # 6. Choose one record per ticket
    # ---------------------------------------------------------

    def choose_record(group):

        if len(group) == 1:
            return group.iloc[0]

        created_date = group["created_at"].min()

        # Before current helpdesk cutoff:
        # prefer legacy Freshdesk record.
        if created_date < CUTOFF_DATE:

            legacy = group[
                group["source_system"]
                .astype(str)
                .str.lower()
                .eq("legacy_fd")
            ]

            if not legacy.empty:
                return legacy.iloc[0]

        # Current system:
        # prefer helpdesk record.
        helpdesk = group[
            group["source_system"]
            .astype(str)
            .str.lower()
            .eq("helpdesk")
        ]

        if not helpdesk.empty:
            return helpdesk.iloc[0]

        return group.iloc[0]

    # Build cleaned dataframe while preserving
    # the complete original row.
    selected_records = []

    for ticket_id, group in df.groupby(
        "ticket_id",
        sort=False
    ):

        selected_record = choose_record(group)

        selected_records.append(
            selected_record
        )

    df_clean = pd.DataFrame(
        selected_records
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 7. Derived response-time fields
    # ---------------------------------------------------------

    df_clean["response_time_minutes"] = (
        (
            df_clean["first_response_at"]
            - df_clean["created_at"]
        ).dt.total_seconds()
        / 60
    )

    df_clean["resolution_time_hours"] = (
        (
            df_clean["resolved_at"]
            - df_clean["first_response_at"]
        ).dt.total_seconds()
        / 3600
    )

    # ---------------------------------------------------------
    # 8. First-response SLA
    # ---------------------------------------------------------

    sla_minutes = {
        "chat": 15,
        "voice": 120,
        "social": 240,
        "email": 480,
    }

    df_clean["sla_target_minutes"] = (
        df_clean["channel"]
        .astype(str)
        .str.lower()
        .map(sla_minutes)
    )

    df_clean["sla_breach"] = (
        df_clean["response_time_minutes"]
        > df_clean["sla_target_minutes"]
    )

    # ---------------------------------------------------------
    # 9. Contact cost
    # ---------------------------------------------------------

    contact_cost = {
        "chat": 210,
        "email": 260,
        "voice": 520,
        "social": 240,
    }

    df_clean["contact_cost_inr"] = (
        df_clean["channel"]
        .astype(str)
        .str.lower()
        .map(contact_cost)
    )

    # ---------------------------------------------------------
    # 10. Repeat-contact proxy
    # ---------------------------------------------------------
    #
    # Exact policy definition requires determining whether
    # the customer contacted us again about the SAME ISSUE.
    #
    # There is no structured issue ID in the dataset.
    #
    # Therefore we use:
    #
    # same customer + same category within 30 days
    #
    # as an operational proxy.

    df_clean = df_clean.sort_values(
        [
            "customer_id",
            "category",
            "resolved_at"
        ]
    ).reset_index(drop=True)

    df_clean["previous_resolution"] = (
        df_clean
        .groupby(
            [
                "customer_id",
                "category"
            ]
        )["resolved_at"]
        .shift(1)
    )

    df_clean[
        "days_since_previous_resolution"
    ] = (
        (
            df_clean["created_at"]
            - df_clean["previous_resolution"]
        ).dt.total_seconds()
        / 86400
    )

    df_clean["repeat_contact_proxy"] = (
        df_clean[
            "days_since_previous_resolution"
        ].between(
            0,
            30,
            inclusive="both"
        )
    )

    df_clean["repeat_contact_proxy"] = (
        df_clean["repeat_contact_proxy"]
        .fillna(False)
    )

    # ---------------------------------------------------------
    # 11. Restore chronological order
    # ---------------------------------------------------------

    df_clean = df_clean.sort_values(
        "created_at"
    ).reset_index(drop=True)

    return df_clean


def save_clean_data(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "cleaned_tickets.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nCleaned data saved to:"
    )

    print(output_file)


def print_summary(df):

    print("\n" + "=" * 50)
    print("CLEANING SUMMARY")
    print("=" * 50)

    print(
        f"Clean tickets:        "
        f"{len(df):,}"
    )

    print(
        f"Unique ticket IDs:    "
        f"{df['ticket_id'].nunique():,}"
    )

    print(
        f"SLA breaches:         "
        f"{df['sla_breach'].sum():,}"
    )

    print(
        f"Repeat-contact proxy: "
        f"{df['repeat_contact_proxy'].sum():,}"
    )

    print(
        f"Tickets with transfer:"
        f" {(df['transfers'] > 0).sum():,}"
    )

    print(
        f"Total refunds:        "
        f"₹{df['refund_amount_inr'].sum():,.0f}"
    )


if __name__ == "__main__":

    datasets = load_data()

    cleaned_tickets = clean_tickets(
        datasets["tickets"]
    )

    print_summary(
        cleaned_tickets
    )

    save_clean_data(
        cleaned_tickets
    )