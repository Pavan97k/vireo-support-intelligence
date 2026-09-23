from pathlib import Path
import pandas as pd

from data_loader import load_data
from data_cleaning import clean_tickets


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


# Business assumptions from the support policy
SLA_CREDIT_INR = 350
INTERNAL_TRANSFER_COST_INR = 305

CURRENT_WEEKLY_VOLUME = 650
WEEKS_PER_QUARTER = 13


def calculate_kpis(df):
    """
    Calculate overall support KPIs.
    """

    total_tickets = len(df)

    sla_breaches = int(df["sla_breach"].sum())

    sla_breach_rate = (
        sla_breaches / total_tickets
        if total_tickets
        else 0
    )

    sla_credit_exposure = (
        sla_breaches * SLA_CREDIT_INR
    )

    repeat_contacts = int(
        df["repeat_contact_proxy"].sum()
    )

    repeat_rate = (
        repeat_contacts / total_tickets
        if total_tickets
        else 0
    )

    repeat_contact_cost = df.loc[
        df["repeat_contact_proxy"],
        "contact_cost_inr"
    ].sum()

    transfer_tickets = int(
        (df["transfers"] > 0).sum()
    )

    transfer_rate = (
        transfer_tickets / total_tickets
        if total_tickets
        else 0
    )

    transfer_cost = (
        df["transfers"].sum()
        * INTERNAL_TRANSFER_COST_INR
    )

    total_contact_cost = (
        df["contact_cost_inr"].sum()
    )

    total_refunds = (
        df["refund_amount_inr"].fillna(0).sum()
    )

    csat = df["csat_score"].dropna()

    average_csat = (
        csat.mean()
        if not csat.empty
        else None
    )

    kpis = {
        "total_tickets": total_tickets,
        "sla_breaches": sla_breaches,
        "sla_breach_rate": sla_breach_rate,
        "sla_credit_exposure_inr": sla_credit_exposure,
        "repeat_contacts": repeat_contacts,
        "repeat_rate": repeat_rate,
        "repeat_contact_cost_inr": repeat_contact_cost,
        "transfer_tickets": transfer_tickets,
        "transfer_rate": transfer_rate,
        "transfer_cost_inr": transfer_cost,
        "total_contact_cost_inr": total_contact_cost,
        "total_refunds_inr": total_refunds,
        "average_csat": average_csat,
    }

    return kpis


def calculate_weekly_volume(df):
    """
    Calculate ticket volume by week.
    """

    weekly = (
        df.set_index("created_at")
        .resample("W-MON")
        .size()
        .reset_index(name="tickets")
    )

    weekly["week"] = (
        weekly["created_at"]
        .dt.strftime("%Y-%m-%d")
    )

    weekly = weekly[
        ["week", "tickets"]
    ]

    return weekly


def calculate_category_metrics(df):
    """
    Calculate performance metrics by support category.
    """

    category = (
        df.groupby("category")
        .agg(
            tickets=("ticket_id", "count"),
            sla_breaches=("sla_breach", "sum"),
            repeat_contacts=(
                "repeat_contact_proxy",
                "sum"
            ),
            transfers=(
                "transfers",
                "sum"
            ),
            refunds_inr=(
                "refund_amount_inr",
                "sum"
            )
        )
        .reset_index()
    )

    category["sla_breach_rate"] = (
        category["sla_breaches"]
        / category["tickets"]
    )

    category["repeat_rate"] = (
        category["repeat_contacts"]
        / category["tickets"]
    )

    category["transfer_rate"] = (
        category["transfers"]
        / category["tickets"]
    )

    return category.sort_values(
        "tickets",
        ascending=False
    )


def calculate_agent_leaderboard(df, agents):
    """
    Create a Tier-1-only agent leaderboard.

    Tier-2 Escalations & Warranty agents are excluded
    from ticket-volume ranking.
    """

    agents_clean = agents.copy()

    # Normalize dates
    agents_clean["from_date"] = pd.to_datetime(
        agents_clean["from_date"],
        errors="coerce"
    )

    agents_clean["to_date"] = pd.to_datetime(
        agents_clean["to_date"],
        errors="coerce"
    )

    # Only Tier 1 agents
    tier1_agents = agents_clean[
        agents_clean["tier"].astype(str) == "1"
    ].copy()

    # Ticket-level agent metrics
    agent_metrics = (
        df.groupby("agent_id")
        .agg(
            tickets_closed=(
                "ticket_id",
                "count"
            ),
            sla_breaches=(
                "sla_breach",
                "sum"
            ),
            repeat_contacts=(
                "repeat_contact_proxy",
                "sum"
            ),
            transfers=(
                "transfers",
                "sum"
            ),
            average_csat=(
                "csat_score",
                "mean"
            )
        )
        .reset_index()
    )

    # Join agent information
    leaderboard = agent_metrics.merge(
        tier1_agents[
            [
                "agent_id",
                "name",
                "team",
                "site",
                "shift",
                "tier"
            ]
        ],
        on="agent_id",
        how="inner"
    )

    leaderboard["sla_breach_rate"] = (
        leaderboard["sla_breaches"]
        / leaderboard["tickets_closed"]
    )

    leaderboard["repeat_rate"] = (
        leaderboard["repeat_contacts"]
        / leaderboard["tickets_closed"]
    )

    leaderboard["transfer_rate"] = (
        leaderboard["transfers"]
        / leaderboard["tickets_closed"]
    )

    # We are intentionally sorting by tickets closed
    # because that is the client's requested leaderboard.
    leaderboard = leaderboard.sort_values(
        "tickets_closed",
        ascending=False
    )

    return leaderboard


def calculate_business_scenario(df):
    """
    Model the potential quarterly value of reducing
    repeat contacts from the current observed proxy rate
    to a 10% target.

    This is a scenario, NOT claimed realized savings.
    """

    current_rate = df["repeat_contact_proxy"].mean()

    target_rate = 0.10

    quarterly_tickets = (
        CURRENT_WEEKLY_VOLUME
        * WEEKS_PER_QUARTER
    )

    avoidable_repeat_contacts = max(
        0,
        (current_rate - target_rate)
        * quarterly_tickets
    )

    repeat_rows = df[
        df["repeat_contact_proxy"]
    ]

    if len(repeat_rows) > 0:
        average_repeat_contact_cost = (
            repeat_rows["contact_cost_inr"]
            .mean()
        )
    else:
        average_repeat_contact_cost = 0

    quarterly_savings = (
        avoidable_repeat_contacts
        * average_repeat_contact_cost
    )

    return {
        "current_repeat_rate": current_rate,
        "target_repeat_rate": target_rate,
        "quarterly_tickets": quarterly_tickets,
        "avoidable_repeat_contacts": (
            avoidable_repeat_contacts
        ),
        "average_repeat_contact_cost_inr": (
            average_repeat_contact_cost
        ),
        "potential_quarterly_savings_inr": (
            quarterly_savings
        )
    }


def print_kpis(kpis):
    """
    Display KPI summary.
    """

    print("\n")
    print("=" * 60)
    print("VIREO SUPPORT KPI SUMMARY")
    print("=" * 60)

    print(
        f"Total tickets:              "
        f"{kpis['total_tickets']:,}"
    )

    print(
        f"SLA breaches:               "
        f"{kpis['sla_breaches']:,}"
    )

    print(
        f"SLA breach rate:            "
        f"{kpis['sla_breach_rate']:.2%}"
    )

    print(
        f"SLA credit exposure:        "
        f"₹{kpis['sla_credit_exposure_inr']:,.0f}"
    )

    print(
        f"Repeat contacts:            "
        f"{kpis['repeat_contacts']:,}"
    )

    print(
        f"Repeat-contact rate:         "
        f"{kpis['repeat_rate']:.2%}"
    )

    print(
        f"Repeat-contact cost:         "
        f"₹{kpis['repeat_contact_cost_inr']:,.0f}"
    )

    print(
        f"Transfer tickets:            "
        f"{kpis['transfer_tickets']:,}"
    )

    print(
        f"Transfer rate:               "
        f"{kpis['transfer_rate']:.2%}"
    )

    print(
        f"Transfer cost:               "
        f"₹{kpis['transfer_cost_inr']:,.0f}"
    )

    print(
        f"Total contact cost:          "
        f"₹{kpis['total_contact_cost_inr']:,.0f}"
    )

    print(
        f"Total refunds:               "
        f"₹{kpis['total_refunds_inr']:,.0f}"
    )

    if kpis["average_csat"] is not None:
        print(
            f"Average CSAT:                "
            f"{kpis['average_csat']:.2f}"
        )

    print("=" * 60)


def print_business_scenario(scenario):
    """
    Display business scenario.
    """

    print("\n")
    print("=" * 60)
    print("BUSINESS VALUE SCENARIO")
    print("=" * 60)

    print(
        f"Current repeat-contact proxy: "
        f"{scenario['current_repeat_rate']:.2%}"
    )

    print(
        f"Target:                        "
        f"{scenario['target_repeat_rate']:.2%}"
    )

    print(
        f"Quarterly tickets assumed:     "
        f"{scenario['quarterly_tickets']:,}"
    )

    print(
        f"Avoidable repeat contacts:     "
        f"{scenario['avoidable_repeat_contacts']:.0f}"
    )

    print(
        f"Average repeat contact cost:   "
        f"₹{scenario['average_repeat_contact_cost_inr']:.2f}"
    )

    print(
        f"Potential quarterly savings:   "
        f"₹{scenario['potential_quarterly_savings_inr']:,.0f}"
    )

    print("=" * 60)


def save_outputs(weekly, category, leaderboard):
    """
    Save analytical outputs as CSV files.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    weekly.to_csv(
        OUTPUT_DIR / "weekly_volume.csv",
        index=False
    )

    category.to_csv(
        OUTPUT_DIR / "category_metrics.csv",
        index=False
    )

    leaderboard.to_csv(
        OUTPUT_DIR / "tier1_agent_leaderboard.csv",
        index=False
    )

    print("\nOutput files saved:")
    print(" - outputs/weekly_volume.csv")
    print(" - outputs/category_metrics.csv")
    print(" - outputs/tier1_agent_leaderboard.csv")


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    datasets = load_data()

    # ---------------------------------------------------------
    # Clean tickets
    # ---------------------------------------------------------

    tickets = clean_tickets(
        datasets["tickets"]
    )

    # ---------------------------------------------------------
    # Calculate KPIs
    # ---------------------------------------------------------

    kpis = calculate_kpis(tickets)

    print_kpis(kpis)

    # ---------------------------------------------------------
    # Weekly volume
    # ---------------------------------------------------------

    weekly = calculate_weekly_volume(
        tickets
    )

    # ---------------------------------------------------------
    # Category metrics
    # ---------------------------------------------------------

    category = calculate_category_metrics(
        tickets
    )

    # ---------------------------------------------------------
    # Agent leaderboard
    # ---------------------------------------------------------

    leaderboard = calculate_agent_leaderboard(
        tickets,
        datasets["agents"]
    )

    # ---------------------------------------------------------
    # Business scenario
    # ---------------------------------------------------------

    scenario = calculate_business_scenario(
        tickets
    )

    print_business_scenario(
        scenario
    )

    # ---------------------------------------------------------
    # Save analytical outputs
    # ---------------------------------------------------------

    save_outputs(
        weekly,
        category,
        leaderboard
    )

    print("\nSUCCESS: Metrics pipeline completed.")