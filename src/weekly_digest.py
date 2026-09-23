from pathlib import Path
import pandas as pd

from data_loader import load_data
from data_cleaning import clean_tickets
from metrics import (
    calculate_kpis,
    calculate_category_metrics,
    calculate_agent_leaderboard,
    calculate_business_scenario,
)


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


def generate_digest(tickets, agents, ai_results=None):

    kpis = calculate_kpis(tickets)
    categories = calculate_category_metrics(tickets)
    leaderboard = calculate_agent_leaderboard(
        tickets,
        agents
    )
    scenario = calculate_business_scenario(tickets)

    lines = []

    lines.append("# Vireo Audio — Support Weekly Digest")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")

    lines.append(
        f"- Unique tickets analyzed: "
        f"{kpis['total_tickets']:,}"
    )

    lines.append(
        f"- First-response SLA breach rate: "
        f"{kpis['sla_breach_rate']:.2%}"
    )

    lines.append(
        f"- Repeat-contact proxy: "
        f"{kpis['repeat_rate']:.2%}"
    )

    lines.append(
        f"- Transfer rate: "
        f"{kpis['transfer_rate']:.2%}"
    )

    lines.append(
        f"- Total refunds: "
        f"₹{kpis['total_refunds_inr']:,.0f}"
    )

    if kpis["average_csat"] is not None:
        lines.append(
            f"- Average CSAT: "
            f"{kpis['average_csat']:.2f}"
        )

    lines.append("")

    lines.append("## Business Opportunity")
    lines.append("")

    lines.append(
        f"The observed repeat-contact proxy is "
        f"{scenario['current_repeat_rate']:.2%}. "
        f"A 10% target would correspond to approximately "
        f"{scenario['avoidable_repeat_contacts']:.0f} "
        f"fewer repeat contacts per quarter at the stated "
        f"650 tickets/week volume."
    )

    lines.append("")

    lines.append(
        f"Modeled quarterly avoided contact cost: "
        f"₹{scenario['potential_quarterly_savings_inr']:,.0f}."
    )

    lines.append("")

    lines.append(
        "*This is a modeled opportunity, not realized savings. "
        "Repeat contact is an operational proxy using the same "
        "customer + category within 30 days because the dataset "
        "does not contain a structured issue identifier.*"
    )

    lines.append("")

    # ---------------------------------------------------------
    # Category insights
    # ---------------------------------------------------------

    lines.append("## Category Signals")
    lines.append("")

    category_view = categories.sort_values(
        "repeat_rate",
        ascending=False
    ).head(5)

    for _, row in category_view.iterrows():

        lines.append(
            f"- **{row['category']}** — "
            f"{int(row['tickets']):,} tickets, "
            f"{row['repeat_rate']:.2%} repeat-contact proxy, "
            f"{row['sla_breach_rate']:.2%} SLA breach rate."
        )

    lines.append("")

    # ---------------------------------------------------------
    # AI insights
    # ---------------------------------------------------------

    lines.append("## AI-Assisted Ticket Themes")
    lines.append("")

    if ai_results is not None and not ai_results.empty:

        valid_ai = ai_results[
            ai_results["theme"] != "ERROR"
        ]

        theme_counts = (
            valid_ai["theme"]
            .value_counts()
            .head(5)
        )

        for theme, count in theme_counts.items():

            evidence_rows = valid_ai[
                valid_ai["theme"] == theme
            ]

            evidence_ids = (
                evidence_rows["ticket_id"]
                .astype(str)
                .head(3)
                .tolist()
            )

            lines.append(
                f"- **{theme}** — "
                f"{count} sampled tickets. "
                f"Evidence tickets: "
                f"{', '.join(evidence_ids)}"
            )

    else:

        lines.append(
            "- AI analysis results were not available."
        )

    lines.append("")

    # ---------------------------------------------------------
    # Tier 1 leaderboard
    # ---------------------------------------------------------

    lines.append("## Tier-1 Agent Leaderboard")
    lines.append("")

    lines.append(
        "| Agent | Team | Tickets | SLA breach | CSAT |"
    )
    lines.append(
        "|---|---|---:|---:|---:|"
    )

    for _, row in leaderboard.head(10).iterrows():

        csat = row["average_csat"]

        if pd.isna(csat):
            csat_text = "N/A"
        else:
            csat_text = f"{csat:.2f}"

        lines.append(
            f"| {row['name']} | "
            f"{row['team']} | "
            f"{int(row['tickets_closed'])} | "
            f"{row['sla_breach_rate']:.2%} | "
            f"{csat_text} |"
        )

    lines.append("")

    lines.append(
        "*Tier-2 Escalations & Warranty agents are excluded "
        "from this ticket-volume leaderboard because their "
        "cases are measured differently under the support policy.*"
    )

    lines.append("")

    # ---------------------------------------------------------
    # Data quality
    # ---------------------------------------------------------

    duplicate_ids = (
        12528 - 11875
    )

    lines.append("## Data Quality / Methodology")
    lines.append("")

    lines.append(
        f"- {duplicate_ids:,} duplicate ticket IDs were removed "
        f"during deduplication."
    )

    lines.append(
        "- Pre-cutoff duplicate records were resolved by "
        "preferring the legacy source; current-period records "
        "prefer the current helpdesk."
    )

    lines.append(
        "- Repeat contacts are an operational proxy, not an "
        "exact semantic match to the policy definition."
    )

    lines.append(
        "- Numerical KPIs are calculated deterministically; "
        "AI is used for ticket-text interpretation."
    )

    return "\n".join(lines)


def main():

    print("Generating Vireo Support Digest...")

    datasets = load_data()

    tickets = clean_tickets(
        datasets["tickets"]
    )

    ai_file = (
        OUTPUT_DIR
        / "ai_ticket_analysis.csv"
    )

    if ai_file.exists():

        ai_results = pd.read_csv(
            ai_file
        )

    else:

        ai_results = None

    digest = generate_digest(
        tickets,
        datasets["agents"],
        ai_results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "weekly_digest.md"
    )

    output_file.write_text(
        digest,
        encoding="utf-8"
    )

    print(
        f"\nDigest saved to:\n{output_file}"
    )

    print("\n" + "=" * 60)
    print(digest)
    print("=" * 60)


if __name__ == "__main__":
    main()