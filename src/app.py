import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vireo Support Intelligence",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def find_file(keyword):
    files = list(DATA_DIR.iterdir())

    for file in files:
        if keyword.lower() in file.name.lower():
            return file

    return None


@st.cache_data
def load_data():

    cleaned_file = OUTPUT_DIR / "cleaned_tickets.csv"

    if cleaned_file.exists():
        tickets = pd.read_csv(cleaned_file)
    else:
        ticket_file = find_file("tickets")

        if ticket_file is None:
            raise FileNotFoundError("tickets.csv not found in data/")

        tickets = pd.read_csv(ticket_file)

    return tickets


# --------------------------------------------------
# Load data
# --------------------------------------------------

try:
    tickets = load_data()

except Exception as e:
    st.error(f"Unable to load ticket data: {e}")
    st.stop()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Vireo Support Intelligence")

st.caption(
    "AI-assisted support analytics, SLA monitoring, "
    "repeat-contact analysis and operational insights."
)

st.divider()


# --------------------------------------------------
# Basic calculations
# --------------------------------------------------

total_tickets = len(tickets)

# SLA breach
if "sla_breach" in tickets.columns:
    breach_count = int(tickets["sla_breach"].sum())
else:
    breach_count = 0

breach_rate = (
    breach_count / total_tickets * 100
    if total_tickets
    else 0
)

# Transfers
if "transfers" in tickets.columns:
    transfer_count = int((tickets["transfers"] > 0).sum())
    transfer_rate = transfer_count / total_tickets * 100
else:
    transfer_count = 0
    transfer_rate = 0

# Refunds
if "refund_amount_inr" in tickets.columns:
    refunds = pd.to_numeric(
        tickets["refund_amount_inr"],
        errors="coerce"
    ).fillna(0).sum()
else:
    refunds = 0


# --------------------------------------------------
# KPI cards
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Tickets",
    f"{total_tickets:,}"
)

col2.metric(
    "SLA Breach Rate",
    f"{breach_rate:.2f}%"
)

col3.metric(
    "Transfer Rate",
    f"{transfer_rate:.2f}%"
)

col4.metric(
    "Refunds",
    f"₹{refunds:,.0f}"
)


st.divider()


# --------------------------------------------------
# Business opportunity
# --------------------------------------------------

st.subheader("💰 Business Opportunity")

current_repeat_rate = 12.58
target_repeat_rate = 10.0
weekly_volume = 650
weeks_per_quarter = 13
average_repeat_cost = 259.41

current_contacts = (
    weekly_volume *
    weeks_per_quarter *
    current_repeat_rate / 100
)

target_contacts = (
    weekly_volume *
    weeks_per_quarter *
    target_repeat_rate / 100
)

avoided_contacts = current_contacts - target_contacts

quarterly_saving = (
    avoided_contacts *
    average_repeat_cost
)

st.info(
    f"""
Operational repeat-contact proxy: **{current_repeat_rate:.2f}%**

Target: **{target_repeat_rate:.1f}%**

At the stated planning volume of **650 tickets/week**, this
would avoid approximately **{avoided_contacts:.0f} repeat contacts
per quarter**, representing approximately **₹{quarterly_saving:,.0f}
per quarter** in avoided contact cost.

This is a modeled opportunity, not a realized saving.
"""
)


# --------------------------------------------------
# Weekly volume
# --------------------------------------------------

st.subheader("📈 Weekly Ticket Volume")

if "created_at" in tickets.columns:

    tickets["created_at"] = pd.to_datetime(
        tickets["created_at"],
        errors="coerce"
    )

    weekly = (
        tickets
        .dropna(subset=["created_at"])
        .set_index("created_at")
        .resample("W")
        .size()
        .reset_index(name="tickets")
    )

    st.line_chart(
        weekly.set_index("created_at")["tickets"]
    )

else:
    st.warning("created_at column not available.")


# --------------------------------------------------
# Category analysis
# --------------------------------------------------

st.subheader("🔎 Category Analysis")

if "category" in tickets.columns:

    category = (
        tickets
        .groupby("category")
        .size()
        .reset_index(name="tickets")
        .sort_values("tickets", ascending=False)
    )

    st.dataframe(
        category,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# Channel analysis
# --------------------------------------------------

st.subheader("📞 Channel Analysis")

if "channel" in tickets.columns:

    channel = (
        tickets
        .groupby("channel")
        .size()
        .reset_index(name="tickets")
        .sort_values("tickets", ascending=False)
    )

    st.bar_chart(
        channel.set_index("channel")["tickets"]
    )

    st.dataframe(
        channel,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# AI analysis
# --------------------------------------------------

st.subheader("🤖 AI Ticket Analysis")

ai_file = OUTPUT_DIR / "ai_ticket_analysis.csv"

if ai_file.exists():

    ai = pd.read_csv(ai_file)

    st.write(
        f"AI analysis available for **{len(ai)} tickets**."
    )

    if "theme" in ai.columns:

        theme_counts = (
            ai["theme"]
            .value_counts()
            .reset_index()
        )

        theme_counts.columns = [
            "theme",
            "tickets"
        ]

        st.bar_chart(
            theme_counts.set_index("theme")["tickets"]
        )

        st.dataframe(
            ai,
            use_container_width=True,
            hide_index=True
        )

else:

    st.warning(
        "AI analysis file not found. "
        "Run src/ai_analysis.py locally to generate it."
    )


# --------------------------------------------------
# Data quality
# --------------------------------------------------

st.subheader("⚠️ Data Quality Notes")

st.markdown(
    """
- Duplicate ticket IDs were identified across the legacy and current helpdesk exports.
- Duplicate IDs were deduplicated before KPI calculations.
- Repeat-contact analysis uses a customer + category + 30-day operational proxy.
- Tier 2 Escalations & Warranty cases should not be compared with Tier 1 using tickets closed per week.
- Modeled savings represent potential opportunity rather than realized financial impact.
"""
)


st.divider()

st.caption(
    "Vireo Support Intelligence | AI-assisted support analytics"
)