from pathlib import Path
import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from data_loader import load_data
from data_cleaning import clean_tickets


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROMPT_FILE = (
    BASE_DIR
    / "prompts"
    / "ticket_analysis.txt"
)

OUTPUT_DIR = BASE_DIR / "outputs"


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "llama-3.1-8b-instant"
)


def load_prompt():
    """
    Load the system prompt from prompts/ticket_analysis.txt.
    """

    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


def create_client():
    """
    Create Groq client.
    """

    if not API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to the .env file."
        )

    return Groq(api_key=API_KEY)


def analyze_ticket(client, system_prompt, ticket):
    """
    Send one ticket to the LLM and request structured JSON.
    """

    customer_message = str(
        ticket.get("customer_message", "")
    )

    agent_notes = str(
        ticket.get("agent_notes", "")
    )

    ticket_id = str(
        ticket.get("ticket_id", "")
    )

    user_prompt = f"""
Analyze this support ticket.

Ticket ID:
{ticket_id}

Category from dataset:
{ticket.get("category", "")}

Channel:
{ticket.get("channel", "")}

Priority:
{ticket.get("priority", "")}

Customer message:
{customer_message}

Agent notes:
{agent_notes}

Return only the requested JSON object.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    return json.loads(content)


def analyze_sample(df, sample_size=30):
    """
    Analyze a small sample of tickets.

    We start with 30 tickets to keep API usage
    and cost controlled during development.
    """

    client = create_client()
    system_prompt = load_prompt()

    # Reproducible sample
    sample = df.sample(
        n=min(sample_size, len(df)),
        random_state=42
    ).copy()

    results = []

    print(
        f"\nRunning AI analysis on "
        f"{len(sample)} tickets..."
    )

    for index, (_, ticket) in enumerate(
        sample.iterrows(),
        start=1
    ):

        ticket_id = ticket["ticket_id"]

        print(
            f"Analyzing {index}/{len(sample)}: "
            f"{ticket_id}"
        )

        try:

            result = analyze_ticket(
                client,
                system_prompt,
                ticket
            )

            results.append(result)

        except Exception as error:

            print(
                f"ERROR for {ticket_id}: "
                f"{error}"
            )

            results.append(
                {
                    "ticket_id": ticket_id,
                    "theme": "ERROR",
                    "issue_summary": "",
                    "root_cause": "",
                    "recommended_action": "",
                    "confidence": "low",
                    "evidence": str(error),
                }
            )

    return pd.DataFrame(results)


def save_results(results):
    """
    Save AI results.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "ai_ticket_analysis.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nAI results saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("VIREO AI TICKET ANALYSIS")
    print("=" * 60)

    # Load original data
    datasets = load_data()

    # Clean tickets
    tickets = clean_tickets(
        datasets["tickets"]
    )

    # Analyze 30 tickets first
    results = analyze_sample(
        tickets,
        sample_size=30
    )

    # Display results
    print("\nAI ANALYSIS PREVIEW")
    print("=" * 60)

    print(
        results[
            [
                "ticket_id",
                "theme",
                "root_cause",
                "confidence",
            ]
        ].to_string(index=False)
    )

    # Save
    save_results(results)

    print(
        "\nSUCCESS: AI analysis completed."
    )