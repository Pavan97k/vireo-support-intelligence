from pathlib import Path
import json
import os
import time

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
    "openai/gpt-oss-20b"
)


# ---------------------------------------------------------
# Load system prompt
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Create Groq client
# ---------------------------------------------------------

def create_client():
    """
    Create and return Groq client.
    """

    if not API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to the .env file."
        )

    return Groq(api_key=API_KEY)


# ---------------------------------------------------------
# Analyze one ticket
# ---------------------------------------------------------

def analyze_ticket(
    client,
    system_prompt,
    ticket,
    max_retries=5
):
    """
    Send one ticket to the LLM and request structured JSON.

    Temporary API rate-limit errors and invalid responses
    are retried automatically.
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

Return ONLY one valid JSON object.

Do not return markdown.
Do not return ```json.
Do not add explanations outside the JSON object.

The JSON must contain:
- theme
- issue_summary
- root_cause
- recommended_action
- confidence
- evidence
"""

    last_error = None

    # -----------------------------------------------------
    # Retry loop
    # -----------------------------------------------------

    for attempt in range(1, max_retries + 1):

        try:

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
                response_format={
                    "type": "json_object"
                },
            )

            # -------------------------------------------------
            # Extract model response
            # -------------------------------------------------

            content = response.choices[0].message.content

            if not content:
                raise ValueError(
                    "The model returned an empty response."
                )

            content = content.strip()

            # -------------------------------------------------
            # Remove markdown fences if returned
            # -------------------------------------------------

            if content.startswith("```"):

                if content.startswith("```json"):
                    content = content[
                        len("```json"):
                    ]

                elif content.startswith("```"):
                    content = content[
                        len("```"):
                    ]

                if content.endswith("```"):
                    content = content[:-3]

                content = content.strip()

            # -------------------------------------------------
            # Parse JSON
            # -------------------------------------------------

            result = json.loads(content)

            # -------------------------------------------------
            # Validate result
            # -------------------------------------------------

            if not isinstance(result, dict):
                raise ValueError(
                    "Model response is not a JSON object."
                )

            # -------------------------------------------------
            # Ensure required fields exist
            # -------------------------------------------------

            required_fields = [
                "theme",
                "issue_summary",
                "root_cause",
                "recommended_action",
                "confidence",
                "evidence",
            ]

            for field in required_fields:

                if field not in result:
                    result[field] = ""

            # -------------------------------------------------
            # Always use original ticket ID
            # -------------------------------------------------

            result["ticket_id"] = ticket_id

            return result

        except Exception as error:

            last_error = error

            error_text = str(error).lower()

            # -------------------------------------------------
            # Detect rate-limit / 429 errors
            # -------------------------------------------------

            is_rate_limit = (
                "429" in error_text
                or "rate_limit" in error_text
                or "rate limit" in error_text
                or "tokens per minute" in error_text
                or "tpm" in error_text
            )

            # -------------------------------------------------
            # Detect JSON errors
            # -------------------------------------------------

            is_json_error = (
                "json" in error_text
                or "expecting value" in error_text
                or "decode" in error_text
            )

            # -------------------------------------------------
            # Retry temporary problems
            # -------------------------------------------------

            if is_rate_limit:

                # Progressive wait:
                # 4, 7, 10, 13, 16 seconds
                wait_time = 4 + (
                    (attempt - 1) * 3
                )

                print(
                    f"Rate limit for {ticket_id}. "
                    f"Retry {attempt}/{max_retries} "
                    f"in {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            if is_json_error:

                wait_time = 2 + attempt

                print(
                    f"Invalid JSON for {ticket_id}. "
                    f"Retry {attempt}/{max_retries} "
                    f"in {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            # -------------------------------------------------
            # Retry other temporary API failures
            # -------------------------------------------------

            if attempt < max_retries:

                wait_time = 2 + attempt

                print(
                    f"Temporary error for {ticket_id}: "
                    f"{error}"
                )

                print(
                    f"Retry {attempt}/{max_retries} "
                    f"in {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            # -------------------------------------------------
            # No more retries
            # -------------------------------------------------

            break

    # ---------------------------------------------------------
    # All retries failed
    # ---------------------------------------------------------

    raise RuntimeError(
        f"AI analysis failed for {ticket_id} "
        f"after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------
# Analyze sample
# ---------------------------------------------------------

def analyze_sample(
    df,
    sample_size=30
):
    """
    Analyze a reproducible sample of tickets.

    We start with 30 tickets to keep API usage
    and cost controlled during development.
    """

    client = create_client()

    system_prompt = load_prompt()

    # -----------------------------------------------------
    # Reproducible sample
    # -----------------------------------------------------

    sample = df.sample(
        n=min(sample_size, len(df)),
        random_state=42
    ).copy()

    results = []

    print(
        f"\nRunning AI analysis on "
        f"{len(sample)} tickets..."
    )

    # -----------------------------------------------------
    # Analyze each ticket
    # -----------------------------------------------------

    for index, (_, ticket) in enumerate(
        sample.iterrows(),
        start=1
    ):

        ticket_id = str(
            ticket["ticket_id"]
        )

        print(
            f"Analyzing {index}/{len(sample)}: "
            f"{ticket_id}"
        )

        try:

            result = analyze_ticket(
                client,
                system_prompt,
                ticket,
                max_retries=5
            )

            results.append(result)

            # -------------------------------------------------
            # Small delay between successful requests
            # This reduces TPM pressure.
            # -------------------------------------------------

            time.sleep(1.5)

        except Exception as error:

            print(
                f"FINAL FAILURE for {ticket_id}: "
                f"{error}"
            )

            # -------------------------------------------------
            # Do NOT invent an AI result.
            # Mark as ABSTAIN if all retries fail.
            # -------------------------------------------------

            results.append(
                {
                    "ticket_id": ticket_id,
                    "theme": "ABSTAIN",
                    "issue_summary": "",
                    "root_cause": "",
                    "recommended_action": "",
                    "confidence": "low",
                    "evidence": (
                        "AI analysis could not be completed "
                        "after automatic retries. "
                        f"Reason: {error}"
                    ),
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

def save_results(results):
    """
    Save AI results to outputs/ai_ticket_analysis.csv.
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("VIREO AI TICKET ANALYSIS")
    print("=" * 60)

    print(
        f"\nUsing model: {MODEL_NAME}"
    )

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    datasets = load_data()

    # -----------------------------------------------------
    # Clean tickets
    # -----------------------------------------------------

    tickets = clean_tickets(
        datasets["tickets"]
    )

    # -----------------------------------------------------
    # Analyze 30 tickets
    # -----------------------------------------------------

    results = analyze_sample(
        tickets,
        sample_size=30
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nAI ANALYSIS PREVIEW")
    print("=" * 60)

    preview_columns = [
        "ticket_id",
        "theme",
        "root_cause",
        "confidence",
    ]

    available_columns = [
        column
        for column in preview_columns
        if column in results.columns
    ]

    print(
        results[
            available_columns
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Count failures
    # -----------------------------------------------------

    if "theme" in results.columns:

        error_count = (
            results["theme"]
            .astype(str)
            .str.upper()
            .eq("ERROR")
            .sum()
        )

        abstain_count = (
            results["theme"]
            .astype(str)
            .str.upper()
            .eq("ABSTAIN")
            .sum()
        )

        print(
            f"\nAI ERROR COUNT: {error_count}"
        )

        print(
            f"AI ABSTAIN COUNT: {abstain_count}"
        )

        print(
            f"AI SUCCESS COUNT: "
            f"{len(results) - error_count - abstain_count}"
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    save_results(results)

    print(
        "\nSUCCESS: AI analysis completed."
    )