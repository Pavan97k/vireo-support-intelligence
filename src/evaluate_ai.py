from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

AI_FILE = BASE_DIR / "outputs" / "ai_ticket_analysis.csv"
EVALUATION_DIR = BASE_DIR / "evaluation"


def load_ai_results():
    if not AI_FILE.exists():
        raise FileNotFoundError(
            "AI analysis file not found.\n"
            "Run: python src/ai_analysis.py"
        )

    return pd.read_csv(AI_FILE)


def create_review_file(df):
    """
    Create a human-review template.

    The AI prediction is kept separate from the
    human judgement so we don't treat the AI output
    as ground truth.
    """

    review = df[
        [
            "ticket_id",
            "theme",
            "issue_summary",
            "root_cause",
            "recommended_action",
            "confidence",
            "evidence",
        ]
    ].copy()

    review["human_theme"] = ""
    review["theme_correct"] = ""
    review["root_cause_correct"] = ""
    review["evidence_supported"] = ""
    review["human_notes"] = ""

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        EVALUATION_DIR
        / "ai_review_sample.csv"
    )

    review.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nReview file created:\n{output_file}"
    )

    print("\nHuman review instructions:")
    print("- human_theme: enter the theme you believe is correct")
    print("- theme_correct: Yes / No")
    print("- root_cause_correct: Yes / No / NA")
    print("- evidence_supported: Yes / No")
    print("- human_notes: explain any important error")


def show_summary(df):
    print("\n")
    print("=" * 60)
    print("AI EVALUATION SAMPLE")
    print("=" * 60)

    print(
        f"Tickets available for review: {len(df)}"
    )

    print(
        "\nAI theme distribution:"
    )

    print(
        df["theme"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":

    results = load_ai_results()

    show_summary(results)

    create_review_file(results)