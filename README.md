# Vireo Audio — Support Intelligence

AI-assisted support analytics tool built for the Vireo Audio support-ticket assignment.

## What it does

The tool:

- Loads support tickets and reference datasets.
- Identifies duplicate ticket IDs across source systems.
- Cleans and normalizes ticket data.
- Calculates deterministic support KPIs.
- Calculates an operational repeat-contact proxy.
- Produces category-level support metrics.
- Produces a Tier-1 agent leaderboard.
- Uses Groq LLM analysis to identify recurring ticket themes.
- Produces a weekly support digest.
- Provides a Streamlit dashboard.

## Project structure

```text
vireo-support-intelligence/
├── data/
├── evaluation/
├── outputs/
├── prompts/
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── metrics.py
│   ├── ai_analysis.py
│   ├── evaluate_ai.py
│   ├── weekly_digest.py
│   └── app.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md