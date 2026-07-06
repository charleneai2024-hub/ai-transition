"""End-to-end smoke test: real feedback -> Claude -> FeedbackLabel.

Loads one row of cleaned feedback, asks Claude to label it as JSON, and
parses the response with FeedbackLabel so bad LLM output is caught by
Pydantic instead of silently flowing downstream.
"""

import json

import pandas as pd

from llm_client import call_llm
from models import FeedbackLabel

PROMPT_TEMPLATE = """Analyze the sentiment and topic of this piece of user feedback.

Feedback: "{text}"

Respond with ONLY a JSON object (no markdown fences, no extra text) with
exactly these fields:
- "sentiment": one of "positive", "negative", "neutral"
- "topic": a short topic label (e.g. "performance", "ui", "pricing")
- "confidence": a number between 0 and 1
"""


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` wrapping in case the model adds it anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    return text


def main() -> None:
    df = pd.read_parquet("data/feedback_clean.parquet")
    feedback_text = df.iloc[0]["Text"]
    print("Feedback text:", feedback_text)

    raw_text = strip_code_fences(call_llm(PROMPT_TEMPLATE.format(text=feedback_text), max_tokens=200))
    print("\nRaw LLM output:", raw_text)

    parsed_json = json.loads(raw_text)
    label = FeedbackLabel.model_validate(parsed_json)

    print("\nParsed FeedbackLabel:", label)


if __name__ == "__main__":
    main()
