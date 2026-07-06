"""End-to-end vertical slice: sampled feedback -> LLM -> validated labels -> batch.

Ties together load_feedback.py's output, llm_client.py's retrying call_llm,
and models.py's FeedbackLabel/FeedbackBatch into one pass over real data.
"""

import json

import pandas as pd
from pydantic import ValidationError

from llm_client import call_llm
from models import FeedbackBatch, FeedbackLabel
from try_structured import PROMPT_TEMPLATE, strip_code_fences

SAMPLE_SIZE = 5
RANDOM_STATE = 42


def label_one(text: str) -> FeedbackLabel:
    """Ask the LLM to label one piece of feedback and validate its JSON reply."""
    raw_text = strip_code_fences(call_llm(PROMPT_TEMPLATE.format(text=text), max_tokens=200))
    return FeedbackLabel.model_validate(json.loads(raw_text))


def main() -> None:
    df = pd.read_parquet("data/feedback_clean.parquet")
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    texts: list[str] = []
    labels: list[FeedbackLabel] = []
    failed_count = 0

    for text in sample["Text"]:
        try:
            label = label_one(text)
        except (json.JSONDecodeError, ValidationError) as error:
            failed_count += 1
            print(f"[warning] skipping feedback (validation failed): {error}")
            continue
        texts.append(text)
        labels.append(label)

    # total must equal the number of items that actually made it into the
    # batch, or FeedbackBatch's own model_validator rejects it.
    batch = FeedbackBatch(items=labels, total=len(labels))

    print("\n--- Results ---")
    for text, label in zip(texts, batch.items):
        preview = text[:50]
        print(f"{preview!r} -> sentiment={label.sentiment} topic={label.topic} confidence={label.confidence}")

    print(f"\nSucceeded: {len(batch.items)}, Failed: {failed_count}")


if __name__ == "__main__":
    main()
