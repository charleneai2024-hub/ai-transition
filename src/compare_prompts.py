"""Compare PROMPT_TEMPLATE (v1) vs PROMPT_TEMPLATE_V2 on the same feedback sample.

Runs both prompt versions through call_llm on the same 10 sampled feedback
rows, prints results side by side, and reports how many distinct topic
values each version produces.
"""

import json

import pandas as pd
from pydantic import ValidationError

from llm_client import call_llm
from models import FeedbackLabel
from prompts import PROMPT_TEMPLATE_V2
from try_structured import PROMPT_TEMPLATE, strip_code_fences

SAMPLE_SIZE = 10
RANDOM_STATE = 42


def label_with(prompt_template: str, text: str) -> FeedbackLabel | None:
    """Run one prompt template on one feedback text; None if the reply doesn't validate."""
    try:
        raw_text = strip_code_fences(call_llm(prompt_template.format(text=text)))
        return FeedbackLabel.model_validate(json.loads(raw_text))
    except (json.JSONDecodeError, ValidationError) as error:
        print(f"[warning] labeling failed: {error}")
        return None


def format_label(label: FeedbackLabel | None) -> str:
    if label is None:
        return "FAILED"
    return f"{label.sentiment}/{label.topic}/{label.confidence}"


def main() -> None:
    df = pd.read_parquet("data/feedback_clean.parquet")
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    v1_topics: list[str] = []
    v2_topics: list[str] = []

    print(f"{'feedback preview':<54} | {'v1 (sentiment/topic/conf)':<36} | v2 (sentiment/topic/conf)")
    print("-" * 130)

    for text in sample["Text"]:
        label_v1 = label_with(PROMPT_TEMPLATE, text)
        label_v2 = label_with(PROMPT_TEMPLATE_V2, text)

        if label_v1 is not None:
            v1_topics.append(label_v1.topic)
        if label_v2 is not None:
            v2_topics.append(label_v2.topic)

        preview = repr(text[:50])
        print(f"{preview:<54} | {format_label(label_v1):<36} | {format_label(label_v2)}")

    print()
    print(f"v1 distinct topics ({len(set(v1_topics))}): {sorted(set(v1_topics))}")
    print(f"v2 distinct topics ({len(set(v2_topics))}): {sorted(set(v2_topics))}")


if __name__ == "__main__":
    main()
