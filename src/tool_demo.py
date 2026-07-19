"""Minimal tool-calling demo.

Claude decides when it needs data, asks us to run a local function
(get_sentiment_stats), we execute it against feedback_clean.parquet, and
feed the result back until Claude produces a final answer.
"""

import json

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-8"
DATA_PATH = "data/feedback_clean.parquet"

client = anthropic.Anthropic()


def get_sentiment_stats(platform: str) -> dict:
    """Real local function: feedback count and sentiment shares for one platform."""
    df = pd.read_parquet(DATA_PATH)
    subset = df[df["Platform"] == platform]
    count = len(subset)
    if count == 0:
        return {"platform": platform, "count": 0, "positive_pct": 0.0, "negative_pct": 0.0, "neutral_pct": 0.0}

    shares = subset["sentiment_3"].value_counts(normalize=True)
    return {
        "platform": platform,
        "count": count,
        "positive_pct": round(float(shares.get("positive", 0.0)) * 100, 1),
        "negative_pct": round(float(shares.get("negative", 0.0)) * 100, 1),
        "neutral_pct": round(float(shares.get("neutral", 0.0)) * 100, 1),
    }


# Tool registration: name/description/input_schema tell Claude when and how
# to call get_sentiment_stats. The enum restricts platform to valid values.
TOOLS = [
    {
        "name": "get_sentiment_stats",
        "description": (
            "Get the feedback count and positive/negative/neutral sentiment "
            "shares (as percentages) for one social media platform."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["Twitter", "Facebook", "Instagram"],
                    "description": "The social media platform to get sentiment stats for.",
                }
            },
            "required": ["platform"],
        },
    }
]


def run_tool(name: str, tool_input: dict) -> dict:
    if name == "get_sentiment_stats":
        return get_sentiment_stats(**tool_input)
    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    user_question = "Twitter/Facebook/Instagram三个平台，哪个平台负面反馈占比最高。"
    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
        for block in tool_use_blocks:
            print(f"[tool_use] Claude called {block.name}({block.input})")

        if response.stop_reason != "tool_use":
            final_text = next(block.text for block in response.content if block.type == "text")
            print(f"\n[final answer]\n{final_text}")
            break

        # Claude's turn (including its tool_use blocks) must go back before the results.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            result = run_tool(block.name, block.input)
            print(f"[tool_result] {block.name} -> {result}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    main()
