"""Day 1 deliverable: confirm the environment works by making one LLM call."""

import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("❌ ANTHROPIC_API_KEY not found. Copy .env.example to .env and paste your key.")

client = Anthropic()  # picks up the key from the environment automatically

msg = client.messages.create(
    model="claude-haiku-4-5",   # cheapest model — perfect for a smoke test
    max_tokens=100,
    messages=[
        {"role": "user", "content": "用一句话确认你能收到我的消息。"}
    ],
)

# The response is a list of content blocks; text lives in .content[0].text
print("✅ LLM replied:", msg.content[0].text)
print("📊 tokens — in:", msg.usage.input_tokens, "| out:", msg.usage.output_tokens)
