"""Reusable LLM client wrapper: retry with exponential backoff + cost logging."""

import random
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

# USD per 1M tokens: (input_price, output_price)
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0

# max_retries=0: we own the retry loop below instead of the SDK's built-in one.
_client = anthropic.Anthropic(max_retries=0)


def _is_retryable(error: anthropic.APIStatusError) -> bool:
    """Rate limits and server errors are transient; everything else (bad request, auth, permission, not found) will fail the same way every time."""
    return isinstance(error, (anthropic.RateLimitError, anthropic.InternalServerError))


def _log_cost(model: str, usage) -> None:
    input_price, output_price = MODEL_PRICING.get(model, (0.0, 0.0))
    cost = (usage.input_tokens / 1_000_000) * input_price + (usage.output_tokens / 1_000_000) * output_price
    print(
        f"[llm_client] model={model} input_tokens={usage.input_tokens} "
        f"output_tokens={usage.output_tokens} cost=${cost:.6f}"
    )


def call_llm(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> str:
    """Send one prompt to Claude and return the text response.

    Retries rate limits and server errors (429, 5xx) with exponential backoff,
    up to MAX_RETRIES attempts. Connection errors (including timeouts) are
    also retried, since they're transient by nature. Authentication,
    bad-request, and permission errors are raised immediately — retrying
    won't change the outcome.
    """
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = _client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            _log_cost(model, response.usage)
            return response.content[0].text
        except anthropic.APIStatusError as error:
            if not _is_retryable(error):
                raise
            last_error = error
        except anthropic.APIConnectionError as error:
            last_error = error

        delay = BASE_DELAY_SECONDS * (2**attempt) + random.uniform(0, 1)
        print(
            f"[llm_client] retryable error ({last_error}); "
            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
        )
        time.sleep(delay)

    raise last_error
