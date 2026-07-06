# ai-transition

A 16-week Data Analyst → AI engineering transition project.
Building toward a flagship **User Insight Copilot** (LLM classification + RAG + agent + deployment).

## Week 1 — build a tiny LLM app while refreshing Python

**Status: complete ✅**

| Day | File | Goal |
|-----|------|------|
| Mon 7/7 | `src/hello.py` | Environment + first LLM call ✅ |
| Tue 7/8 | `src/load_feedback.py` | Load + clean a real feedback dataset ✅ |
| Wed 7/9 | `src/models.py` | Pydantic models for structured LLM output ✅ |
| Thu 7/10 | `src/llm_client.py` | Reusable LLM client (retry + cost logging) ✅ |
| Fri 7/11 | `src/pipeline_demo.py` | End-to-end: data → LLM → parsed model → print ✅ |

**What's in place:**

- **Environment**: `uv`-managed virtualenv, `.env` for the Anthropic API key, dependencies pinned in `pyproject.toml`.
- **Data cleaning**: `load_feedback.py` collapses a raw social-media sentiment dataset into 3-class labels (`positive` / `negative` / `neutral`) and writes `data/feedback_clean.parquet`.
- **Pydantic models**: `models.py` defines `FeedbackLabel`, `FeedbackBatch`, and `WeeklyInsight` for structured LLM output, with field-level validation (sentiment enum, confidence range) and cross-field validation (`FeedbackBatch.total` must match `len(items)`).
- **LLM client**: `llm_client.py` wraps the Anthropic SDK with exponential-backoff retries (rate limits, server errors, connection issues) and per-call cost logging, so callers don't have to reimplement retry logic.
- **End-to-end demo**: `pipeline_demo.py` samples 5 feedback rows, labels each with `call_llm`, validates the JSON reply against `FeedbackLabel`, skips and logs any row that fails validation instead of crashing, and assembles the successes into a `FeedbackBatch`.

## Setup (Day 1)

```bash
# 1. install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. install dependencies into a local .venv
uv sync

# 3. add your key
cp .env.example .env      # then paste your real key into .env

# 4. run the smoke test
uv run src/hello.py
```

Success = you see `✅ LLM replied:` followed by a sentence.
