# ai-transition

A 16-week Data Analyst → AI engineering transition project.
Building toward a flagship **User Insight Copilot** (LLM classification + RAG + agent + deployment).

## Data

This project uses two datasets:

1. **Social media sentiment data** — already cleaned and included in the repo at `data/feedback_clean.parquet` (see `src/load_feedback.py` for the cleaning pipeline).
2. **AI4I 2020 Predictive Maintenance Dataset** — not included in the repo. Download it from [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) or the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), and place the CSV at `data/Predictive Maintenance Dataset (AI4I 2020).csv`. See `src/inspect_machine_data.py` for an initial data-quality check.

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

## Day 2 (Tue 7/8) — load + clean feedback data

```bash
uv run src/load_feedback.py
```

Reads `data/sentimentdataset.csv`, collapses the raw fine-grained emotion
labels into 3 classes (`positive` / `negative` / `neutral`), and writes the
cleaned dataset to `data/feedback_clean.parquet`.

Success = the script prints the raw shape, the cleaned shape, and the
3-class distribution, then reports `Saved -> data/feedback_clean.parquet`.

## Day 3 (Wed 7/9) — Pydantic models for structured LLM output

`src/models.py` has no script to run on its own — it defines the schemas
(`FeedbackLabel`, `FeedbackBatch`, `WeeklyInsight`) that every later script
imports. Exercise it directly with a quick check:

```bash
uv run python -c "
from src.models import FeedbackLabel
print(FeedbackLabel(sentiment='positive', topic='ui', confidence=0.9))
"
```

Success = a `FeedbackLabel(...)` instance prints back; passing an invalid
`sentiment` or a `confidence` outside `[0, 1]` raises a `ValidationError`.

## Day 4 (Thu 7/10) — reusable LLM client with retry

```bash
uv run src/try_structured.py
```

Pulls one row of real feedback, calls `call_llm` (from `src/llm_client.py`)
to label it, and parses the JSON reply into a `FeedbackLabel`.

Success = you see the feedback text, the raw LLM JSON, the parsed
`FeedbackLabel(...)`, and a `[llm_client] ... cost=$...` line showing token
usage and estimated cost.

## Day 5 (Fri 7/11) — end-to-end pipeline

```bash
uv run src/pipeline_demo.py
```

Samples 5 feedback rows (fixed `random_state` for reproducibility), labels
each one through `call_llm`, validates every reply against `FeedbackLabel`
(skipping and logging any row that fails validation instead of crashing),
and assembles the successes into a `FeedbackBatch`.

Success = a `--- Results ---` block listing each feedback preview with its
sentiment/topic/confidence, followed by a `Succeeded: N, Failed: M` summary.
