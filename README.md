# ai-transition

A 16-week Data Analyst → AI engineering transition project.
Building toward a flagship **User Insight Copilot** (LLM classification + RAG + agent + deployment).

## Week 1 — build a tiny LLM app while refreshing Python

| Day | File | Goal |
|-----|------|------|
| Mon 7/7 | `src/hello.py` | Environment + first LLM call ✅ (done in this scaffold) |
| Tue 7/8 | `src/load_feedback.py` | Load + clean a real feedback dataset |
| Wed 7/9 | `src/models.py` | Pydantic models for structured LLM output |
| Thu 7/10 | `src/llm_client.py` | Reusable LLM client (retry + cost logging) |
| Fri 7/11 | end-to-end | data → LLM → parsed model → print |

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
