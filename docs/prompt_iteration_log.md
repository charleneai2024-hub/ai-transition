# Prompt Iteration Log — Feedback Sentiment/Topic Labeling

Working notes on iterating the LLM prompt used to label social media feedback
with `sentiment` / `topic` / `confidence` (see `src/models.py::FeedbackLabel`).
Kept as a record of *what broke, what fixed it, and what it exposed next* —
this is the source material for talking through real prompt engineering
practice, not just the final prompt.

**Method**: every version was compared against the previous one on the same
fixed sample of 10 rows from `data/feedback_clean.parquet` (`random_state=42`),
so differences in output are attributable to the prompt change, not to which
rows got sampled.

## V1 → V2

**V1** (`src/try_structured.py::PROMPT_TEMPLATE`): a plain instruction with no
separation between instructions and input, no examples, a free-text `topic`
field, and no guidance on how to set `confidence`.

**What V1 exposed** (`src/compare_prompts.py`): on 10 rows, the model
produced **10 distinct topic strings** — a new invented phrase almost every
time (`athletic performance`, `digital art hobby`, `sleep/anxiety`,
`gaming_tournament`, ...), with inconsistent formatting (spaces, underscores,
slashes). An unconstrained topic field is not usable for downstream
aggregation (e.g. "top topics this week").

**What V2 changed** (`src/prompts.py::PROMPT_TEMPLATE_V2`):
1. Wrapped the feedback text in `<feedback>` tags to separate it from the instructions
2. Added 2 few-shot examples (1 negative, 1 positive) with full JSON output
3. Constrained `topic` to 1-2 lowercase words from a fixed candidate list
4. Added confidence anchors: 0.9+ = clear sentiment, 0.5-0.7 = mixed/vague

**What V2 exposed** (`src/compare_prompts.py`, same 10 rows):
- **Domain mismatch**: the candidate topic list (`stability`, `performance`,
  `ui`, `notifications`, `account`, ...) was written for *app/product
  feedback*. The actual dataset is social-media lifestyle posts. Result:
  topic collapsed to just **2 values** (`content`, `other`) — everything that
  didn't fit the app vocabulary fell into the catch-all.
- **Over-conservative sentiment**: 4 of 10 rows that had a clear directional
  sentiment in V1 flipped to `neutral` in V2 (e.g. "A soul weathered by the
  storm of heartbreak..." went negative → neutral). The "0.5–0.7 = mixed or
  vague" anchor language pushed the model to hedge on anything indirect or
  figurative, not just genuinely ambiguous cases.

## V2 → V3

**Diagnosis first**: before writing V3, manually read 30 randomly sampled
rows (`random_state=7`) from `feedback_clean.parquet` to build a topic
taxonomy grounded in what the data actually contains, instead of guessing
again.

**What V3 changed** (`src/prompts.py::PROMPT_TEMPLATE_V3`):
1. Replaced the topic vocabulary with 6 categories derived from the 30-row
   read (`relationships`, `social_event`, `nature`, `achievement`,
   `daily_life`, `emotion`) + `other`, each with a one-line definition
2. Expanded few-shot examples from 2 to 4, deliberately spanning 4 different
   topics so the model doesn't just imitate whichever categories happened to
   appear in the examples
3. Made `other` an explicit last resort: "use ONLY when the feedback does
   not fit any category above"
4. Rewrote the sentiment anchor: indirect/figurative language with a clear
   sentiment direction should still get `positive`/`negative` (at a lower
   confidence to reflect the indirectness); `neutral` is reserved for
   genuinely mixed or emotionless content

**What V3 fixed** (`src/compare_prompts_v2_v3.py`, same 10 rows):
- Distinct topics used: 2 → 4 (`content`/`other` → `achievement`/`emotion`/
  `relationships`/`social_event`)
- Neutral rate on the fixed 10 rows: 40% (V2) → 0% (V3); mean confidence
  rose from ~0.73 to ~0.85 — the model stopped retreating to neutral on
  indirect language.

**What V3 exposed** (new problem, not yet fixed):
- **`achievement` became a new soft catch-all**: 6 of 10 rows were tagged
  `achievement`, including cases like "exploring digital art" and "attending
  a workshop" that are closer to hobby/interest exploration than an actual
  milestone or competition. The taxonomy is missing a `hobby`/personal-
  interest category, and `achievement`'s definition needs to require
  explicit milestone/competition/record language to stop absorbing anything
  vaguely accomplishment-shaped.
- **Confidence calibration for "indirectness" isn't fully consistent** — one
  metaphor-heavy row ("fearful shadows dancing") got a *higher* confidence
  (0.85) than a comparably indirect row ("storm of heartbreak", 0.75). The
  anchor moves the model in the right direction but isn't a precise dial.

## Next step: build a real eval set

A fixed 10-row sample was enough to catch obvious failure modes cheaply
(topic collapse, over-conservative sentiment), but it can't:
- distinguish a real prompt effect from single-row model noise
- measure topic *accuracy*, since there's no ground-truth topic label to
  compare against — only self-consistency between prompt versions has been
  checked so far, not correctness
- reliably characterize how well the sentiment anchor generalizes across
  more indirect/figurative language than 10 rows happen to contain

Planned next iteration:
1. Hand-label a held-out eval set (50–100 rows) with the *correct* topic
   (against the finalized taxonomy) and sentiment, independent of any
   prompt's output
2. Score each prompt version against that ground truth (per-class
   precision/recall/F1 for topic, accuracy for sentiment) instead of only
   diffing versions against each other
3. Specifically stress-test the two open V3 issues — `achievement`
   over-tagging and confidence calibration on indirect language — with rows
   chosen to probe those failure modes
4. Turn this into a repeatable eval script (not an ad-hoc compare script),
   so future prompt edits can be measured against a fixed baseline instead
   of eyeballing a 10-row diff

## File map

| File | Role |
|---|---|
| `src/try_structured.py` | `PROMPT_TEMPLATE` (v1) + smoke test |
| `src/prompts.py` | `PROMPT_TEMPLATE_V2`, `PROMPT_TEMPLATE_V3` |
| `src/compare_prompts.py` | v1 vs v2 comparison on 10 fixed rows |
| `src/compare_prompts_v2_v3.py` | v2 vs v3 comparison on the same 10 rows |
