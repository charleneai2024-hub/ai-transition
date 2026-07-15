"""Prompt templates for LLM-based feedback labeling.

PROMPT_TEMPLATE_V2 tightens the original PROMPT_TEMPLATE (see
try_structured.py) with: XML tags separating the feedback text from
instructions, few-shot examples, a constrained topic vocabulary, and
confidence anchors.

PROMPT_TEMPLATE_V3 fixes two problems found by comparing v1 vs v2 on
data/feedback_clean.parquet: (1) v2's topic candidates were an app-feedback
vocabulary (stability, performance, ui, ...) that doesn't match this
dataset's actual content (social media posts about everyday life), so
almost everything fell into the "other" bucket; (2) v2's confidence
anchoring made the model retreat to "neutral" too readily on merely
indirect/figurative language. v3 replaces the topic vocabulary with
categories derived from actually reading 30 sampled rows, adds 2 more
few-shot examples spanning different topics so the model doesn't just
imitate the examples' categories, makes "other" an explicit last resort,
and rewrites the sentiment anchor so indirect language still gets a
directional (not neutral) label, at lower confidence.
"""

PROMPT_TEMPLATE_V2 = """Analyze the sentiment and topic of the user feedback in the <feedback> tags below.

Respond with ONLY a JSON object (no markdown fences, no extra text) with
exactly these fields:
- "sentiment": one of "positive", "negative", "neutral"
- "topic": 1-2 lowercase English words. Prefer one of these candidates if it
  fits: stability, performance, ui, content, sharing, notifications, account,
  other.
- "confidence": a number between 0 and 1. Anchor your score: 0.9 or above
  means the sentiment is clear and unambiguous; 0.5-0.7 means the sentiment
  is mixed, sarcastic, or vague.

Example 1:
<feedback>Crashes every time I open the app, completely unusable now.</feedback>
{{"sentiment": "negative", "topic": "stability", "confidence": 0.95}}

Example 2:
<feedback>Love the new dashboard, so much faster and cleaner than before!</feedback>
{{"sentiment": "positive", "topic": "performance", "confidence": 0.92}}

Now analyze this feedback:
<feedback>{text}</feedback>
"""

PROMPT_TEMPLATE_V3 = """Analyze the sentiment and topic of the user feedback in the <feedback> tags below.
This feedback comes from social media posts about everyday life, not app or product reviews.

Respond with ONLY a JSON object (no markdown fences, no extra text) with
exactly these fields:

- "sentiment": one of "positive", "negative", "neutral".
  Anchor your judgment: if the sentiment direction is clear even when it is
  expressed indirectly or through metaphor, still choose "positive" or
  "negative" and use a lower confidence to reflect that indirectness. Only
  use "neutral" when the feedback is genuinely mixed (both positive and
  negative) or carries no emotional content at all.

- "topic": 1-2 lowercase English words, chosen from this list:
  - relationships: connection with family, friends, or partners - love,
    trust, closeness, or its loss (loneliness, betrayal)
  - social_event: a gathering, party, concert, celebration, or festival
    shared with other people
  - nature: outdoor scenery, wildlife, weather, or natural landscapes
  - achievement: a personal milestone, competition, record, or
    accomplishment
  - daily_life: an ordinary routine, food, home, or small everyday
    pleasure
  - emotion: a standalone mood or feeling described with no specific
    event or activity attached
  - other: use ONLY when the feedback does not fit any category above

- "confidence": a number between 0 and 1, consistent with the sentiment
  anchoring described above.

Example 1:
<feedback>Betrayal, a venomous serpent slithering through the garden of trust, poisoning roots.</feedback>
{{"sentiment": "negative", "topic": "relationships", "confidence": 0.75}}

Example 2:
<feedback>Drenched in confetti at a concert, a kaleidoscope of colors igniting the night sky.</feedback>
{{"sentiment": "positive", "topic": "social_event", "confidence": 0.8}}

Example 3:
<feedback>In the midst of a soccer match, an unexpected own goal creates a moment of despair for the player.</feedback>
{{"sentiment": "negative", "topic": "achievement", "confidence": 0.85}}

Example 4:
<feedback>In the midst of the Amazon rainforest, a symphony of wildlife creates an orchestra of nature's wonders.</feedback>
{{"sentiment": "neutral", "topic": "nature", "confidence": 0.9}}

Now analyze this feedback:
<feedback>{text}</feedback>
"""
