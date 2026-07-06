"""Pydantic output models for parsing LLM responses."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FeedbackLabel(BaseModel):
    """Result of analyzing a single piece of user feedback."""

    sentiment: Literal["positive", "negative", "neutral"]
    topic: str = Field(..., min_length=1, description="Short topic label, e.g. 'performance', 'ui', 'pricing'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in this label, from 0 to 1")


class FeedbackBatch(BaseModel):
    """Aggregated labels for a batch of feedback items."""

    items: list[FeedbackLabel]
    total: int = Field(..., ge=0, description="Total number of feedback items in this batch")

    @model_validator(mode="after")
    def check_total_matches_items(self) -> "FeedbackBatch":
        """Reject batches where the reported total disagrees with the actual item count."""
        if self.total != len(self.items):
            raise ValueError(f"total ({self.total}) does not match len(items) ({len(self.items)})")
        return self


class WeeklyInsight(BaseModel):
    """Weekly report insight distilled from a batch of labeled feedback."""

    top_topics: list[str] = Field(..., description="Most frequently mentioned topics, ranked by importance")
    sentiment_breakdown: dict[Literal["positive", "negative", "neutral"], float] = Field(
        ..., description="Share of feedback in each sentiment class, e.g. {'positive': 0.6, 'negative': 0.3, 'neutral': 0.1}"
    )
    summary: str = Field(..., min_length=1, description="One-sentence summary of the week's feedback")

    @field_validator("sentiment_breakdown")
    @classmethod
    def check_shares_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        """Each sentiment share must be a proportion between 0 and 1."""
        for sentiment, share in value.items():
            if not 0.0 <= share <= 1.0:
                raise ValueError(f"share for '{sentiment}' must be between 0 and 1, got {share}")
        return value
