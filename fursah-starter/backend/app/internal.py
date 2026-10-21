"""Small validated models used by the single-agent recommendation workflow."""
from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .schemas import NonBlank, OpportunityDetailsResponse, SearchConstraints


class SourceEvidence(BaseModel):
    source_id: NonBlank
    source_name: NonBlank
    published_at: datetime
    content: str = Field(min_length=1)

    @field_validator('content')
    @classmethod
    def retain_source_text(cls, value):
        if not value.strip():
            raise ValueError('Source content must not be blank')
        return value  # Preserve original whitespace as evidence.


class OpportunityRecord(OpportunityDetailsResponse):
    opportunity_type: Literal['workshop', 'course', 'internship', 'competition', 'volunteering', 'event'] | None = None
    delivery_mode: Literal['in_person', 'online', 'hybrid'] | None = None
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sources: list[SourceEvidence] = Field(min_length=1)
    # Shared links are related evidence, not an instruction to merge identities.
    related_sources: list[SourceEvidence] = Field(default_factory=list)
    conflicting_fields: list[str] = Field(default_factory=list)

    @field_validator('start_date', 'end_date')
    @classmethod
    def valid_date(cls, value):
        if value is not None:
            date.fromisoformat(value)
        return value

    @field_validator('deadline')
    @classmethod
    def valid_deadline(cls, value):
        if value is not None:
            if len(value) == 10:
                date.fromisoformat(value)
            elif datetime.fromisoformat(value.replace('Z', '+00:00')).tzinfo is None:
                raise ValueError('Exact deadline needs a timezone')
        return value

    @model_validator(mode='after')
    def source_identity(self):
        ids = [source.source_id for source in self.sources]
        if len(set(ids)) != len(ids) or set(ids) != set(self.source_ids):
            raise ValueError('Source evidence must match source_ids exactly')
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError('Activity start must precede its end')
        return self


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    # Empty keywords means a deliberately broad opportunity search.
    keywords: list[NonBlank] = Field(default_factory=list, max_length=8)
    excluded_keywords: list[NonBlank] = Field(default_factory=list, max_length=8)
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    excluded_locations: list[NonBlank] = Field(default_factory=list, max_length=10)
    excluded_types: list[Literal['workshop', 'course', 'internship', 'competition', 'volunteering', 'event']] = Field(default_factory=list)
    excluded_modes: list[Literal['in_person', 'online', 'hybrid']] = Field(default_factory=list)


class EvidenceExplanation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    opportunity_id: NonBlank
    # Python selects source passages; server verifies each verbatim.
    quotes: list[Annotated[NonBlank, Field(min_length=8, max_length=160)]] = Field(min_length=1, max_length=1)


class ExplanationBatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    explanations: list[EvidenceExplanation] = Field(min_length=1, max_length=3)


class CandidateCheck(BaseModel):
    accepted: bool
    missing_fields: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    eligibility_status: Literal['confirmed', 'uncertain'] = 'uncertain'
