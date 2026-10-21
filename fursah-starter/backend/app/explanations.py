"""Deterministic source excerpts; no model-generated identifiers or evidence."""
from .infrastructure.errors import EvidenceValidationError
from .internal import EvidenceExplanation, ExplanationBatch, OpportunityRecord, SearchPlan
from .repository import contains


def build_explanations(plan: SearchPlan, records: list[OpportunityRecord]) -> ExplanationBatch:
    explanations = []
    for record in records:
        # Overlapping windows preserve the original characters, including Arabic
        # and newlines. Normalized search offsets must never index raw evidence.
        passages = [
            source.content[start:start + 160].strip()
            for source in record.sources
            for start in range(0, len(source.content), 80)
            if len(source.content[start:start + 160].strip()) >= 8
        ]
        if not passages:
            raise EvidenceValidationError()
        # Prefer query subject evidence; stable source/window order breaks ties.
        quote = max(passages, key=lambda text: sum(contains(text, term) for term in plan.keywords))
        explanations.append(EvidenceExplanation(opportunity_id=record.opportunity_id, quotes=[quote]))
    return ExplanationBatch(explanations=explanations)
