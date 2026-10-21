"""JSON-backed opportunity store. No vector database or runtime extraction."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pydantic import ValidationError
from .internal import OpportunityRecord, SearchPlan
from .schemas import OpportunityDetailsResponse
from .infrastructure.data_package import DatasetLoader
from .infrastructure.errors import DataAccessError


def normalize(value: str) -> str:
    value = ''.join(c for c in unicodedata.normalize('NFKC', value).casefold()
                    if unicodedata.category(c) != 'Mn')
    return value.translate(str.maketrans('أإآى', 'اااي'))


def contains(text: str, term: str) -> bool:
    # Word boundaries prevent e.g. تحدي matching تحديد, SQL matching NoSQL.
    return re.search(r'(?<!\w)' + re.escape(normalize(term)) + r'(?!\w)', normalize(text)) is not None


def topic_matches(text: str, term: str) -> bool:
    # A deliberately small, directional subject vocabulary, not embeddings or
    # record enrichment. A broad analysis query may retrieve a Power BI activity;
    # a specific Power BI query must not retrieve every data-analysis activity.
    analysis_terms = {'تحليل البيانات', 'تحليل بيانات', 'data analysis', 'data analytics'}
    return contains(text, term) or (
        normalize(term) in analysis_terms and contains(text, 'Power BI')
    )


class OpportunityRepository:
    def __init__(self, bridge: DatasetLoader) -> None:
        self.bridge = bridge

    def load(self) -> list[OpportunityRecord]:
        try:
            records = [OpportunityRecord.model_validate(row)
                       for row in self.bridge.load_prepared_opportunities()]
            if len({r.opportunity_id for r in records}) != len(records):
                raise ValueError('Duplicate opportunity IDs')
            raw = {r['source_id']: r for r in self.bridge.load_raw_opportunities()}
            for record in records:
                for source in record.sources + record.related_sources:
                    original = raw.get(source.source_id)
                    if (original is None or source.content != original['content']
                            or source.source_name != original['source_name']
                            or source.published_at.isoformat() != original['published_at']):
                        raise ValueError('Source evidence differs from raw data')
                fingerprint = hashlib.sha256(json.dumps(
                    record.model_dump(mode='json', exclude={'evidence_version'}),
                    sort_keys=True, ensure_ascii=False,
                ).encode()).hexdigest()[:20]
                record.evidence_version = fingerprint
            return records
        except (ValidationError, ValueError, TypeError) as exc:
            raise DataAccessError('Prepared records failed validation.') from exc

    def is_ready(self) -> bool:
        return bool(self.load())

    def search(self, plan: SearchPlan) -> list[OpportunityRecord]:
        def matches(record):
            text = record.title + ' ' + ' '.join(record.topics) + ' ' + (record.description or '')
            return ((not plan.keywords or any(topic_matches(text, word) for word in plan.keywords))
                    and not any(topic_matches(text, word) for word in plan.excluded_keywords))
        return [r for r in self.load() if matches(r)]

    def get_record(self, opportunity_id: str) -> OpportunityRecord | None:
        return next((r for r in self.load() if r.opportunity_id == opportunity_id), None)

    def get_details(self, opportunity_id: str) -> OpportunityDetailsResponse | None:
        record = self.get_record(opportunity_id)
        return OpportunityDetailsResponse.model_validate(record.model_dump()) if record else None
