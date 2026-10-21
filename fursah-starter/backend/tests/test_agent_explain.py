"""Offline deterministic explanation and strict evidence verification regressions."""
import json

import pytest
from pydantic import ValidationError

from app.explanations import build_explanations
from app.infrastructure.errors import EvidenceValidationError
from app.internal import ExplanationBatch, OpportunityRecord, SearchPlan
from app.validation import validate_explanations


@pytest.fixture
def records():
    return [OpportunityRecord.model_validate({
        'opportunity_id': f'OPP-TEST-{i}', 'title': f'ورشة Python {i}',
        'opportunity_type': 'workshop', 'delivery_mode': 'online',
        'source_ids': [f'SRC-TEST-{i}'],
        'sources': [{'source_id': f'SRC-TEST-{i}', 'source_name': 'مصدر اختبار',
                     'published_at': '2026-10-01T12:00:00+03:00',
                     'content': f'ورشة تطبيقية حول Python رقم {i}. المشاركة عن بعد بالكامل. '
                                + 'تفاصيل مصدرية إضافية. ' * 12}],
    }) for i in range(1, 4)]


def batch_for(records):
    return {'explanations': [
        {'opportunity_id': r.opportunity_id,
         'quotes': [r.sources[0].content.split('.')[0] + '.']}
        for r in records
    ]}

@pytest.mark.parametrize('count', [1, 2, 3])
def test_one_two_three_recommendations_have_compact_grounded_explanations(records, count):
    selected = records[:count]
    result = build_explanations(SearchPlan(keywords=['Python']), selected)
    assert len(result.explanations) == count
    assert len(validate_explanations(result, selected)) == count
    assert all(len(item.quotes) == 1 and 8 <= len(item.quotes[0]) <= 160 for item in result.explanations)
    assert sum(len(item.quotes[0]) for item in result.explanations) <= 480
    assert len(result.model_dump_json()) < 800
    assert result == build_explanations(SearchPlan(keywords=['Python']), selected)


def test_subject_passage_after_long_intro_and_unicode_preserves_original_text(records):
    record = records[0]
    record.sources[0].content = ('مقدمةَ  غير مرتبطة\r\n' * 50
        + 'ورشة Python وتطبيقات معالجة البيانات مع الحفاظ على النص الأصلي. '
        + 'خاتمة عامة ' * 30)
    result = build_explanations(SearchPlan(keywords=['Python']), [record])
    quote = result.explanations[0].quotes[0]
    assert 'Python' in quote and quote in record.sources[0].content
    validate_explanations(result, [record])


def test_source_too_short_fails_closed_without_inventing_evidence(records):
    records[0].sources[0].content = 'قصير'
    with pytest.raises(EvidenceValidationError):
        build_explanations(SearchPlan(), records[:1])


def test_quote_can_come_from_second_original_source(records):
    record = records[0]
    other = records[1].sources[0].model_copy(deep=True)
    record.sources[0].content = 'مقدمة عامة لا تتضمن الموضوع المطلوب. '
    other.content = 'المشاركة في ورشة Python عن بعد وفق الإعلان الأصلي.'
    record.sources.append(other)
    record.source_ids.append(other.source_id)
    result = build_explanations(SearchPlan(keywords=['Python']), [record])
    assert 'Python' in result.explanations[0].quotes[0]
    validate_explanations(result, [record])


@pytest.mark.parametrize('content', ['{"explanations":[', 'not JSON', 'null'])
def test_malformed_or_truncated_explanation_json_is_still_rejected(content):
    with pytest.raises(ValidationError):
        ExplanationBatch.model_validate_json(content)


@pytest.mark.parametrize('defect', [
    'empty', 'too_many', 'long_quote', 'two_quotes', 'short_quote', 'blank_quote',
    'scalar_quotes', 'null_quotes', 'missing_id', 'extra_analysis',
])
def test_invalid_or_verbose_structured_output_stays_rejected(records, defect):
    payload = batch_for(records[:1])
    item = payload['explanations'][0]
    if defect == 'empty': payload['explanations'] = []
    elif defect == 'too_many': payload['explanations'] = batch_for(records)['explanations'] + [item]
    elif defect == 'long_quote': item['quotes'] = ['x' * 161]
    elif defect == 'two_quotes': item['quotes'] *= 2
    elif defect == 'short_quote': item['quotes'] = ['x' * 7]
    elif defect == 'blank_quote': item['quotes'] = [' ' * 8]
    elif defect == 'scalar_quotes': item['quotes'] = item['quotes'][0]
    elif defect == 'null_quotes': item['quotes'] = None
    elif defect == 'missing_id': del item['opportunity_id']
    elif defect == 'extra_analysis': item['analysis'] = 'Do not accept extra prose.'
    with pytest.raises(ValidationError):
        ExplanationBatch.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize('defect', ['invented_quote', 'invented_id', 'missing_id', 'duplicate_id', 'wrong_source'])
def test_short_schema_valid_output_still_requires_original_evidence(records, defect):
    selected = records[:2]
    payload = batch_for(selected)
    item = payload['explanations'][0]
    if defect == 'invented_quote': item['quotes'] = ['An invented statement absent from the original source.']
    elif defect == 'invented_id': item['opportunity_id'] = 'INVENTED'
    elif defect == 'missing_id': payload['explanations'].pop()
    elif defect == 'duplicate_id': payload['explanations'][1]['opportunity_id'] = item['opportunity_id']
    elif defect == 'wrong_source': item['quotes'] = payload['explanations'][1]['quotes'].copy()
    result = ExplanationBatch.model_validate(payload)
    with pytest.raises(EvidenceValidationError):
        validate_explanations(result, selected)
