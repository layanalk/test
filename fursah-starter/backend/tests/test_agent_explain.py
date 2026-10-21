"""Compact explanation and budget-failure coverage; no real model calls."""
import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app.agent import OpenRouterAgent
from app.infrastructure.config import Settings
from app.infrastructure.errors import EvidenceValidationError, InvalidModelOutputError
from app.internal import ExplanationBatch, OpportunityRecord
from app.validation import validate_explanations


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'unit-test-placeholder-not-a-real-key')
    return Settings(_env_file=None, openrouter_model='qwen/qwen3.7-flash',
                    model_max_retries=1, model_output_retries=1)


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


def batch_for(records, longest=False):
    return {'explanations': [
        {'opportunity_id': r.opportunity_id,
         'quotes': [r.sources[0].content[:160].strip() if longest else r.sources[0].content.split('.')[0] + '.']}
        for r in records
    ]}


def mock_agent(settings, responses):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        assert request.url == 'https://openrouter.ai/api/v1/chat/completions'
        assert calls[-1]['max_tokens'] == 1800
        assert calls[-1]['response_format'] == {'type': 'json_object'}
        assert calls[-1]['temperature'] == 0
        content, finish = responses[min(len(calls) - 1, len(responses) - 1)]
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, separators=(',', ':'))
        return httpx.Response(200, json={'choices': [{
            'finish_reason': finish, 'message': {'content': content},
        }]})

    return OpenRouterAgent(settings, httpx.MockTransport(handler)), calls


@pytest.mark.parametrize('count', [1, 2, 3])
def test_one_two_three_recommendations_get_compact_grounded_explanations(settings, records, count):
    selected = records[:count]
    agent, calls = mock_agent(settings, [(batch_for(selected), 'stop')])
    result = asyncio.run(agent.explain('Python', selected))
    assert len(result.explanations) == count and len(calls) == 1
    checked = validate_explanations(result, selected)
    assert list(checked) == [r.opportunity_id for r in selected]
    assert all(len(item.quotes) == 1 and 8 <= len(item.quotes[0]) <= 160 for item in result.explanations)
    prompt = calls[0]['messages'][0]['content']
    assert 'EXACTLY ONE' in prompt and '8 to 160 characters' in prompt
    assert 'VERBATIM' in prompt and 'compact JSON' in prompt
    assert 'No reasoning, commentary, paraphrases, ellipses, or other fields' in prompt
    assert 'do not decide eligibility' in prompt
    payload = json.loads(calls[0]['messages'][1]['content'])
    assert [r['opportunity_id'] for r in payload['opportunities']] == [r.opportunity_id for r in selected]
    assert payload['opportunities'][0]['sources'][0]['content'] == selected[0].sources[0].content


def test_three_maximum_length_quotes_still_have_bounded_output_size(settings, records):
    agent, calls = mock_agent(settings, [(batch_for(records, longest=True), 'stop')])
    result = asyncio.run(agent.explain('Python', records))
    validate_explanations(result, records)
    assert len(result.explanations) == 3 and len(calls) == 1
    assert sum(len(item.quotes[0]) for item in result.explanations) <= 480
    assert len(result.model_dump_json()) < 800  # Character bound, not an assumed tokenizer count.


@pytest.mark.parametrize('defect', [
    'empty', 'too_many', 'long_quote', 'two_quotes', 'short_quote', 'blank_quote',
    'scalar_quotes', 'null_quotes', 'missing_id', 'extra_analysis',
])
def test_invalid_or_verbose_structured_output_stays_rejected(settings, records, defect):
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
    agent, calls = mock_agent(settings, [(payload, 'stop')])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.explain('Python', records[:1]))
    assert len(calls) == 2


@pytest.mark.parametrize('content', [None, '{"explanations":['])
def test_budget_exhaustion_remains_controlled_and_diagnostic(settings, records, caplog, content):
    agent, calls = mock_agent(settings, [(content, 'length')])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.explain('Python', records))
    assert len(calls) == 2
    logs = '\n'.join(record.getMessage() for record in caplog.records if record.name == 'uvicorn.error')
    assert 'schema=ExplanationBatch' in logs and 'finish=length' in logs
    assert 'attempt=1/2' in logs and 'attempt=2/2' in logs
    assert 'json_type' in logs if content is None else 'json_invalid' in logs
    assert 'unit-test-placeholder-not-a-real-key' not in logs
    if content is None:
        assert 'content_kind=NoneType content_chars=0' in logs


@pytest.mark.parametrize('network,output,expected', [(0, 0, 1), (1, 1, 2), (8, 8, 3)])
def test_explanation_retries_keep_the_existing_hard_bound(settings, records, network, output, expected):
    settings.model_max_retries = network
    settings.model_output_retries = output
    agent, calls = mock_agent(settings, [(None, 'length')])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.explain('Python', records))
    assert len(calls) == expected


def test_one_failed_attempt_can_recover_without_more_retries(settings, records):
    agent, calls = mock_agent(settings, [(None, 'length'), (batch_for(records), 'stop')])
    result = asyncio.run(agent.explain('Python', records))
    assert len(calls) == 2 and len(validate_explanations(result, records)) == 3


@pytest.mark.parametrize('defect', ['invented_quote', 'invented_id', 'missing_id', 'duplicate_id', 'wrong_source'])
def test_short_schema_valid_output_still_requires_original_evidence(settings, records, defect):
    selected = records[:2]
    payload = batch_for(selected)
    item = payload['explanations'][0]
    if defect == 'invented_quote': item['quotes'] = ['An invented statement absent from the original source.']
    elif defect == 'invented_id': item['opportunity_id'] = 'INVENTED'
    elif defect == 'missing_id': payload['explanations'].pop()
    elif defect == 'duplicate_id': payload['explanations'][1]['opportunity_id'] = item['opportunity_id']
    elif defect == 'wrong_source': item['quotes'] = payload['explanations'][1]['quotes'].copy()
    agent, calls = mock_agent(settings, [(payload, 'stop')])
    result = asyncio.run(agent.explain('Python', selected))
    assert len(calls) == 1
    with pytest.raises(EvidenceValidationError):
        validate_explanations(result, selected)
