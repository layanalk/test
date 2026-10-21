"""Provider contract and failure regressions, entirely at a mocked HTTP boundary."""
import asyncio
import json

import httpx
import pytest

from app.agent import OpenRouterAgent
from app.infrastructure.config import Settings
from app.infrastructure.errors import InvalidModelOutputError, ModelExecutionError
from app.internal import SearchPlan


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'unit-test-placeholder-not-a-real-key')
    return Settings(_env_file=None, openrouter_model='qwen/qwen3.7-flash',
                    model_max_retries=1, model_output_retries=1)


def completion(content=None, finish='stop', **message):
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {'choices': [{'finish_reason': finish, 'message': {'content': content, **message}}]}


def mock_agent(settings, responses):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        response = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(response, Exception):
            raise response
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, json=response)

    return OpenRouterAgent(settings, httpx.MockTransport(handler)), calls


def test_native_schema_enforces_all_fields_arrays_enums_and_nulls(settings):
    agent, calls = mock_agent(settings, [completion(SearchPlan().model_dump(mode='json'))])
    result = asyncio.run(agent.interpret('فرص مناسبة'))
    assert result == SearchPlan() and len(calls) == 1
    body = calls[0]
    assert body['reasoning'] == {'enabled': False}
    assert body['provider'] == {'require_parameters': True}
    assert body['max_tokens'] == 1800 and 'max_completion_tokens' not in body
    wire = body['response_format']
    assert wire['type'] == 'json_schema'
    assert wire['json_schema']['name'] == 'SearchPlan' and wire['json_schema']['strict'] is True
    schema = wire['json_schema']['schema']
    assert schema['additionalProperties'] is False
    assert set(schema['required']) == set(schema['properties'])
    assert schema['properties']['keywords']['type'] == 'array'
    nested = schema['$defs']['SearchConstraints']
    assert nested['additionalProperties'] is False
    assert set(nested['required']) == set(nested['properties'])
    location = nested['properties']['location_preference']['anyOf']
    assert {part['type'] for part in location} == {'array', 'null'}
    assert nested['properties']['delivery_modes']['items']['enum'] == ['in_person', 'online', 'hybrid']
    assert 'default' not in json.dumps(schema)
    # Building a stricter wire schema must not mutate the local API/Pydantic schema.
    assert SearchPlan.model_json_schema()['$defs']['SearchConstraints']['properties']['available_from']['default'] is None
    assert json.loads(body['messages'][1]['content']) == {'query': 'فرص مناسبة'}
    assert 'sources' not in body['messages'][1]['content']


@pytest.mark.parametrize('content,finish', [
    ('{"keywords": [', 'length'), (None, 'length'),
    ('{}', 'length'),  # Even parseable JSON must be rejected when truncated.
    ('not JSON', 'stop'), ('```json\n{}\n```', 'stop'),
    (None, 'stop'), ({'constraints': None}, 'stop'),
    ({'keywords': 'Python'}, 'stop'), ({'constraints': {'delivery_modes': ['remote']}}, 'stop'),
    ({'constraints': {'max_budget_sar': '0'}}, 'stop'),
    ({'constraints': {'available_from': '2026-12-01', 'available_to': '2026-11-01'}}, 'stop'),
    ({}, 'content_filter'), ({}, 'error'), ({}, 'tool_calls'), ({}, None),
])
def test_malformed_incomplete_and_refused_output_never_reaches_search(settings, caplog, content, finish):
    agent, calls = mock_agent(settings, [completion(content, finish)])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.interpret('Python'))
    assert len(calls) == 2
    logs = '\n'.join(record.getMessage() for record in caplog.records)
    assert 'schema=SearchPlan' in logs and 'attempt=2/2' in logs
    assert 'unit-test-placeholder-not-a-real-key' not in logs
    assert 'not JSON' not in logs


@pytest.mark.parametrize('envelope', [{}, {'choices': []}, {'choices': None},
                                       {'choices': [{'message': None}]}, {'choices': [None]}])
def test_malformed_provider_envelope_is_controlled(settings, envelope):
    agent, calls = mock_agent(settings, [envelope])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.interpret('Python'))
    assert len(calls) == 2


def test_reasoning_heavy_response_is_not_parsed_or_logged(settings, caplog):
    private_reasoning = 'synthetic-private-reasoning ' + SearchPlan(keywords=['Python']).model_dump_json()
    response = completion(None, 'length', reasoning=private_reasoning,
                          reasoning_details=[{'text': private_reasoning}])
    response['usage'] = {'completion_tokens': 1800, 'completion_tokens_details': {'reasoning_tokens': 1800}}
    agent, calls = mock_agent(settings, [response])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.interpret('Python'))
    assert len(calls) == 2
    assert all(body['reasoning'] == {'enabled': False} for body in calls)
    logs = '\n'.join(record.getMessage() for record in caplog.records)
    assert 'finish=length' in logs and 'content_kind=NoneType content_chars=0' in logs
    assert private_reasoning not in logs


def test_valid_final_json_is_used_instead_of_reasoning_text(settings, caplog):
    agent, calls = mock_agent(settings, [completion({'keywords': ['Python']}, reasoning='private-reasoning-marker')])
    assert asyncio.run(agent.interpret('Python')).keywords == ['Python']
    assert len(calls) == 1 and 'private-reasoning-marker' not in caplog.text


def test_non_json_http_success_is_controlled(settings):
    agent, calls = mock_agent(settings, [httpx.Response(200, text='<html>private-provider-marker</html>')])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.interpret('Python'))
    assert len(calls) == 2


def test_truncated_completion_can_recover_with_one_compact_retry(settings):
    agent, calls = mock_agent(settings, [completion('{"keywords":[', 'length'),
                                         completion({'keywords': ['Python']})])
    assert asyncio.run(agent.interpret('Python')).keywords == ['Python']
    assert len(calls) == 2
    assert calls[0]['max_tokens'] == calls[1]['max_tokens'] == 1800
    assert len(calls[1]['messages']) == 3
    assert calls[1]['messages'][-1]['role'] == 'user'
    assert '{"keywords":[' not in calls[1]['messages'][-1]['content']


@pytest.mark.parametrize('status', [400, 401, 402, 403, 404, 422])
def test_permanent_errors_including_unsupported_parameters_do_not_retry_or_downgrade(settings, caplog, status):
    agent, calls = mock_agent(settings, [httpx.Response(status, json={
        'error': {'message': 'private-unsupported-parameter-marker'},
    })])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('Python'))
    assert len(calls) == 1
    assert calls[0]['response_format']['type'] == 'json_schema'
    assert calls[0]['reasoning'] == {'enabled': False}
    assert calls[0]['provider'] == {'require_parameters': True}
    assert 'private-unsupported-parameter-marker' not in caplog.text


@pytest.mark.parametrize('status', [408, 429, 500, 502, 503, 504])
def test_transient_provider_failures_remain_bounded(settings, status):
    agent, calls = mock_agent(settings, [httpx.Response(status, json={'error': 'transient'})])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('Python'))
    assert len(calls) == 2


def test_transport_timeout_retries_with_same_timeout_and_stops(settings):
    agent, calls = mock_agent(settings, [httpx.ReadTimeout('private-timeout-marker')])
    with pytest.raises(ModelExecutionError) as error:
        asyncio.run(agent.interpret('Python'))
    assert len(calls) == 2 and 'private-timeout-marker' not in str(error.value)


def test_network_and_output_failures_share_one_retry_budget(settings):
    agent, calls = mock_agent(settings, [httpx.Response(503), completion('{', 'length'), completion({})])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.interpret('Python'))
    assert len(calls) == 2
