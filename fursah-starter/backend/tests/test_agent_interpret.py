"""Interpretation shape regressions using an offline OpenRouter HTTP boundary."""
import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app.agent import OpenRouterAgent
from app.infrastructure.config import Settings
from app.infrastructure.errors import InvalidModelOutputError
from app.internal import SearchPlan


LIST_FIELDS = {
    'keywords': 'Python',
    'excluded_keywords': 'SQL',
    'excluded_locations': 'الرياض',
    'excluded_types': 'internship',
    'excluded_modes': 'online',
    'constraints.location_preference': 'الرياض',
    'constraints.opportunity_types': 'workshop',
    'constraints.delivery_modes': 'online',
}
INVALID_SHAPES = [
    (path, value)
    for path, singleton in LIST_FIELDS.items()
    for value in (singleton, {'value': singleton}, None)
    if not (path == 'constraints.location_preference' and value is None)
]


@pytest.fixture
def settings(monkeypatch):
    # A non-secret placeholder is used only with MockTransport; no real key is read.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'unit-test-placeholder-not-a-real-key')
    return Settings(_env_file=None, openrouter_model='qwen/qwen3.7-flash',
                    model_max_retries=1, model_output_retries=1)


def payload_for(path, value):
    if path.startswith('constraints.'):
        return {'constraints': {path.split('.')[1]: value}}
    return {path: value}


def mock_agent(settings, responses):
    calls = []

    def handler(request):
        assert request.url == 'https://openrouter.ai/api/v1/chat/completions'
        body = json.loads(request.content)
        calls.append(body)
        assert body['model'] == 'qwen/qwen3.7-flash'
        assert body['max_tokens'] == 1800
        assert body['temperature'] == 0
        assert body['response_format']['type'] == 'json_schema'
        assert body['response_format']['json_schema']['strict'] is True
        assert body['reasoning'] == {'enabled': False}
        assert body['provider'] == {'require_parameters': True}
        response = responses[min(len(calls) - 1, len(responses) - 1)]
        content = json.dumps(response, ensure_ascii=False)
        # Match the local failure's response size without changing its valid JSON.
        content += ' ' * max(0, 413 - len(content))
        return httpx.Response(200, json={'choices': [{
            'finish_reason': 'stop', 'message': {'content': content},
        }]})

    return OpenRouterAgent(settings, httpx.MockTransport(handler)), calls


def test_prompt_explicitly_defines_arrays_and_valid_unspecified_structure(settings):
    agent, calls = mock_agent(settings, [{}])
    result = asyncio.run(agent.interpret('فرص مناسبة'))
    assert isinstance(result, SearchPlan) and len(calls) == 1
    prompt = calls[0]['messages'][0]['content']
    for path in LIST_FIELDS:
        assert path in prompt
    assert 'must always be JSON arrays' in prompt
    assert 'Use [] when unspecified; never use null, a string, or an object' in prompt
    assert 'Even one value must be wrapped in an array' in prompt
    assert 'when unspecified, use null or []' in prompt
    assert 'constraints itself must be an object, never null' in prompt
    # Check the actual reference sent to the model, independently of model output.
    reference = prompt.split('Populate it using only the explicit query:\n', 1)[1]
    reference = json.loads(reference.split('\nJSON schema:\n', 1)[0])
    for path in LIST_FIELDS:
        expected = None if path == 'constraints.location_preference' else []
        value = reference['constraints'][path.split('.')[1]] if path.startswith('constraints.') else reference[path]
        assert value == expected
    assert reference['constraints']['max_duration_days'] is None
    assert reference['constraints']['max_budget_sar'] is None
    SearchPlan.model_validate(reference)
    assert json.loads(calls[0]['messages'][1]['content']) == {'query': 'فرص مناسبة'}


@pytest.mark.parametrize('path,singleton', LIST_FIELDS.items())
def test_single_value_is_accepted_only_as_an_array(settings, path, singleton):
    agent, calls = mock_agent(settings, [payload_for(path, [singleton])])
    result = asyncio.run(agent.interpret('synthetic request'))
    value = result.constraints.model_dump()[path.split('.')[1]] if path.startswith('constraints.') else getattr(result, path)
    assert value == [singleton] and len(calls) == 1


@pytest.mark.parametrize('value', [None, []])
def test_unspecified_location_accepts_null_or_empty_array(settings, value):
    agent, calls = mock_agent(settings, [payload_for('constraints.location_preference', value)])
    result = asyncio.run(agent.interpret('فرص مناسبة'))
    assert result.constraints.location_preference == value
    assert len(calls) == 1


@pytest.mark.parametrize('path,value', INVALID_SHAPES)
def test_non_list_shapes_remain_rejected_without_coercion(settings, caplog, path, value):
    payload = payload_for(path, value)
    with pytest.raises(ValidationError) as error:
        SearchPlan.model_validate_json(json.dumps(payload, ensure_ascii=False))
    details = error.value.errors(include_input=False, include_context=False, include_url=False)
    assert {item['type'] for item in details} == {'list_type'}
    assert ['.'.join(str(part) for part in item['loc']) for item in details] == [path]
    agent, calls = mock_agent(settings, [payload])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.interpret('synthetic request'))
    assert len(calls) == 2
    diagnostics = '\n'.join(record.getMessage() for record in caplog.records if record.name == 'uvicorn.error')
    assert 'schema=SearchPlan' in diagnostics and 'finish=stop' in diagnostics
    assert 'content_kind=str content_chars=413' in diagnostics
    assert "error_types=['list_type']" in diagnostics
    assert 'attempt=1/2' in diagnostics and 'attempt=2/2' in diagnostics
    assert 'unit-test-placeholder-not-a-real-key' not in diagnostics


@pytest.mark.parametrize('network_retries,output_retries,expected_attempts', [
    (0, 0, 1), (0, 2, 1), (2, 0, 1), (1, 1, 2), (2, 2, 3), (8, 8, 3),
])
def test_invalid_shape_retry_budget_is_unchanged_and_capped(settings, network_retries, output_retries, expected_attempts):
    settings.model_max_retries = network_retries
    settings.model_output_retries = output_retries
    agent, calls = mock_agent(settings, [{'keywords': None}])
    with pytest.raises(InvalidModelOutputError):
        asyncio.run(agent.interpret('synthetic request'))
    assert len(calls) == expected_attempts


def test_retry_accepts_corrected_response_without_coercing_first_response(settings):
    agent, calls = mock_agent(settings, [{'keywords': 'Python'}, {'keywords': ['Python']}])
    result = asyncio.run(agent.interpret('Python'))
    assert result.keywords == ['Python'] and len(calls) == 2
