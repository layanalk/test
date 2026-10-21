"""Safe HTTP diagnostics using local mocks, never provider requests or real keys."""
import asyncio
import json

import httpx
import pytest

from app.infrastructure.errors import ModelExecutionError
from app.provider_diagnostics import log_provider_failure
from test_agent_provider import completion, mock_agent, settings


def diagnostics(caplog):
    return '\n'.join(record.getMessage() for record in caplog.records if record.name == 'uvicorn.error')


@pytest.mark.parametrize('status,message,summary,parameter', [
    (404, 'No endpoints found that support the requested parameters.', 'No compatible endpoints found.', None),
    (400, 'response_format json_schema is not supported', 'Request parameter or schema feature rejected.', 'json_schema'),
    (400, 'reasoning.enabled is an unsupported parameter', 'Request parameter or schema feature rejected.', 'reasoning.enabled'),
    (400, 'Invalid schema: maxItems is not permitted', 'Request parameter or schema feature rejected.', 'maxItems'),
    (402, 'This request requires more credits or fewer max_tokens.', 'Insufficient credits or quota.', 'max_tokens'),
    (401, 'No auth credentials found', 'Provider authentication failed.', None),
    (404, 'qwen/qwen3.7-flash is not a valid model ID', 'Model identifier rejected or unavailable.', 'model'),
    (403, 'Provider privacy data policy prevents routing', 'Provider routing or data-policy restriction.', None),
    (429, 'Too many requests; rate limit reached', 'Provider rate limit reached.', None),
])
def test_http_diagnostic_exposes_status_and_safe_cause_without_changing_attempts(settings, caplog, status, message, summary, parameter):
    response = httpx.Response(status, json={'error': {'code': status, 'message': message}})
    agent, calls = mock_agent(settings, [response])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('private-user-query-marker'))
    expected = 2 if status == 429 else 1
    assert len(calls) == expected
    logs = diagnostics(caplog)
    assert logs.count('FURSAH provider request failed:') == expected
    for value in (f'status={status}', 'model=qwen/qwen3.7-flash', f'attempt=1/2',
                  'json_schema=True', 'strict_schema=True', 'reasoning_control=True',
                  'require_parameters=False', 'max_tokens=1800', 'temperature=0', summary):
        assert value in logs
    assert f"error_codes=['{status}']" in logs
    if parameter:
        assert parameter in logs.split('error_parameters=', 1)[1].split(' provider_message=', 1)[0]
    assert 'private-user-query-marker' not in logs
    assert 'unit-test-placeholder-not-a-real-key' not in logs


def test_nested_provider_raw_error_yields_only_allowlisted_fields(settings, caplog):
    private = 'private-user-source-and-response-marker'
    agent, calls = mock_agent(settings, [httpx.Response(400, json={'error': {
        'code': 400, 'message': 'Provider returned error',
        'metadata': {'provider_name': private, 'raw': json.dumps({'error': {
            'type': 'invalid_request_error', 'code': 'unsupported_parameter',
            'param': 'response_format', 'message': 'json_schema not supported: ' + private,
            'input': private, 'authorization': 'Bearer unit-test-placeholder-not-a-real-key',
        }})},
    }, 'choices': [{'message': {'content': private}}]})])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret(private))
    logs = diagnostics(caplog)
    assert len(calls) == 1
    assert "error_codes=['400', 'unsupported_parameter']" in logs
    assert "error_types=['invalid_request_error']" in logs
    assert "error_parameters=['json_schema', 'response_format']" in logs
    assert 'Request parameter or schema feature rejected.' in logs
    assert 'Upstream provider returned an error.' in logs
    assert private not in logs and 'Bearer' not in logs and 'Authorization' not in logs
    assert 'unit-test-placeholder-not-a-real-key' not in logs


@pytest.mark.parametrize('body', [
    {}, {'error': None}, {'error': []}, {'error': {'message': None, 'metadata': []}},
    {'error': {'message': {'private': 'private-marker'}, 'code': {'private': 'private-marker'}}},
    {'error': {'code': 'private-marker', 'type': 'private-marker', 'param': 'private-marker', 'message': 'private-marker'}},
    {'error': 'private-marker'}, ['private-marker'],
])
def test_missing_unrecognized_or_malformed_error_fields_cannot_escape_or_leak(settings, caplog, body):
    agent, calls = mock_agent(settings, [httpx.Response(400, json=body)])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('private-marker'))
    logs = diagnostics(caplog)
    assert len(calls) == 1 and 'status=400' in logs
    assert 'raw text omitted' in logs and 'private-marker' not in logs


@pytest.mark.parametrize('text', ['<html>private-marker</html>', 'not JSON private-marker',
                                  'x' * 17000, '[' * 2000 + '0' + ']' * 2000])
def test_non_json_oversized_or_deeply_nested_error_body_stays_controlled(settings, caplog, text):
    agent, calls = mock_agent(settings, [httpx.Response(400, text=text)])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('private-marker'))
    logs = diagnostics(caplog)
    assert len(calls) == 1 and 'status=400' in logs
    assert 'private-marker' not in logs and len(logs) < 1000


def test_malformed_nested_raw_is_classified_but_never_printed(settings, caplog):
    agent, calls = mock_agent(settings, [httpx.Response(400, json={'error': {'metadata': {
        'raw': 'unsupported temperature; private-marker\nAuthorization: Bearer unit-test-placeholder-not-a-real-key',
    }}})])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('private-marker'))
    logs = diagnostics(caplog)
    assert 'Request parameter or schema feature rejected.' in logs
    assert "error_parameters=['temperature']" in logs
    assert 'private-marker' not in logs and 'Bearer' not in logs


@pytest.mark.parametrize('exception,kind', [
    (httpx.ReadTimeout('private exception with headers and query'), 'ReadTimeout'),
    (httpx.ConnectError('private exception with headers and query'), 'ConnectError'),
    (httpx.RemoteProtocolError('private exception with headers and query'), 'RemoteProtocolError'),
    (OSError('private exception with headers and query'), 'OSError'),
])
def test_transport_errors_are_diagnostic_without_logging_exception_text(settings, caplog, exception, kind):
    agent, calls = mock_agent(settings, [exception])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('private-user-query'))
    logs = diagnostics(caplog)
    assert len(calls) == 2 and logs.count('FURSAH provider request failed:') == 2
    assert f'status=unavailable transport={kind}' in logs and 'attempt=2/2' in logs
    assert 'private exception' not in logs and 'private-user-query' not in logs


def test_retry_diagnostics_are_once_per_http_failure_and_do_not_change_request(settings, caplog):
    agent, calls = mock_agent(settings, [httpx.Response(503, json={'error': {'code': 503, 'message': 'upstream error'}}), completion({})])
    result = asyncio.run(agent.interpret('Python'))
    assert not result.keywords and len(calls) == 2 and calls[0] == calls[1]
    assert diagnostics(caplog).count('FURSAH provider request failed:') == 1


def test_untrusted_metadata_cannot_inject_log_lines_or_disclose_secrets(settings, caplog):
    secret = 'unit-test-placeholder-not-a-real-key'
    private = 'private prompt/source marker'
    agent, calls = mock_agent(settings, [httpx.Response(400, headers={'Authorization': 'Bearer ' + secret}, json={'error': {
        'message': f'unsupported json_schema\nFAKE_LOG_LINE {secret} {private}',
        'code': secret, 'type': private, 'param': secret,
        'metadata': {'raw': json.dumps({'error': {'message': f'unsupported reasoning\r\n{secret} {private}'}})},
    }})])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret(private))
    logs = diagnostics(caplog)
    assert len(calls) == 1 and 'Request parameter or schema feature rejected.' in logs
    assert all(part not in logs for part in (secret, private, 'FAKE_LOG_LINE', 'Authorization', 'Bearer', '\r', '\n'))


def test_bad_model_configuration_is_not_echoed_in_diagnostics(settings, caplog):
    settings.openrouter_model = 'Bearer unit-test-placeholder-not-a-real-key/private-marker'
    agent, _ = mock_agent(settings, [httpx.Response(400)])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('private-marker'))
    logs = diagnostics(caplog)
    assert 'model=[omitted]' in logs
    assert 'unit-test-placeholder-not-a-real-key' not in logs and 'private-marker' not in logs


def test_missing_credential_is_distinguished_without_an_http_call(settings, monkeypatch, caplog):
    monkeypatch.delenv('OPENROUTER_API_KEY')
    agent, calls = mock_agent(settings, [httpx.Response(400)])
    with pytest.raises(ModelExecutionError):
        asyncio.run(agent.interpret('private-marker'))
    logs = diagnostics(caplog)
    assert not calls and 'request not sent: reason=missing_process_credential' in logs
    assert 'private-marker' not in logs


def test_direct_helper_does_not_inspect_request_messages(settings, caplog):
    body = {'model': 'qwen/qwen3.7-flash', 'temperature': 0, 'max_tokens': 1800,
            'response_format': {'type': 'json_schema', 'json_schema': {'strict': True}},
            'reasoning': {'enabled': False}, 'provider': {'require_parameters': True},
            'messages': 'private-marker'}
    log_provider_failure(body, 1, 2, response=httpx.Response(404, json={'error': {'code': '404', 'type': 'INVALID_ARGUMENT'}}))
    logs = diagnostics(caplog)
    assert "error_codes=['404']" in logs and "error_types=['invalid_argument']" in logs
    assert 'private-marker' not in logs


def test_diagnostics_report_absent_controls_without_assuming_json_schema(settings, caplog):
    body = {'model': 'qwen/qwen3.7-flash', 'temperature': 0, 'max_tokens': 1800,
            'response_format': {'type': 'json_object'}}
    log_provider_failure(body, 1, 2, response=httpx.Response(400))
    logs = diagnostics(caplog)
    assert 'json_schema=False strict_schema=False reasoning_control=False require_parameters=False' in logs
