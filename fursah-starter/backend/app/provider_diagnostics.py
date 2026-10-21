"""Allowlisted provider diagnostics: never log raw messages, bodies or headers."""
import json
import logging
import re

import httpx


_ERROR_LABELS = frozenset({
    'invalid_request_error', 'invalid_request', 'bad_request', 'invalid_argument',
    'unsupported_parameter', 'unsupported_value', 'invalid_parameter', 'invalid_value',
    'invalid_json_schema', 'invalid_schema', 'model_not_found', 'no_available_provider',
    'no_endpoints_found', 'provider_error', 'server_error', 'internal_server_error',
    'authentication_error', 'invalid_api_key', 'insufficient_quota', 'insufficient_credits',
    'permission_error', 'rate_limit_error', 'rate_limit_exceeded', 'timeout',
})
_PARAMETERS = (
    'response_format', 'json_schema', 'strict', 'reasoning', 'reasoning.enabled',
    'provider.require_parameters', 'require_parameters', 'max_tokens',
    'max_completion_tokens', 'temperature', 'model',
    'maxItems', 'minItems', 'additionalProperties', '$ref', '$defs', 'format',
    'exclusiveMinimum', 'minimum', 'minLength', 'maxLength',
)
_MESSAGES = (
    (('no endpoints',), 'No compatible endpoints found.'),
    (('not a valid model', 'unknown model', 'model not found', 'invalid model'), 'Model identifier rejected or unavailable.'),
    (('unsupported', 'not supported', 'does not support', 'not allowed', 'not permitted',
      'unknown parameter', 'unrecognized parameter', 'unexpected keyword', 'invalid schema',
      'invalid parameter', 'invalid value'), 'Request parameter or schema feature rejected.'),
    (('insufficient credits', 'more credits', 'not enough credits', 'insufficient balance',
      'payment required', 'insufficient_quota'), 'Insufficient credits or quota.'),
    (('invalid api key', 'no auth credentials', 'user not found', 'unauthorized',
      'authentication'), 'Provider authentication failed.'),
    (('data policy', 'privacy', 'data collection', 'zdr'), 'Provider routing or data-policy restriction.'),
    (('rate limit', 'too many requests'), 'Provider rate limit reached.'),
    (('provider returned error', 'upstream'), 'Upstream provider returned an error.'),
)


def _safe_model(value):
    if (isinstance(value, str) and len(value) <= 120
            and re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.:-]+', value)
            and not re.search(r'sk-|bearer|secret|api[_-]?key', value, re.IGNORECASE)):
        return value
    return '[omitted]'


def _error_details(response):
    codes, types, parameters, messages = set(), set(), set(), []

    def visit(node, depth=0):
        if depth > 3 or not isinstance(node, dict):
            return
        for field, output in (('code', codes), ('type', types)):
            value = node.get(field)
            if type(value) is int and 100 <= value <= 599:
                output.add(str(value))
            elif isinstance(value, str) and value.casefold() in _ERROR_LABELS:
                output.add(value.casefold())
            elif isinstance(value, str) and re.fullmatch(r'[1-5][0-9]{2}', value):
                output.add(value)
        for field in ('param', 'parameter'):
            value = node.get(field)
            if isinstance(value, str) and value in _PARAMETERS:
                parameters.add(value)
        message = node.get('message')
        if isinstance(message, str):
            messages.append(message[:2048])
        nested = node.get('error')
        if isinstance(nested, str):
            messages.append(nested[:2048])
        visit(nested, depth + 1)
        metadata = node.get('metadata')
        if isinstance(metadata, dict):
            raw = metadata.get('raw')
            if isinstance(raw, str) and len(raw) <= 8192:
                try:
                    raw = json.loads(raw)
                except (ValueError, RecursionError):
                    messages.append(raw[:2048])
            visit(raw, depth + 1)

    # Parse only error fields in bounded error bodies. Ignore choices/content,
    # provider metadata other than raw errors, URLs, headers and request bodies.
    if response is not None and len(response.content) <= 16384:
        try:
            data = response.json()
            if isinstance(data, dict):
                error = data.get('error')
                if isinstance(error, str):
                    messages.append(error[:2048])
                else:
                    visit(error)
        except (ValueError, UnicodeError, RecursionError):
            pass
    combined = '\n'.join(messages)
    lower = combined.casefold()
    summaries = [summary for markers, summary in _MESSAGES if any(marker in lower for marker in markers)]
    for name in _PARAMETERS:
        if re.search(r'(?<![\w])' + re.escape(name) + r'(?![\w])', combined, re.IGNORECASE):
            parameters.add(name)
    return (sorted(codes), sorted(types), sorted(parameters),
            ' '.join(summaries) or 'Unrecognized or absent provider message; raw text omitted.')


def log_provider_failure(body, attempt, total, *, response=None, transport='none'):
    codes, types, parameters, message = _error_details(response)
    logging.getLogger('uvicorn.error').warning(
        'FURSAH provider request failed: status=%s transport=%s model=%s '
        'attempt=%d/%d json_schema=%s strict_schema=%s reasoning_control=%s '
        'require_parameters=%s max_tokens=%d temperature=%s '
        'error_codes=%s error_types=%s error_parameters=%s provider_message=%s',
        response.status_code if response is not None else 'unavailable', transport,
        _safe_model(body['model']), attempt, total,
        body.get('response_format', {}).get('type') == 'json_schema',
        body.get('response_format', {}).get('json_schema', {}).get('strict') is True,
        body.get('reasoning', {}).get('enabled') is False,
        body.get('provider', {}).get('require_parameters') is True,
        body['max_tokens'], body['temperature'], codes, types, parameters, message,
    )


def transport_kind(exc):
    # Do not stringify exceptions: their messages can contain URLs/credentials.
    for kind in (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout,
                 httpx.PoolTimeout, httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                 httpx.WriteError, httpx.RemoteProtocolError, httpx.ProxyError):
        if isinstance(exc, kind):
            return kind.__name__
    return 'HTTPError' if isinstance(exc, httpx.HTTPError) else 'OSError'
