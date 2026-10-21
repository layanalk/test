"""Exercise the actual LangGraph and FastAPI through a fake OpenRouter HTTP boundary."""
import asyncio
import importlib.util
import json
from datetime import datetime
import httpx
import pytest
from fastapi.testclient import TestClient
from app.agent import OpenRouterAgent
from app.infrastructure.config import Settings
from app.infrastructure.data_package import DatasetLoader
from app.infrastructure.dependencies import AppDependencies, get_app_dependencies
from app.infrastructure.errors import EvidenceValidationError, InvalidModelOutputError, ModelExecutionError, WorkflowBudgetExceededError
from app.infrastructure.tracing import Tracer
from app.internal import SearchPlan
from app.main import app
from app.repository import OpportunityRepository, contains
from app.schemas import DiscoveryRequest, SearchConstraints
from app.tools import deadline_state
from app.workflow import discover

NOW = datetime.fromisoformat('2026-10-21T12:00:00+03:00')

@pytest.fixture
def deps(monkeypatch):
    # Override any developer key. No test can use a real key.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'unit-test-placeholder-not-a-real-key')
    settings = Settings(_env_file=None, evaluation_time=NOW, model_max_retries=1,
                        model_output_retries=1, workflow_timeout_seconds=10)
    bridge = DatasetLoader(settings)
    return AppDependencies(settings, bridge, OpportunityRepository(bridge), OpenRouterAgent(settings), Tracer(), NOW)


def stub(deps, plan=None, failure=None):
    calls = []
    def handler(request):
        assert request.url == 'https://openrouter.ai/api/v1/chat/completions'
        assert request.headers['Authorization'] == 'Bearer unit-test-placeholder-not-a-real-key'
        body = json.loads(request.content)
        calls.append(body)
        if failure == 'http':
            return httpx.Response(401, json={'error': 'provider-body-must-not-be-exposed'})
        if failure == 'invalid_json':
            content = 'not JSON'
        elif failure == 'invalid_schema':
            content = json.dumps({'unexpected': True})
        else:
            payload = json.loads(body['messages'][1]['content'])
            if 'opportunities' in payload:
                items = [{'opportunity_id': r['opportunity_id'], 'quotes': [r['sources'][0]['content'].split('\n')[0]]}
                         for r in payload['opportunities']]
                if failure == 'invented_id': items[0]['opportunity_id'] = 'INVENTED'
                if failure == 'invented_quote': items[0]['quotes'] = ['This evidence was invented by the model.']
                content = json.dumps({'explanations': items}, ensure_ascii=False)
            else:
                content = (plan or SearchPlan()).model_dump_json()
        return httpx.Response(200, json={'choices': [{'message': {'content': content}}]})
    deps.agent = OpenRouterAgent(deps.settings, httpx.MockTransport(handler))
    return calls


def request(**overrides):
    return DiscoveryRequest.model_validate({'request_id': 'test-search', 'query': 'أبحث عن فرص مناسبة',
        'profile': {'interests': [], 'skills': None}, 'constraints': {}, **overrides})


def run(deps, req=None):
    return asyncio.run(discover(req or request(), deps))


def test_graph_returns_three_grounded_results(deps):
    calls = stub(deps)
    result = run(deps)
    assert result.status == 'ok' and len(result.recommendations) == 3
    assert len(calls) == 2
    for rec in result.recommendations:
        record = deps.repository.get_record(rec.opportunity_id)
        assert rec.evidence_version == record.evidence_version and rec.source_ids == record.source_ids
        assert rec.eligibility_status == 'uncertain'
        assert any(s.content.split('\n')[0] in rec.reason for s in record.sources)
    assert {r.opportunity_id for r in result.recommendations}.isdisjoint({'OPP-0014','OPP-0054','OPP-0099','OPP-0005','OPP-0045'})


def test_missing_skills_then_resubmission(deps):
    stub(deps, SearchPlan(keywords=['Python'], constraints=SearchConstraints(
        opportunity_types=['internship'], location_preference=['نجران'])))
    result = run(deps, request(query='تدريب Python'))
    assert result.status == 'needs_info' and result.missing_fields == ['profile.skills']
    assert run(deps, request(profile={'skills': []})).status == 'no_match'
    result = run(deps, request(profile={'skills': [{'name': 'Python'}]}))
    assert result.status == 'ok' and result.recommendations[0].opportunity_id == 'OPP-0086'
    assert result.recommendations[0].eligibility_status == 'uncertain'


def test_budget_unknown_is_not_free_and_duration_is_enforced(deps):
    stub(deps)
    result = run(deps, request(constraints={'max_budget_sar': 0, 'max_duration_days': 3}))
    assert result.status == 'ok'
    for rec in result.recommendations:
        record = deps.repository.get_record(rec.opportunity_id)
        assert record.cost == 0 and record.duration <= 3
    assert 'OPP-0070' not in [r.opportunity_id for r in result.recommendations]


def test_conflict_needs_choice_and_does_not_overwrite_preferences(deps):
    stub(deps, SearchPlan(constraints=SearchConstraints(location_preference=['جدة'])))
    req = request(query='فرص في جدة', constraints={'location_preference': ['ينبع']})
    result = run(deps, req)
    assert result.status == 'needs_info' and result.constraint_conflict
    for choice, expected in [('profile', 'ينبع'), ('query', 'جدة')]:
        result = run(deps, req.model_copy(update={'constraint_resolution': choice}))
        assert result.status == 'ok' and all(rec.location == expected for rec in result.recommendations)
    assert req.constraints.location_preference == ['ينبع']


def test_negation_excludes_records(deps):
    stub(deps, SearchPlan(excluded_modes=['online']))
    result = run(deps)
    assert result.status == 'ok' and all(r.delivery_mode != 'online' for r in result.recommendations)


def test_no_match_is_not_fake_success(deps):
    calls = stub(deps, SearchPlan(keywords=['موضوعغيرموجود']))
    result = run(deps)
    assert result.status == 'no_match' and not result.recommendations and len(calls) == 1


@pytest.mark.parametrize('failure,exception', [('invented_id',EvidenceValidationError),('invented_quote',EvidenceValidationError),
    ('invalid_json',InvalidModelOutputError),('invalid_schema',InvalidModelOutputError),('http',ModelExecutionError)])
def test_model_failures_are_controlled(deps, failure, exception):
    calls = stub(deps, failure=failure)
    with pytest.raises(exception): run(deps)
    assert len(calls) <= 2


def test_no_key_does_not_prevent_startup_or_details(deps, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setenv('FURSAH_MODEL_API_KEY', 'must-not-be-used')
    assert not deps.agent.is_configured()
    with pytest.raises(ModelExecutionError): run(deps)
    app.dependency_overrides[get_app_dependencies] = lambda: deps
    try:
        with TestClient(app) as client:
            assert client.get('/health').status_code == 200
            result = client.get('/ready')
            assert result.status_code == 503 and result.json()['checks']['storage']
            result = client.post('/api/v1/discover', json=request().model_dump(mode='json'))
            assert result.status_code == 503 and result.json()['error_code'] == 'FURSAH_MODEL_ERROR'
            assert client.get('/api/v1/opportunities/OPP-0070').status_code == 200
    finally: app.dependency_overrides.clear()


def test_api_end_to_end_with_mock_openrouter(deps):
    stub(deps)
    app.dependency_overrides[get_app_dependencies] = lambda: deps
    try:
        with TestClient(app) as client:
            assert client.get('/ready').status_code == 200
            result = client.post('/api/v1/discover', json=request().model_dump(mode='json'))
            assert result.status_code == 200
            body = result.json()
            assert body['status'] == 'ok' and 1 <= len(body['recommendations']) <= 3
            rec = body['recommendations'][0]
            details = client.get('/api/v1/opportunities/' + rec['opportunity_id'])
            assert details.status_code == 200 and details.json()['source_ids'] == rec['source_ids']
            assert details.json()['evidence_version'] == rec['evidence_version']
            assert client.get('/api/v1/opportunities/missing').status_code == 404
            assert client.post('/api/v1/discover', json={}).status_code == 422
            cors = client.options('/api/v1/discover', headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST'})
            assert cors.headers['access-control-allow-origin'] == 'http://localhost:5173'
    finally: app.dependency_overrides.clear()


def test_provider_errors_never_expose_body_or_key(deps, capsys):
    stub(deps, failure='http')
    app.dependency_overrides[get_app_dependencies] = lambda: deps
    try:
        with TestClient(app) as client:
            result = client.post('/api/v1/discover', json=request().model_dump(mode='json'))
            assert result.status_code == 503
            text = result.text + capsys.readouterr().out
            assert 'provider-body-must-not-be-exposed' not in text
            assert 'unit-test-placeholder-not-a-real-key' not in text
    finally: app.dependency_overrides.clear()


def test_full_dataset_is_reproducible_and_preserves_evidence(deps, monkeypatch):
    records = deps.repository.load()
    assert len(records) == 1000 and all(r.evidence_version for r in records)
    data = deps.settings.data_dir
    monkeypatch.syspath_prepend(str(data))
    spec = importlib.util.spec_from_file_location('prepare_mvp', data / 'prepare_mvp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.prepare() == json.loads((data / 'prepared/opportunities.json').read_text())


@pytest.mark.parametrize('deadline,expected', [('2026-10-20','expired'),('2026-10-21','uncertain'),
    ('2026-10-22','not_passed'),(None,'unknown'),('2026-10-21T12:00:00+03:00','expired'),
    ('2026-10-21T12:00:01+03:00','not_passed'),('2026-10-21T12:00:00','uncertain')])
def test_deadlines(deadline, expected):
    assert deadline_state(deadline, NOW) == expected


def test_word_matching_avoids_substrings():
    assert contains('معرفة Python مطلوبة', 'Python')
    assert not contains('تحديد السؤال', 'تحدي') and not contains('NoSQL', 'SQL')


def test_workflow_timeout_is_bounded(deps):
    class SlowAgent:
        async def interpret(self, query): await asyncio.sleep(1)
    deps.agent = SlowAgent()
    deps.settings.workflow_timeout_seconds = 0.01
    with pytest.raises(WorkflowBudgetExceededError): run(deps)


def test_incompatible_saved_and_query_date_bounds_need_user_choice(deps):
    stub(deps, SearchPlan(constraints=SearchConstraints(available_to='2026-11-01')))
    req = request(constraints={'available_from': '2026-12-01'})
    result = run(deps, req)
    assert result.status == 'needs_info' and result.constraint_conflict
    result = run(deps, req.model_copy(update={'constraint_resolution': 'query'}))
    assert result.status in {'ok', 'no_match'}
    assert req.constraints.available_from.isoformat() == '2026-12-01'


def test_user_approved_exclusion_drops_only_conflicting_saved_filter(deps):
    stub(deps, SearchPlan(excluded_modes=['online']))
    req = request(constraints={'delivery_modes': ['online']})
    assert run(deps, req).constraint_conflict
    result = run(deps, req.model_copy(update={'constraint_resolution': 'query'}))
    assert result.status == 'ok' and all(r.delivery_mode != 'online' for r in result.recommendations)
    result = run(deps, req.model_copy(update={'constraint_resolution': 'profile'}))
    assert result.status == 'ok' and all(r.delivery_mode == 'online' for r in result.recommendations)


def test_date_window_and_location_never_admit_unknown_matches(deps):
    stub(deps)
    result = run(deps, request(constraints={'location_preference': ['جدة'],
                                           'available_from': '2026-12-01', 'available_to': '2027-01-02'}))
    assert result.status == 'ok'
    assert 'OPP-0256' in [r.opportunity_id for r in result.recommendations]
    for rec in result.recommendations:
        record = deps.repository.get_record(rec.opportunity_id)
        assert record.location == 'جدة'
        assert record.start_date >= '2026-12-01'
        assert record.end_date <= '2027-01-02'


def test_corrupt_store_does_not_claim_readiness(deps, tmp_path):
    (tmp_path / 'prepared').mkdir()
    (tmp_path / 'prepared/opportunities.json').write_text('[{"opportunity_id":"broken"}]')
    deps.bridge.data_dir = tmp_path
    checks, _ = deps.readiness()
    assert not checks['storage']


def test_rewritten_source_evidence_is_rejected(deps, tmp_path):
    from app.infrastructure.errors import DataAccessError
    data = deps.settings.data_dir
    (tmp_path / 'prepared').mkdir()
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw/raw_opportunities.csv').write_bytes((data / 'raw/raw_opportunities.csv').read_bytes())
    records = json.loads((data / 'prepared/opportunities.json').read_text())
    records[0]['sources'][0]['content'] = 'Changed source evidence'
    (tmp_path / 'prepared/opportunities.json').write_text(json.dumps(records))
    deps.bridge.data_dir = tmp_path
    with pytest.raises(DataAccessError): deps.repository.load()


def test_transient_openrouter_failure_retries_once(deps):
    calls = []
    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, json={'error': 'rate limited'})
        return httpx.Response(200, json={'choices': [{'message': {'content': SearchPlan().model_dump_json()}}]})
    deps.agent = OpenRouterAgent(deps.settings, httpx.MockTransport(handler))
    plan = asyncio.run(deps.agent.interpret('فرص مناسبة'))
    assert plan.keywords == [] and len(calls) == 2


def test_malformed_prepared_dates_are_validation_errors():
    from pydantic import ValidationError
    from app.internal import OpportunityRecord
    with pytest.raises(ValidationError):
        OpportunityRecord.model_validate({'opportunity_id': 'bad', 'title': 'bad', 'source_ids': ['src'],
            'opportunity_type': 'workshop', 'delivery_mode': 'online', 'start_date': 'not-a-date',
            'sources': [{'source_id': 'src', 'source_name': 'source', 'published_at': NOW, 'content': 'evidence'}]})


@pytest.mark.parametrize('term', ['Python', 'SQL', 'تصميم', 'المحاسبة', 'خدمة العملاء'])
def test_full_dataset_search_filter_and_ranking_smoke(deps, term):
    from app.tools import rank_candidates
    from app.validation import check_candidate
    from app.schemas import UserProfile, Skill
    profile = UserProfile(skills=[Skill(name=s) for s in ['Python','SQL','Excel','Power BI']])
    plan = SearchPlan(keywords=[term])
    candidates = deps.repository.search(plan)
    assert len(deps.repository.load()) == 1000 and len(candidates) > 3
    accepted = [r for r in candidates if check_candidate(r, plan.constraints, plan, profile, NOW).accepted]
    selected = rank_candidates(accepted, plan, profile)
    assert len(selected) == 3
    assert [r.opportunity_id for r in selected] == [r.opportunity_id for r in rank_candidates(accepted, plan, profile)]
    assert all(any(contains(s.content, term) for s in r.sources) for r in selected)


def test_full_dataset_recommendation_api_respects_three_result_limit(deps):
    plan = SearchPlan(keywords=['SQL'])
    stub(deps, plan)
    app.dependency_overrides[get_app_dependencies] = lambda: deps
    try:
        with TestClient(app) as client:
            result = client.post('/api/v1/discover', json=request(query='فرص SQL', profile={
                'skills':[{'name':'SQL'}]}).model_dump(mode='json'))
            assert result.status_code == 200
            body = result.json()
            assert body['status'] == 'ok' and len(body['recommendations']) == 3
            for rec in body['recommendations']:
                details = client.get('/api/v1/opportunities/' + rec['opportunity_id'])
                assert details.status_code == 200
                assert details.json()['evidence_version'] == rec['evidence_version']
    finally:
        app.dependency_overrides.clear()
