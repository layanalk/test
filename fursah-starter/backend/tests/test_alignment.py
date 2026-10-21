"""MVP alignment through the real graph/API with offline model responses."""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.batching import SearchBatchCache
from app.explanations import build_match_summary
from app.infrastructure.dependencies import get_app_dependencies
from app.infrastructure.errors import BatchExpiredError, EvidenceValidationError
from app.internal import OpportunityRecord, SearchPlan
from app.main import app
from app.schemas import DiscoveryRequest, DiscoveryResponse, SearchConstraints, UserProfile
from app.tools import rank_candidates
from app.validation import check_candidate, constraint_issues
from test_mvp import NOW, deps, request, run, stub


def opportunity(i, **overrides):
    data = {
        'opportunity_id': f'TEST-{i:03}', 'evidence_version': f'version-{i}',
        'title': f'Python data analysis workshop {i}', 'topics': ['Python', 'data analysis'],
        'description': 'Python data analysis workshop from a fictional test announcement.',
        'opportunity_type': 'workshop', 'delivery_mode': 'online', 'location': 'الرياض',
        'cost': 50, 'duration': 2, 'start_date': '2026-12-05', 'end_date': '2026-12-06',
        'deadline': '2026-12-01', 'application_status': 'open',
        'remote_participation_allowed': True, 'physical_attendance_required': False,
        'required_skills': [], 'eligibility_requirements': {'mandatory': [], 'preferred': []},
        'source_ids': [f'SRC-TEST-{i}'],
        'sources': [{'source_id': f'SRC-TEST-{i}', 'source_name': 'Test publisher',
                     'published_at': '2026-10-01T00:00:00+00:00',
                     'content': f'Original fictional Python data analysis announcement number {i}. Cost and dates are test fixtures.'}],
        **overrides,
    }
    return OpportunityRecord.model_validate(data)


def supply(deps, monkeypatch, records):
    # Exercise actual repository lexical search against controlled local fixtures.
    monkeypatch.setattr(deps.repository, 'load', lambda: records)


@pytest.mark.parametrize('count', [1, 2, 3])
def test_normal_counts_include_factual_summary_and_independent_quote(deps, monkeypatch, count):
    records = [opportunity(i) for i in range(count)]
    supply(deps, monkeypatch, records)
    calls = stub(deps, SearchPlan(keywords=['Python']))
    result = run(deps, request(profile={'interests': ['data analysis']},
                  constraints={'delivery_modes': ['online'], 'max_budget_sar': 100}))
    assert result.status == 'ok' and len(result.recommendations) == count and len(calls) == 1
    assert not result.similar_alternatives and not result.has_more
    for rec in result.recommendations:
        assert all(term in rec.match_summary for term in ['Python', 'data analysis', 'عن بعد', '50 ريال'])
        quote = rec.reason.split('«', 1)[1].rsplit('»', 1)[0]
        original = next(r for r in records if r.opportunity_id == rec.opportunity_id)
        assert quote in original.sources[0].content


def test_next_previous_batches_are_stable_unique_and_make_no_more_llm_calls(deps, monkeypatch):
    records = [opportunity(i) for i in reversed(range(8))]
    supply(deps, monkeypatch, records)
    plan = SearchPlan(keywords=['Python'])
    calls = stub(deps, plan)
    req = request()
    first = run(deps, req)
    assert first.batch_token and first.has_more and first.batch_index == 0
    monkeypatch.delenv('OPENROUTER_API_KEY')  # Navigation must not even need a model credential.
    second = run(deps, req.model_copy(update={'batch_token': first.batch_token, 'batch_index': 1}))
    records.reverse()  # Storage iteration order must not change batch ordering.
    third = run(deps, req.model_copy(update={'batch_token': first.batch_token, 'batch_index': 2}))
    previous = run(deps, req.model_copy(update={'batch_token': first.batch_token, 'batch_index': 0}))
    ids = [rec.opportunity_id for page in (first, second, third) for rec in page.recommendations]
    expected = [r.opportunity_id for r in rank_candidates(records, plan, req.profile, limit=None)]
    assert ids == expected and len(set(ids)) == 8
    assert [len(page.recommendations) for page in (first, second, third)] == [3, 3, 2]
    assert not third.has_more and third.batch_index == 2
    assert previous.recommendations == first.recommendations and len(calls) == 1


@pytest.mark.parametrize('changed', ['query', 'profile', 'constraints', 'constraint_resolution'])
def test_batch_token_cannot_reuse_a_different_search(deps, monkeypatch, changed):
    supply(deps, monkeypatch, [opportunity(i) for i in range(5)])
    calls = stub(deps)
    req = request()
    first = run(deps, req)
    data = req.model_dump(mode='json')
    data.update(batch_token=first.batch_token, batch_index=1)
    data[changed] = {'query': 'other query', 'profile': {'interests': ['different']},
                     'constraints': {'max_budget_sar': 500}, 'constraint_resolution': 'query'}[changed]
    with pytest.raises(BatchExpiredError):
        run(deps, DiscoveryRequest.model_validate(data))
    assert len(calls) == 1


@pytest.mark.parametrize('change', ['evidence', 'cohort', 'deadline', 'out_of_range'])
def test_changed_or_invalid_batches_fail_without_reordering_or_llm(deps, monkeypatch, change):
    records = [opportunity(i) for i in range(5)]
    supply(deps, monkeypatch, records)
    calls = stub(deps)
    req = request()
    first = run(deps, req)
    if change == 'evidence': records[0].evidence_version = 'changed-version'
    if change == 'cohort': records.pop(0)
    if change == 'deadline': deps.evaluation_time = NOW + timedelta(days=60)
    with pytest.raises(BatchExpiredError):
        run(deps, req.model_copy(update={'batch_token': first.batch_token,
                                        'batch_index': 99 if change == 'out_of_range' else 1}))
    assert len(calls) == 1


def test_cache_is_bounded_expires_and_does_not_extend_on_read(deps, monkeypatch):
    clock = [0]
    deps.batches = SearchBatchCache(capacity=2, ttl_seconds=5, clock=lambda: clock[0])
    supply(deps, monkeypatch, [opportunity(i) for i in range(5)])
    calls = stub(deps)
    req = request()
    tokens = [run(deps, req).batch_token for _ in range(3)]
    with pytest.raises(BatchExpiredError):
        run(deps, req.model_copy(update={'batch_token': tokens[0], 'batch_index': 1}))
    clock[0] = 4
    run(deps, req.model_copy(update={'batch_token': tokens[1], 'batch_index': 1}))
    clock[0] = 5
    with pytest.raises(BatchExpiredError):
        run(deps, req.model_copy(update={'batch_token': tokens[1], 'batch_index': 1}))
    assert len(calls) == 3


@pytest.mark.parametrize('choice,city', [('query', 'جدة'), ('profile', 'الرياض')])
def test_conflict_resolution_is_retained_during_pagination(deps, monkeypatch, choice, city):
    supply(deps, monkeypatch, [opportunity(i, location='جدة' if i < 5 else 'الرياض') for i in range(10)])
    calls = stub(deps, SearchPlan(constraints=SearchConstraints(location_preference=['جدة'])))
    req = request(query='فرص في جدة', constraints={'location_preference': ['الرياض']})
    conflict = run(deps, req)
    assert conflict.status == 'needs_info' and conflict.constraint_conflict
    chosen = req.model_copy(update={'constraint_resolution': choice})
    first = run(deps, chosen)
    second = run(deps, chosen.model_copy(update={'batch_token': first.batch_token, 'batch_index': 1}))
    assert len(second.recommendations) == 2 and all(r.location == city for r in second.recommendations)
    assert len(calls) == 2  # Conflict submission and approved resubmission, not navigation.


def test_similar_results_are_separate_and_expose_every_failed_preference(deps, monkeypatch):
    records = [opportunity(3, cost=120, location='جدة'), opportunity(2, cost=None),
               opportunity(1, cost=120), opportunity(4, cost=300)]
    supply(deps, monkeypatch, records)
    calls = stub(deps, SearchPlan(keywords=['Python']))
    result = run(deps, request(constraints={'max_budget_sar': 100, 'location_preference': ['الرياض']}))
    assert result.status == 'no_match' and not result.recommendations
    assert len(result.similar_alternatives) == 3 and len(calls) == 1
    assert [r.opportunity_id for r in result.similar_alternatives] == ['TEST-001', 'TEST-002', 'TEST-004']
    assert result.similar_alternatives[1].failed_constraints[0].status == 'unknown'
    for alternative in result.similar_alternatives:
        assert alternative.eligibility_status == 'uncertain'
        assert 'ضمن ميزانيتك' not in alternative.match_summary
        assert {i.field for i in alternative.failed_constraints} == {'max_budget_sar'}
    # A two-failure alternative is still honest when it is the only near miss.
    supply(deps, monkeypatch, records[:1])
    again = run(deps, request(constraints={'max_budget_sar': 100, 'location_preference': ['الرياض']}))
    assert {i.field for i in again.similar_alternatives[0].failed_constraints} == {'max_budget_sar', 'location_preference'}


@pytest.mark.parametrize('reason', ['unrelated', 'expired', 'closed', 'excluded', 'skill_missing', 'attendance_conflict', 'too_many', 'no_anchor'])
def test_no_alternatives_when_unmeaningful_or_unsafe(deps, monkeypatch, reason):
    record = opportunity(1, cost=120)
    plan = SearchPlan(keywords=['Python'])
    constraints = {'max_budget_sar': 100}
    if reason == 'unrelated': plan.keywords = ['unrelated subject']
    elif reason == 'expired': record.deadline = '2026-01-01'
    elif reason == 'closed': record.application_status = 'closed'
    elif reason == 'excluded': plan.excluded_modes = ['online']
    elif reason == 'skill_missing': record.required_skills = ['SQL']
    elif reason == 'attendance_conflict': record.physical_attendance_required = True
    elif reason == 'too_many': constraints.update(location_preference=['جدة'], max_duration_days=1)
    elif reason == 'no_anchor': plan.keywords = []
    supply(deps, monkeypatch, [record])
    stub(deps, plan)
    result = run(deps, request(constraints=constraints, profile={'skills': []}))
    assert result.status == 'no_match' and not result.similar_alternatives


@pytest.mark.parametrize('anchor', ['interest', 'type', 'mode', 'location'])
def test_broad_queries_need_a_real_interest_or_preference_anchor(deps, monkeypatch, anchor):
    supply(deps, monkeypatch, [opportunity(1, cost=120)])
    stub(deps)
    options = {'constraints': {'max_budget_sar': 100}}
    if anchor == 'interest': options['profile'] = {'interests': ['Python']}
    else:
        field, value = {'type': ('opportunity_types', ['workshop']), 'mode': ('delivery_modes', ['online']),
                        'location': ('location_preference', ['الرياض'])}[anchor]
        options['constraints'][field] = value
    assert len(run(deps, request(**options)).similar_alternatives) == 1


def test_needs_info_is_not_replaced_by_alternatives(deps, monkeypatch):
    supply(deps, monkeypatch, [opportunity(1, required_skills=['Python'])])
    calls = stub(deps, SearchPlan(keywords=['Python']))
    result = run(deps)
    assert result.status == 'needs_info' and result.missing_fields == ['profile.skills']
    assert not result.similar_alternatives and len(calls) == 1


def test_two_strict_matches_are_not_padded_with_near_misses(deps, monkeypatch):
    supply(deps, monkeypatch, [opportunity(1), opportunity(2), opportunity(3, cost=120)])
    calls = stub(deps, SearchPlan(keywords=['Python']))
    result = run(deps, request(constraints={'max_budget_sar': 100}))
    assert result.status == 'ok' and len(result.recommendations) == 2
    assert not result.similar_alternatives and len(calls) == 1


def test_summary_does_not_claim_unknown_unrequested_or_failed_facts():
    record = opportunity(1, cost=None, duration=None, location=None, delivery_mode=None,
                         start_date=None, end_date=None, required_skills=None)
    constraints = SearchConstraints(max_budget_sar=100, max_duration_days=3,
        delivery_modes=['online'], location_preference=['الرياض'], available_from='2026-12-01', available_to='2026-12-31')
    profile = UserProfile(interests=['unrelated'], city='الرياض', education_level='bachelor')
    summary = build_match_summary(record, SearchPlan(keywords=['Python']), constraints, profile)
    assert summary == 'يطابق موضوع بحثك: Python.'
    assert summary == build_match_summary(record, SearchPlan(keywords=['Python']), constraints, profile)
    broad = build_match_summary(opportunity(2), SearchPlan(), SearchConstraints(), UserProfile())
    assert broad.startswith('نتيجة للبحث العام') and 'ميزانيتك' not in broad and 'عن بعد' not in broad


def test_summary_does_not_claim_known_mismatches_or_unmet_skills():
    record = opportunity(1, cost=120, required_skills=['SQL'])
    constraints = SearchConstraints(
        max_budget_sar=100, max_duration_days=1, delivery_modes=['in_person'],
        opportunity_types=['course'], location_preference=['جدة'], available_from='2026-12-10')
    profile = UserProfile(interests=['unrelated'], skills=[{'name': 'Python'}])
    assert build_match_summary(record, SearchPlan(keywords=['Python']), constraints, profile) == 'يطابق موضوع بحثك: Python.'
    constraints = SearchConstraints(available_to='2026-12-01')
    assert build_match_summary(record, SearchPlan(keywords=['Python']), constraints, profile) == 'يطابق موضوع بحثك: Python.'


def test_shared_constraint_reporting_agrees_with_strict_filtering():
    record = opportunity(1, cost=None, duration=None, location=None, opportunity_type=None,
                         delivery_mode=None, start_date=None, end_date=None)
    constraints = SearchConstraints(max_budget_sar=100, max_duration_days=3, location_preference=['جدة'],
        opportunity_types=['workshop'], delivery_modes=['online'], available_from='2026-12-01', available_to='2026-12-31')
    issues = constraint_issues(record, constraints)
    assert len(issues) == 7 and {issue.status for issue in issues} == {'unknown'}
    assert not check_candidate(record, constraints, SearchPlan(), UserProfile(), NOW).accepted


def test_alternative_quotes_still_pass_through_strict_evidence_verifier(deps, monkeypatch):
    from app import workflow
    from app.internal import ExplanationBatch
    supply(deps, monkeypatch, [opportunity(1, cost=120)])
    calls = stub(deps, SearchPlan(keywords=['Python']))
    monkeypatch.setattr(workflow, 'build_explanations', lambda plan, records: ExplanationBatch.model_validate({
        'explanations': [{'opportunity_id': records[0].opportunity_id, 'quotes': ['Hallucinated source statement.']}]}))
    with pytest.raises(EvidenceValidationError):
        run(deps, request(constraints={'max_budget_sar': 100}))
    assert len(calls) == 1


def test_api_new_contract_pagination_and_expired_sessions(deps, monkeypatch):
    supply(deps, monkeypatch, [opportunity(i) for i in range(5)])
    calls = stub(deps)
    payload = request().model_dump(mode='json')
    app.dependency_overrides[get_app_dependencies] = lambda: deps
    try:
        with TestClient(app) as client:
            first = client.post('/api/v1/discover', json=payload)
            assert first.status_code == 200
            body = DiscoveryResponse.model_validate(first.json())
            second = client.post('/api/v1/discover', json={**payload, 'batch_token': body.batch_token, 'batch_index': 1})
            assert second.status_code == 200 and len(second.json()['recommendations']) == 2
            assert second.json()['recommendations'][0]['match_summary']
            bad = client.post('/api/v1/discover', json={**payload, 'batch_token': 'unknown-session-token', 'batch_index': 1})
            assert bad.status_code == 409 and bad.json()['error_code'] == 'FURSAH_BATCH_EXPIRED'
            assert client.post('/api/v1/discover', json={**payload, 'batch_index': 1}).status_code == 422
            alternatives = client.post('/api/v1/discover', json={**payload, 'constraints': {
                'max_budget_sar': 0, 'opportunity_types': ['workshop']}})
            assert alternatives.status_code == 200
            parsed = DiscoveryResponse.model_validate(alternatives.json())
            assert parsed.status == 'no_match' and not parsed.recommendations and len(parsed.similar_alternatives) == 3
            assert all(item.failed_constraints[0].field == 'max_budget_sar' for item in parsed.similar_alternatives)
        assert len(calls) == 2
    finally:
        app.dependency_overrides.clear()


def test_response_rejects_alternatives_mixed_into_success_and_missing_failure_labels():
    from app.schemas import Recommendation, SimilarAlternative
    with pytest.raises(ValidationError):
        SimilarAlternative.model_validate({'opportunity_id': 'x', 'title': 'x', 'source_ids': ['x'], 'reason': 'x'})
    with pytest.raises(ValidationError):
        DiscoveryResponse(request_id='x', status='no_match', message='x', has_more=True)
    fields = {'opportunity_id': 'x', 'title': 'x', 'source_ids': ['x'], 'reason': 'source quote', 'eligibility_status': 'uncertain'}
    alt = SimilarAlternative(**fields, failed_constraints=[{'field': 'max_budget_sar', 'status': 'unknown', 'message': 'Unknown cost'}])
    with pytest.raises(ValidationError):
        DiscoveryResponse(request_id='x', status='ok', message='x',
                          recommendations=[Recommendation(**fields)], similar_alternatives=[alt])


def test_full_dataset_supports_batches_with_existing_one_call_transport(deps):
    calls = stub(deps)
    req = request()
    first = run(deps, req)
    assert first.has_more
    second = run(deps, req.model_copy(update={'batch_token': first.batch_token, 'batch_index': 1}))
    assert {r.opportunity_id for r in first.recommendations}.isdisjoint(r.opportunity_id for r in second.recommendations)
    assert len(second.recommendations) == 3 and len(calls) == 1
