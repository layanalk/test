"""Final percentages drive ALL eligible-candidate ranking and pagination offline."""
import pytest
from fastapi.testclient import TestClient

from app import scoring
from app.infrastructure.dependencies import get_app_dependencies
from app.infrastructure.errors import BatchExpiredError
from app.internal import SearchPlan
from app.main import app
from app.schemas import SearchConstraints, UserProfile
from app.scoring import match_percentage, score_and_rank
from app.tools import rank_candidates
from test_alignment import opportunity, supply
from test_mvp import deps, request, run, stub


SUBJECTS = ['Python', 'SQL', 'Excel', 'Statistics', 'Visualization']
INTERESTS = ['Health', 'Finance', 'Design', 'Security']


def candidate(i, subjects, interests=(), **overrides):
    return opportunity(i, title=' '.join([*subjects, *interests]), topics=[],
                       description='Fictional test announcement.', **overrides)


def example_pool():
    # Highest score is LAST in retrieval order. A first-three shortcut is wrong.
    return [candidate(1, SUBJECTS[:4]), candidate(2, SUBJECTS[:4], INTERESTS[:2]),
            candidate(3, SUBJECTS[:4], INTERESTS[:3]), candidate(4, SUBJECTS, INTERESTS[:2]),
            candidate(5, SUBJECTS, INTERESTS[:3])]


def test_explicit_weighted_percentages_and_half_up_rounding():
    plan, profile = SearchPlan(keywords=SUBJECTS), UserProfile(interests=INTERESTS)
    constraints = SearchConstraints(max_budget_sar=100)
    assert [match_percentage(r, plan, profile, constraints) for r in example_pool()] == [63, 76, 82, 88, 94]
    assert [percentage for _, percentage in score_and_rank(example_pool(), plan, profile, constraints)] == [94, 88, 82, 76, 63]


def test_inactive_factors_are_excluded_and_missing_information_is_not_invented():
    record = candidate(1, ['Python'])
    assert match_percentage(record, SearchPlan(keywords=['Python', 'SQL']), UserProfile(), SearchConstraints()) == 50
    assert match_percentage(record, SearchPlan(), UserProfile(interests=['Python', 'SQL']), SearchConstraints()) == 50
    assert match_percentage(record, SearchPlan(), UserProfile(), SearchConstraints()) == 0
    assert match_percentage(record, SearchPlan(), UserProfile(), SearchConstraints(max_budget_sar=100)) == 100
    unknown = record.model_copy(update={'cost': None})
    assert match_percentage(unknown, SearchPlan(), UserProfile(), SearchConstraints(max_budget_sar=100)) == 0
    assert match_percentage(record, SearchPlan(), UserProfile(), SearchConstraints(max_budget_sar=0)) == 0


def test_each_supplied_constraint_uses_the_same_known_fact_checks_as_filtering():
    record = opportunity(1, delivery_mode='hybrid')
    constraints = SearchConstraints(location_preference=['الرياض'], delivery_modes=['hybrid'],
        opportunity_types=['workshop'], max_budget_sar=100, max_duration_days=3,
        available_from='2026-12-01', available_to='2026-12-31')
    assert match_percentage(record, SearchPlan(), UserProfile(), constraints) == 100
    fields = ['location', 'delivery_mode', 'opportunity_type', 'cost', 'duration', 'start_date', 'end_date']
    for field in fields:
        unknown = record.model_copy(update={field: None})
        assert match_percentage(unknown, SearchPlan(), UserProfile(), constraints) == 86  # 6 of 7, not 100.


def test_duplicate_blank_and_arabic_variant_terms_do_not_inflate_percentage():
    record = candidate(1, ['Python', 'البيانات'])
    plan = SearchPlan(keywords=['Python', 'SQL'])
    profile = UserProfile(interests=['البيانات', 'التصميم'])
    expected = match_percentage(record, plan, profile, SearchConstraints())
    assert expected == 50
    duplicate = SearchPlan(keywords=['Python', 'PYTHON', 'SQL', 'sql'])
    noisy = UserProfile(interests=['', ' ', 'َ', 'البَيَانَات', 'البيانات', 'التصميم'])
    assert match_percentage(record, duplicate, noisy, SearchConstraints()) == expected


def test_final_percentage_not_the_legacy_raw_score_determines_order():
    # Legacy score would prefer 5 subjects/0 interests (50) to 4/4 (44).
    more_subjects = candidate(1, SUBJECTS)
    more_coverage = candidate(2, SUBJECTS[:4], INTERESTS)
    plan, profile = SearchPlan(keywords=SUBJECTS), UserProfile(interests=INTERESTS)
    ordered = score_and_rank([more_subjects, more_coverage], plan, profile, SearchConstraints(max_budget_sar=100))
    assert [(r.opportunity_id, percent) for r, percent in ordered] == [('TEST-002', 88), ('TEST-001', 75)]


def test_ties_use_final_displayed_integer_and_id_not_hidden_fraction():
    profile = UserProfile(interests=[f'interest{i}' for i in range(300)])
    low = candidate(1, ['interest0', 'interest1', 'interest2'])  # 1.0%.
    high = candidate(2, ['interest0', 'interest1', 'interest2', 'interest3'])  # 1.333...%, also 1%.
    ordered = score_and_rank([high, low], SearchPlan(), profile, SearchConstraints())
    assert [(r.opportunity_id, percent) for r, percent in ordered] == [('TEST-001', 1), ('TEST-002', 1)]


def test_scores_are_absolute_not_normalized_against_other_candidates():
    records = example_pool()
    plan, profile, constraints = SearchPlan(keywords=SUBJECTS), UserProfile(interests=INTERESTS), SearchConstraints(max_budget_sar=100)
    by_id = {r.opportunity_id: value for r, value in score_and_rank(records, plan, profile, constraints)}
    assert score_and_rank(records[:1], plan, profile, constraints)[0][1] == by_id[records[0].opportunity_id]
    assert all(0 <= score <= 100 and isinstance(score, int) for score in by_id.values())


def test_graph_scores_every_eligible_record_before_selection_and_does_not_rescore_in_explain(deps, monkeypatch):
    records = example_pool() + [candidate(6, SUBJECTS, INTERESTS, application_status='closed')]
    supply(deps, monkeypatch, records)
    calls = stub(deps, SearchPlan(keywords=SUBJECTS))
    visited = []
    calculate = scoring.match_percentage

    def spy(record, *args):
        visited.append(record.opportunity_id)
        return calculate(record, *args)

    monkeypatch.setattr(scoring, 'match_percentage', spy)
    req = request(profile={'interests': INTERESTS}, constraints={'max_budget_sar': 100})
    result = run(deps, req)
    assert visited == [r.opportunity_id for r in records[:5]]  # Includes both non-selected eligible candidates.
    assert [(r.opportunity_id, r.relevance_score) for r in result.recommendations] == [
        ('TEST-005', 94), ('TEST-004', 88), ('TEST-003', 82)]
    assert len(calls) == 1
    assert all(r.match_summary and '«' in r.reason for r in result.recommendations)


def test_batches_follow_the_same_global_percentage_order_and_reuse_plan(deps, monkeypatch):
    records = example_pool()
    supply(deps, monkeypatch, records)
    calls = stub(deps, SearchPlan(keywords=SUBJECTS))
    req = request(profile={'interests': INTERESTS}, constraints={'max_budget_sar': 100})
    first = run(deps, req)
    monkeypatch.delenv('OPENROUTER_API_KEY')
    records.reverse()
    second = run(deps, req.model_copy(update={'batch_token': first.batch_token, 'batch_index': 1}))
    previous = run(deps, req.model_copy(update={'batch_token': first.batch_token, 'batch_index': 0}))
    combined = first.recommendations + second.recommendations
    assert [r.relevance_score for r in combined] == [94, 88, 82, 76, 63]
    assert len({r.opportunity_id for r in combined}) == 5 and not second.has_more
    assert previous.recommendations == first.recommendations and len(calls) == 1


def test_percentage_change_invalidates_session_even_if_ids_keep_their_order(deps, monkeypatch):
    supply(deps, monkeypatch, example_pool())
    calls = stub(deps, SearchPlan(keywords=SUBJECTS))
    req = request(profile={'interests': INTERESTS}, constraints={'max_budget_sar': 100})
    first = run(deps, req)
    calculate = scoring.match_percentage
    monkeypatch.setattr(scoring, 'match_percentage', lambda *args: calculate(*args) - 1)
    with pytest.raises(BatchExpiredError):
        run(deps, req.model_copy(update={'batch_token': first.batch_token, 'batch_index': 1}))
    assert len(calls) == 1


@pytest.mark.parametrize('count', [1, 2, 3])
def test_actual_api_returns_one_two_or_three_highest_percentages(deps, monkeypatch, count):
    supply(deps, monkeypatch, example_pool()[-count:])
    calls = stub(deps, SearchPlan(keywords=SUBJECTS))
    req = request(profile={'interests': INTERESTS}, constraints={'max_budget_sar': 100})
    app.dependency_overrides[get_app_dependencies] = lambda: deps
    try:
        with TestClient(app) as client:
            response = client.post('/api/v1/discover', json=req.model_dump(mode='json'))
        assert response.status_code == 200
        recommendations = response.json()['recommendations']
        scores = [r['relevance_score'] for r in recommendations]
        assert len(scores) == count and scores == [94, 88, 82][:count]
        assert all(r['match_summary'] and r['reason'] for r in recommendations)
        assert len(calls) == 1
    finally:
        app.dependency_overrides.clear()


def test_zero_eligible_keeps_labeled_alternatives_not_fake_eligible_percentages(deps, monkeypatch):
    supply(deps, monkeypatch, [opportunity(i, cost=120) for i in range(5)])
    calls = stub(deps, SearchPlan(keywords=['Python']))
    result = run(deps, request(constraints={'max_budget_sar': 100}))
    assert result.status == 'no_match' and not result.recommendations
    assert len(result.similar_alternatives) == 3
    assert all(r.failed_constraints and r.relevance_score is None for r in result.similar_alternatives)
    assert len(calls) == 1


def test_rank_helper_also_scores_before_applying_limit():
    records = example_pool()
    result = rank_candidates(records, SearchPlan(keywords=SUBJECTS), UserProfile(interests=INTERESTS),
                             constraints=SearchConstraints(max_budget_sar=100))
    assert [r.opportunity_id for r in result] == ['TEST-005', 'TEST-004', 'TEST-003']
