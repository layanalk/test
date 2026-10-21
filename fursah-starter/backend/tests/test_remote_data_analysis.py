"""Actual prepared-data regressions; only query interpretation is mocked."""
import pytest

from app.alternatives import select_similar
from app.explanations import build_explanations, build_match_summary
from app.internal import SearchPlan
from app.repository import contains, normalize, topic_matches
from app.schemas import SearchConstraints, UserProfile
from app.scoring import match_percentage, score_and_rank
from app.validation import check_candidate, constraint_issues, validate_explanations
from test_mvp import NOW, deps, request, run, stub


def workshop_plan(**constraints):
    return SearchPlan(keywords=['تحليل البيانات'], constraints=SearchConstraints(
        opportunity_types=['workshop'], **constraints))


def test_actual_power_bi_records_and_literal_retrieval_gap(deps):
    records = {r.opportunity_id: r for r in deps.repository.load()}
    workshop = records['OPP-0738']
    assert normalize(workshop.title) == 'لوحات power bi خطوة بخطوة'
    assert workshop.topics == ['لوحات Power BI خطوة بخطوة']
    assert workshop.opportunity_type == 'workshop'
    assert workshop.delivery_mode == 'online' and workshop.location is None
    assert workshop.remote_participation_allowed is True
    assert workshop.physical_attendance_required is False
    assert workshop.deadline == '2026-11-01'
    assert workshop.start_date == workshop.end_date == '2027-01-03'
    assert workshop.application_status == 'requires_verification'
    text = workshop.title + ' ' + ' '.join(workshop.topics) + ' ' + workshop.description
    assert not contains(text, 'تحليل البيانات')  # The old literal-only search lost it here.
    assert topic_matches(text, 'تحليل البيانات')
    hits = {r.opportunity_id for r in deps.repository.search(workshop_plan())}
    assert {'OPP-0738', 'OPP-0320', 'OPP-0321'} <= hits
    for key in ['OPP-0320', 'OPP-0321']:
        assert records[key].opportunity_type == 'course'
        assert records[key].location is None and records[key].delivery_mode == 'online'


@pytest.mark.parametrize('term', ['تحليل البيانات', 'تحليل بيانات', 'data analysis', 'DATA ANALYTICS', 'تَحْلِيل البَيَانَات'])
def test_subject_vocabulary_is_directional_bounded_and_not_a_type_or_mode(term):
    assert topic_matches('لوحات Power BI خطوة بخطوة', term)
    assert not topic_matches('دورة إدارة المشاريع', term)
    assert not topic_matches('Power BING', term)
    assert not topic_matches('تحليل البيانات باستخدام Excel', 'Power BI')
    assert not topic_matches('Power BI', 'ورشة')
    assert not topic_matches('Power BI', 'عن بعد')


def test_actual_dataset_switch_to_remote_reaches_top_one_with_verified_evidence(deps):
    plan = workshop_plan(location_preference=['الرياض'])
    calls = stub(deps, plan)
    original = request(query='ورشة في تحليل البيانات بالرياض')
    first = run(deps, original)
    assert first.status == 'no_match' and not first.recommendations
    remote = original.model_copy(update={'constraints': SearchConstraints(delivery_modes=['online'])})
    result = run(deps, remote)
    assert result.status == 'ok' and not result.similar_alternatives
    assert [(r.opportunity_id, r.relevance_score) for r in result.recommendations] == [('OPP-0738', 100)]
    assert len(calls) == 2  # One interpretation for each changed search, no explanation call.
    assert plan.constraints.location_preference == ['الرياض']  # Saved preference is not destroyed.
    assert any('عند الحاجة إلى حضور فعلي فقط' in f for f in result.interpreted_filters)
    rec = result.recommendations[0]
    assert rec.eligibility_status == 'uncertain' and rec.caveats
    assert 'يرتبط بموضوع بحثك: تحليل البيانات' in rec.match_summary
    assert 'عن بعد' in rec.match_summary and 'الموقع الذي اخترته' not in rec.match_summary
    record = deps.repository.get_record(rec.opportunity_id)
    quote = rec.reason.split('«', 1)[1].rsplit('»', 1)[0]
    assert any(quote in source.content for source in record.sources)


def test_all_actual_eligible_records_are_ranked_and_course_type_remains_hard(deps):
    plan = workshop_plan()
    constraints = SearchConstraints(location_preference=['الرياض'], delivery_modes=['online'], opportunity_types=['workshop'])
    records = deps.repository.search(plan)
    eligible = [r for r in records if check_candidate(r, constraints, plan, UserProfile(), NOW).accepted]
    assert [(r.opportunity_id, score) for r, score in score_and_rank(eligible, plan, UserProfile(), constraints)] == [('OPP-0738', 100)]
    courses = [r for r in records if r.opportunity_id in {'OPP-0320', 'OPP-0321'}]
    for record in courses:
        assert [i.field for i in constraint_issues(record, constraints)] == ['opportunity_types']
        assert not check_candidate(record, constraints, plan, UserProfile(), NOW).accepted
    alternatives = select_similar(courses, plan, constraints, UserProfile(), NOW)
    assert {r.opportunity_id for r, _, _ in alternatives} == {'OPP-0320', 'OPP-0321'}
    assert all([i.field for i in issues] == ['opportunity_types'] for _, _, issues in alternatives)
    batch = build_explanations(plan, courses)
    assert set(validate_explanations(batch, courses)) == {'OPP-0320', 'OPP-0321'}


def test_courses_are_only_labeled_alternatives_when_no_exact_workshop_exists(deps, monkeypatch):
    courses = [r for r in deps.repository.load() if r.opportunity_id in {'OPP-0320', 'OPP-0321'}]
    monkeypatch.setattr(deps.repository, 'load', lambda: courses)
    calls = stub(deps, workshop_plan(location_preference=['الرياض']))
    result = run(deps, request(query='ورشة في تحليل البيانات بالرياض', constraints={'delivery_modes': ['online']}))
    assert result.status == 'no_match' and not result.recommendations and len(calls) == 1
    assert {r.opportunity_id for r in result.similar_alternatives} == {'OPP-0320', 'OPP-0321'}
    assert all([i.field for i in r.failed_constraints] == ['opportunity_types'] for r in result.similar_alternatives)
    assert all(r.relevance_score is None for r in result.similar_alternatives)


@pytest.mark.parametrize('change', [
    {'delivery_mode': 'hybrid'}, {'delivery_mode': None},
    {'physical_attendance_required': True}, {'physical_attendance_required': None},
    {'remote_participation_allowed': False}, {'remote_participation_allowed': None},
    {'conflicting_fields': ['delivery_mode']},
])
def test_unknown_or_nonremote_attendance_does_not_bypass_city(deps, change):
    record = deps.repository.get_record('OPP-0738').model_copy(update=change)
    constraints = SearchConstraints(location_preference=['الرياض'], delivery_modes=['online'])
    assert 'location_preference' in [i.field for i in constraint_issues(record, constraints)]


def test_city_is_still_hard_without_explicit_remote_switch_and_for_mixed_modes(deps):
    record = deps.repository.get_record('OPP-0738')
    for modes in [[], ['online', 'in_person']]:
        constraints = SearchConstraints(location_preference=['الرياض'], delivery_modes=modes)
        assert [i.field for i in constraint_issues(record, constraints)] == ['location_preference']
    # An inapplicable physical city earns no invented matching credit.
    constraints = SearchConstraints(location_preference=['الرياض'], delivery_modes=['online'])
    without_city = constraints.model_copy(update={'location_preference': None})
    assert match_percentage(record, workshop_plan(), UserProfile(), constraints) == match_percentage(record, workshop_plan(), UserProfile(), without_city)


def test_subject_exclusions_also_cover_related_power_bi(deps):
    hits = deps.repository.search(SearchPlan(excluded_keywords=['تحليل البيانات']))
    assert not any('Power BI' in r.title for r in hits)
    assert not build_match_summary(deps.repository.get_record('OPP-0738'), SearchPlan(keywords=['SQL']), SearchConstraints(), UserProfile()).startswith('يطابق موضوع بحثك')
