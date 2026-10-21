"""One LangGraph workflow and one main AI agent; user decisions use API resubmission."""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TypedDict
from uuid import uuid4
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError
from .alternatives import select_similar
from .batching import SearchSession
from .explanations import build_explanations, build_match_summary
from .infrastructure.dependencies import AppDependencies
from .infrastructure.errors import BatchExpiredError, WorkflowBudgetExceededError
from .internal import CandidateCheck, OpportunityRecord, SearchPlan
from .repository import normalize
from .schemas import ConstraintIssue, DiscoveryRequest, DiscoveryResponse, Recommendation, SearchConstraints, SimilarAlternative
from .tools import rank_candidates, search_opportunities
from .validation import check_candidate, validate_explanations


class DiscoveryState(TypedDict, total=False):
    request: DiscoveryRequest
    deps: AppDependencies
    plan: SearchPlan
    constraints: SearchConstraints
    filters: list[str]
    candidates: list[OpportunityRecord]
    checks: dict[str, CandidateCheck]
    selected: list[OpportunityRecord]
    response: DiscoveryResponse
    batch_session: SearchSession
    batch_token: str | None
    has_more: bool
    similar_issues: dict[str, list[ConstraintIssue]]


def response(state, status, message, **kwargs):
    return DiscoveryResponse(request_id=state['request'].request_id, status=status,
                             message=message, interpreted_filters=state.get('filters', []), **kwargs)


async def interpret(state):
    request = state['request']
    if request.batch_token:
        session = state['deps'].batches.get(request)
        return {'plan': session.plan.model_copy(deep=True),
                'constraints': session.constraints.model_copy(deep=True),
                'filters': list(session.filters), 'batch_session': session}
    plan = await state['deps'].agent.interpret(request.query)
    effective = request.constraints.model_dump()
    conflicts = []
    # Only explicit query constraints override saved values, and only with user approval.
    for name, query_value in plan.constraints.model_dump().items():
        if query_value is None or query_value == []:
            continue
        saved = effective[name]
        differs = (set(saved) != set(query_value)) if isinstance(saved, list) and isinstance(query_value, list) else saved != query_value
        if saved is not None and saved != [] and differs:
            conflicts.append('constraints.' + name)
            if request.constraint_resolution == 'profile':
                continue
        effective[name] = query_value
    # A saved positive preference and a query exclusion also require a decision.
    exclusions = {'location_preference': plan.excluded_locations,
                  'opportunity_types': plan.excluded_types, 'delivery_modes': plan.excluded_modes}
    for field, excluded in exclusions.items():
        saved = getattr(request.constraints, field) or []
        overlap = {normalize(v) for v in saved} & {normalize(v) for v in excluded}
        if overlap:
            conflicts.append('constraints.' + field)
            if request.constraint_resolution == 'profile':
                setattr(plan, {'location_preference': 'excluded_locations', 'opportunity_types': 'excluded_types',
                               'delivery_modes': 'excluded_modes'}[field], [v for v in excluded if normalize(v) not in overlap])
            elif request.constraint_resolution == 'query':
                retained = [v for v in (effective[field] or []) if normalize(v) not in overlap]
                effective[field] = (retained or None) if field == 'location_preference' else retained
    # Individually valid query/saved windows can become invalid when combined.
    if (effective['available_from'] and effective['available_to']
            and effective['available_from'] > effective['available_to']):
        conflicts.extend(['constraints.available_from', 'constraints.available_to'])
        selected = request.constraints if request.constraint_resolution == 'profile' else plan.constraints
        effective['available_from'] = selected.available_from
        effective['available_to'] = selected.available_to
    if conflicts and request.constraint_resolution is None:
        return {'response': response(state, 'needs_info', 'توجد قيود مختلفة بين نص البحث وتفضيلاتك المحفوظة. أيهما نستخدم لهذا البحث؟',
                                     missing_fields=list(dict.fromkeys(conflicts)), constraint_conflict=True)}
    constraints = SearchConstraints.model_validate(effective)
    filters = []
    if plan.keywords:
        filters.append('الموضوع: ' + '، '.join(plan.keywords))
    labels = {'location_preference': 'الموقع', 'opportunity_types': 'نوع الفرصة', 'delivery_modes': 'طريقة المشاركة',
              'available_from': 'متاح من', 'available_to': 'متاح حتى', 'max_duration_days': 'أقصى مدة بالأيام',
              'max_budget_sar': 'أقصى ميزانية بالريال'}
    translations = {'workshop': 'ورشة', 'course': 'دورة', 'internship': 'تدريب مهني', 'competition': 'مسابقة',
                    'volunteering': 'تطوع', 'event': 'فعالية', 'online': 'عن بعد', 'in_person': 'حضوري', 'hybrid': 'مدمج'}
    for field, value in constraints.model_dump(mode='json').items():
        if value is not None and value != []:
            text = '، '.join(translations.get(v, v) for v in value) if isinstance(value, list) else str(value)
            filters.append(labels[field] + ': ' + text)
    excluded = plan.excluded_keywords + plan.excluded_locations + plan.excluded_types + plan.excluded_modes
    if excluded:
        filters.append('استبعاد: ' + '، '.join(translations.get(v, v) for v in excluded))
    return {'plan': plan, 'constraints': constraints, 'filters': filters}


def search(state):
    return {'candidates': search_opportunities(state['plan'], state['deps'].repository)}


def validate(state):
    deps = state['deps']
    checks = {record.opportunity_id: check_candidate(record, state['constraints'], state['plan'],
              state['request'].profile, deps.evaluation_time) for record in state['candidates']}
    accepted = [r for r in state['candidates'] if checks[r.opportunity_id].accepted]
    if not accepted:
        if 'batch_session' in state:
            raise BatchExpiredError()
        missing = sorted({f for check in checks.values() for f in check.missing_fields})
        if missing:
            result = response(state, 'needs_info', 'بعض الفرص المطابقة تتطلب مهارات محددة. أضف مهاراتك أو اختر صراحة أنك لا تملكها ثم أعد البحث.',
                              missing_fields=missing, missing_explanations=['لا نطلب هذه المعلومات إلا لوجود متطلب مهارة في مصدر فرصة مطابقة.'])
        else:
            similar = select_similar(state['candidates'], state['plan'], state['constraints'],
                                     state['request'].profile, deps.evaluation_time)
            if similar:
                return {'candidates': [r for r, _, _ in similar],
                        'checks': {r.opportunity_id: checked for r, checked, _ in similar},
                        'similar_issues': {r.opportunity_id: issues for r, _, issues in similar}}
            result = response(state, 'no_match', 'لم نجد فرصة مدعومة بالأدلة ضمن الإعلانات تطابق هذه القيود. يمكنك تعديل البحث؛ لم نخفف القيود تلقائيًا.')
        return {'response': result}
    return {'candidates': accepted, 'checks': checks}


def rank(state):
    if state.get('similar_issues'):
        return {'selected': state['candidates'][:3]}
    request = state['request']
    ordered = rank_candidates(state['candidates'], state['plan'], request.profile, limit=None)
    signature = tuple((r.opportunity_id, r.evidence_version or '') for r in ordered)
    if 'batch_session' in state and state['batch_session'].ordered != signature:
        # Revalidate the cohort without silently shifting page boundaries.
        raise BatchExpiredError()
    start = request.batch_index * 3
    if start >= len(ordered):
        raise BatchExpiredError()
    token = request.batch_token
    if not token and len(ordered) > 3:
        token = state['deps'].batches.create(request, state['plan'], state['constraints'], state['filters'], signature)
    return {'selected': ordered[start:start + 3], 'batch_token': token,
            'has_more': start + 3 < len(ordered)}


def explain(state):
    selected = state['selected']
    batch = build_explanations(state['plan'], selected)
    quotes = validate_explanations(batch, selected)
    recommendations = []
    for record in selected:
        checked = state['checks'][record.opportunity_id]
        # Only verified source passages enter the explanation. The agent cannot assign eligibility or IDs.
        reason = 'من نص الإعلان: ' + ' — '.join('«' + q + '»' for q in quotes[record.opportunity_id])
        issues = state.get('similar_issues', {}).get(record.opportunity_id)
        model = SimilarAlternative if issues else Recommendation
        extra = {'failed_constraints': issues} if issues else {}
        recommendations.append(model(
            opportunity_id=record.opportunity_id, evidence_version=record.evidence_version,
            title=record.title, opportunity_type=record.opportunity_type, location=record.location,
            delivery_mode=record.delivery_mode, deadline=record.deadline,
            application_status=record.application_status, eligibility_status='uncertain' if issues else checked.eligibility_status,
            source_ids=record.source_ids, reason=reason, caveats=checked.caveats,
            match_summary=build_match_summary(record, state['plan'], state['constraints'], state['request'].profile),
            next_action='راجع التفاصيل والشروط مع الجهة. بيانات وعناوين هذا التمرين خيالية.',
            **extra,
        ))
    if state.get('similar_issues'):
        return {'response': response(state, 'no_match', 'لا توجد نتائج تستوفي جميع قيودك. هذه بدائل قريبة لا تطابق بعض القيود؛ راجع الاختلافات قبل تغيير بحثك.',
                                     similar_alternatives=recommendations)}
    return {'response': response(state, 'ok', 'هذه نتائج مستندة إلى الإعلانات التعليمية؛ الشروط والمعلومات غير المؤكدة تحتاج إلى مراجعة الجهة.',
                                recommendations=recommendations, batch_token=state.get('batch_token'),
                                batch_index=state['request'].batch_index, has_more=state.get('has_more', False))}


def continue_or_end(state):
    return END if 'response' in state else 'continue'


@lru_cache(maxsize=1)
def get_graph():
    graph = StateGraph(DiscoveryState)
    graph.add_node('interpret', interpret)
    graph.add_node('search', search)
    graph.add_node('validate', validate)
    graph.add_node('rank', rank)
    graph.add_node('explain', explain)
    graph.add_edge(START, 'interpret')
    graph.add_conditional_edges('interpret', continue_or_end, {'continue': 'search', END: END})
    graph.add_edge('search', 'validate')
    graph.add_conditional_edges('validate', continue_or_end, {'continue': 'rank', END: END})
    graph.add_edge('rank', 'explain')
    graph.add_edge('explain', END)
    return graph.compile()  # No persistent checkpoints or additional agents.


def application_is_ready(deps: AppDependencies) -> bool:
    return deps.agent.is_configured() and bool(get_graph())


async def discover(request: DiscoveryRequest, deps: AppDependencies) -> DiscoveryResponse:
    if request.request_id is None:
        request = request.model_copy(update={'request_id': str(uuid4())})
    if deps.evaluation_time is None:
        from dataclasses import replace
        deps = replace(deps, evaluation_time=deps.settings.resolve_evaluation_time())
    try:
        async with asyncio.timeout(deps.settings.workflow_timeout_seconds):
            result = await get_graph().ainvoke({'request': request, 'deps': deps},
                config={'recursion_limit': deps.settings.max_workflow_steps})
        return DiscoveryResponse.model_validate(result['response'])
    except (TimeoutError, GraphRecursionError):
        raise WorkflowBudgetExceededError() from None
