"""Hard constraints and eligibility are checked in code, never delegated to the LLM."""
from datetime import date
from zoneinfo import ZoneInfo
from .internal import CandidateCheck
from .repository import normalize
from .tools import deadline_state
from .infrastructure.errors import EvidenceValidationError


def check_candidate(record, constraints, plan, profile, evaluation_time):
    caveats = list(record.data_warnings)
    rejected = CandidateCheck(accepted=False)
    # An unresolved registration conflict must not reopen a closed/cancelled source.
    if 'application_status' in record.conflicting_fields:
        return rejected
    if record.application_status in {'closed', 'cancelled', 'not_yet_open', 'not_open_yet'}:
        return rejected
    state = deadline_state(record.deadline, evaluation_time)
    if state == 'expired':
        return rejected
    today = evaluation_time.astimezone(ZoneInfo('Asia/Riyadh')).date()
    if record.end_date and date.fromisoformat(record.end_date) < today:
        return rejected
    if record.start_date and date.fromisoformat(record.start_date) < today:
        caveats.append('النشاط قد بدأ؛ تحقق من إمكانية الالتحاق المتأخر مع الجهة.')
    if state in {'unknown', 'uncertain'}:
        caveats.append('موعد الإغلاق الدقيق غير مؤكد؛ تحقق من الجهة قبل التقديم.')
    if record.application_status != 'open':
        caveats.append('حالة التسجيل الحالية تحتاج إلى تحقق من الجهة؛ الإعلان لا يثبت استمرارها.')
    if record.location and any(normalize(record.location) == normalize(v) for v in plan.excluded_locations):
        return rejected
    if plan.excluded_locations and not record.location and record.delivery_mode != 'online':
        return rejected
    if record.opportunity_type in plan.excluded_types or record.delivery_mode in plan.excluded_modes:
        return rejected
    for field, value in [('opportunity_types', record.opportunity_type), ('delivery_modes', record.delivery_mode)]:
        allowed = getattr(constraints, field)
        if allowed and value not in allowed:
            return rejected
    if constraints.location_preference:
        if record.location is None or normalize(record.location) not in {normalize(v) for v in constraints.location_preference}:
            return rejected
    if constraints.max_budget_sar is not None:
        if record.cost is None or record.cost > constraints.max_budget_sar:
            return rejected
    if constraints.max_duration_days is not None:
        if record.duration is None or record.duration > constraints.max_duration_days:
            return rejected
    if constraints.available_from and (not record.start_date or date.fromisoformat(record.start_date) < constraints.available_from):
        return rejected
    if constraints.available_to and (not record.end_date or date.fromisoformat(record.end_date) > constraints.available_to):
        return rejected
    if record.delivery_mode == 'online' and (record.physical_attendance_required is True or record.remote_participation_allowed is False):
        return rejected
    if record.required_skills:
        if profile.skills is None:
            return CandidateCheck(accepted=False, missing_fields=['profile.skills'])
        known = {normalize(skill.name) for skill in profile.skills}
        if not all(normalize(skill) in known for skill in record.required_skills):
            return rejected
    # The MVP cannot establish age, study year or unstructured requirements.
    requirements = record.eligibility_requirements
    if requirements is None or requirements.mandatory is None:
        caveats.append('شروط الأهلية غير مكتملة في المصدر؛ لا يمكن تأكيد الأهلية.')
    elif requirements.mandatory:
        caveats.append('توجد شروط مصدرية إضافية تحتاج إلى تحقق؛ عرض الفرصة ليس تأكيدًا للأهلية.')
    if record.required_skills is None:
        caveats.append('متطلبات المهارات غير محددة بالكامل في المصدر.')
    # Even a current source cannot establish present registration status.
    uncertain = bool(caveats)
    return CandidateCheck(accepted=True, caveats=list(dict.fromkeys(caveats)),
                          eligibility_status='uncertain' if uncertain else 'confirmed')


def validate_explanations(batch, records):
    expected = {r.opportunity_id: r for r in records}
    returned = [item.opportunity_id for item in batch.explanations]
    if len(returned) != len(set(returned)) or set(returned) != set(expected):
        raise EvidenceValidationError()
    quotes = {}
    for item in batch.explanations:
        sources = expected[item.opportunity_id].sources
        if any(len(quote.strip()) < 8 or not any(quote in source.content for source in sources)
               for quote in item.quotes):
            raise EvidenceValidationError()
        quotes[item.opportunity_id] = item.quotes
    return quotes
