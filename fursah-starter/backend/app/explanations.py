"""Deterministic source excerpts; no model-generated identifiers or evidence."""
from .infrastructure.errors import EvidenceValidationError
from .internal import EvidenceExplanation, ExplanationBatch, OpportunityRecord, SearchPlan
from .repository import contains, normalize, topic_matches
from datetime import date


def build_explanations(plan: SearchPlan, records: list[OpportunityRecord]) -> ExplanationBatch:
    explanations = []
    for record in records:
        # Overlapping windows preserve the original characters, including Arabic
        # and newlines. Normalized search offsets must never index raw evidence.
        passages = [
            source.content[start:start + 160].strip()
            for source in record.sources
            for start in range(0, len(source.content), 80)
            if len(source.content[start:start + 160].strip()) >= 8
        ]
        if not passages:
            raise EvidenceValidationError()
        # Prefer query subject evidence; stable source/window order breaks ties.
        quote = max(passages, key=lambda text: sum(topic_matches(text, term) for term in plan.keywords))
        explanations.append(EvidenceExplanation(opportunity_id=record.opportunity_id, quotes=[quote]))
    return ExplanationBatch(explanations=explanations)


def build_match_summary(record, plan, constraints, profile) -> str:
    """Describe only proven positive matches; never infer overall eligibility."""
    text = record.title + ' ' + ' '.join(record.topics) + ' ' + (record.description or '')
    clauses = []
    subjects = [term for term in plan.keywords if topic_matches(text, term)][:2]
    interests = [term for term in profile.interests if topic_matches(text, term)][:2]
    if subjects:
        label = 'يطابق موضوع بحثك: ' if all(contains(text, term) for term in subjects) else 'يرتبط بموضوع بحثك: '
        clauses.append(label + '، '.join(subjects))
    if interests:
        clauses.append('يرتبط باهتماماتك: ' + '، '.join(interests))
    modes = {'online': 'عن بعد', 'in_person': 'حضوري', 'hybrid': 'مدمج'}
    if constraints.delivery_modes and record.delivery_mode in constraints.delivery_modes:
        clauses.append('طريقة المشاركة المطلوبة: ' + modes[record.delivery_mode])
    if (constraints.location_preference and record.location and
            normalize(record.location) in {normalize(v) for v in constraints.location_preference}):
        clauses.append('في الموقع الذي اخترته: ' + record.location)
    if constraints.max_budget_sar is not None and record.cost is not None and record.cost <= constraints.max_budget_sar:
        clauses.append(f'التكلفة المعلنة {record.cost:g} ريال ضمن ميزانيتك')
    if constraints.max_duration_days is not None and record.duration is not None and record.duration <= constraints.max_duration_days:
        clauses.append(f'المدة المعلنة {record.duration} يوم ضمن حدك')
    types = {'workshop': 'ورشة', 'course': 'دورة', 'internship': 'تدريب',
             'competition': 'مسابقة', 'volunteering': 'تطوع', 'event': 'فعالية'}
    if constraints.opportunity_types and record.opportunity_type in constraints.opportunity_types:
        clauses.append('من نوع الفرص الذي اخترته: ' + types[record.opportunity_type])
    if record.required_skills and profile.skills is not None:
        known = {normalize(skill.name) for skill in profile.skills}
        if all(normalize(skill) in known for skill in record.required_skills):
            clauses.append('المهارات المطلوبة مذكورة في ملفك')
    if constraints.available_from and record.start_date and date.fromisoformat(record.start_date) >= constraints.available_from:
        clauses.append('تبدأ بعد بداية توفرك أو في يومها')
    if constraints.available_to and record.end_date and date.fromisoformat(record.end_date) <= constraints.available_to:
        clauses.append('تنتهي قبل نهاية توفرك أو في يومها')
    return '؛ '.join(clauses[:5]) + '.' if clauses else 'نتيجة للبحث العام؛ لا توجد مطابقة شخصية محددة يمكن إثباتها من تفضيلاتك.'
