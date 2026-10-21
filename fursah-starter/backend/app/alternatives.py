"""Small deterministic near misses, explicitly separate from strict matches."""
from .repository import normalize, topic_matches
from .schemas import SearchConstraints
from .tools import rank_candidates
from .validation import check_candidate, constraint_issues


def select_similar(records, plan, constraints, profile, evaluation_time):
    ranked = rank_candidates(records, plan, profile, limit=None)
    choices = []
    for position, record in enumerate(ranked):
        # Preserve exclusions, registration/deadline safety and required skills.
        checked = check_candidate(record, SearchConstraints(), plan, profile, evaluation_time)
        if not checked.accepted:
            continue
        text = record.title + ' ' + ' '.join(record.topics) + ' ' + (record.description or '')
        if any(topic_matches(text, term) for term in plan.excluded_keywords):
            continue
        terms = plan.keywords or profile.interests
        if terms:
            meaningful = any(topic_matches(text, term) for term in terms)
        else:
            meaningful = (
                bool(constraints.opportunity_types and record.opportunity_type in constraints.opportunity_types)
                or bool(constraints.delivery_modes and record.delivery_mode in constraints.delivery_modes)
                or bool(constraints.location_preference and record.location and
                        normalize(record.location) in {normalize(v) for v in constraints.location_preference})
            )
        issues = constraint_issues(record, constraints)
        # More than two failed preferences is too remote for the course MVP.
        if meaningful and 1 <= len(issues) <= 2:
            choices.append((len(issues), position, record, checked, issues))
    choices.sort(key=lambda item: (item[0], item[1]))
    return [(record, checked, issues) for _, _, record, checked, issues in choices[:3]]
