"""Deterministic coverage of supplied criteria, not an eligibility probability."""
from fractions import Fraction

from .repository import contains, normalize
from .validation import constraint_issues


def _terms(values):
    # Repeated spellings/case variants and blank input must not inflate coverage.
    return sorted({normalize(value.strip()) for value in values if normalize(value.strip())})


def match_percentage(record, plan, profile, constraints) -> int:
    text = record.title + ' ' + ' '.join(record.topics) + ' ' + (record.description or '')
    factors = []
    for terms, weight in ((_terms(plan.keywords), 60), (_terms(profile.interests), 25)):
        if terms:
            factors.append((Fraction(sum(contains(text, term) for term in terms), len(terms)), weight))
    supplied = [field for field, value in constraints.model_dump().items()
                if value is not None and value != []]
    if supplied:
        failed = {issue.field for issue in constraint_issues(record, constraints)}
        factors.append((Fraction(len(supplied) - len(failed), len(supplied)), 15))
    if not factors:
        return 0
    value = 100 * sum(coverage * weight for coverage, weight in factors) / sum(weight for _, weight in factors)
    # Exact rational arithmetic, rounded half up. This FINAL percentage is also
    # the sorting key: do not rank by a hidden unrounded or legacy raw score.
    return int(value + Fraction(1, 2))


def score_and_rank(records, plan, profile, constraints):
    # Materialize ALL percentages before sorting or applying any page limit.
    scored = [(record, match_percentage(record, plan, profile, constraints)) for record in records]
    return sorted(scored, key=lambda item: (-item[1], item[0].opportunity_id))
