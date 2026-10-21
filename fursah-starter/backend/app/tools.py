"""Deterministic retrieval, deadline and ranking helpers used by graph nodes."""
from datetime import date, datetime
from zoneinfo import ZoneInfo
from .repository import contains
from .infrastructure.errors import RecordNotFound


def search_opportunities(criteria, repo):
    return repo.search(criteria)


def get_opportunity_details(opportunity_id, repo):
    return repo.get_details(opportunity_id)


def deadline_state(deadline: str | None, evaluation_time: datetime) -> str:
    if not deadline:
        return 'unknown'
    try:
        if len(deadline) == 10:
            # Date-only deadlines have no proven exact cutoff time or timezone.
            day = date.fromisoformat(deadline)
            today = evaluation_time.astimezone(ZoneInfo('Asia/Riyadh')).date()
            return 'expired' if day < today else ('uncertain' if day == today else 'not_passed')
        instant = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        if instant.tzinfo is None:
            return 'uncertain'
        return 'expired' if evaluation_time >= instant else 'not_passed'
    except ValueError:
        return 'uncertain'


def check_deadline(opportunity_id, repo, evaluation_time):
    record = repo.get_record(opportunity_id)
    if record is None:
        raise RecordNotFound()
    return deadline_state(record.deadline, evaluation_time)


def rank_candidates(records, plan, profile):
    # Subject match dominates; interests only break ties. Stable ordering is repeatable.
    def score(record):
        text = record.title + ' ' + ' '.join(record.topics)
        subject = sum(contains(text, term) for term in plan.keywords)
        interest = sum(contains(text, term) for term in profile.interests)
        return subject * 10 + min(interest, 5)
    return sorted(records, key=lambda record: (-score(record), record.opportunity_id))[:3]
