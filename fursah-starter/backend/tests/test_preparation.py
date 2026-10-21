"""Corpus coverage and conservative extraction regressions for offline preparation."""
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest
from app.internal import OpportunityRecord

DATA = Path(__file__).resolve().parents[2] / 'data'


@pytest.fixture
def preparer(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.syspath_prepend(str(DATA))
    spec = importlib.util.spec_from_file_location('full_preparer', DATA / 'prepare_mvp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(text, source_id='SRC-9001'):
    return {'source_id': source_id, 'source_type': 'noticeboard', 'source_name': 'جهة تدريبية',
            'published_at': '2026-10-01T12:00:00+03:00', 'content': text}


def extract(preparer, text):
    import re
    return preparer.extract_record(source(text), re.findall('«([^»]+)»', text), [])


def test_complete_coverage_report_and_exact_evidence(preparer):
    records, report = preparer.prepare_with_report()
    raw = list(csv.DictReader((DATA / 'raw/raw_opportunities.csv').open(encoding='utf-8-sig')))
    assert report['raw_row_count'] == report['prepared_record_count'] == len(raw) == 1000
    assert report['skipped_invalid_record_count'] == 0 and report['skipped_records'] == []
    assert report['duplicate_source_id_row_count'] == report['exact_duplicate_text_count'] == 0
    assert report['shared_link_group_count'] == 100 and report['shared_link_announcement_count'] == 200
    assert report['extra_shared_link_announcements'] == 100
    assert report['conflicting_link_group_count'] == 30
    assert report['records_with_conflicting_evidence_count'] == 60
    assert report['records_with_incomplete_important_fields_count'] == 978
    assert {r['source_ids'][0] for r in records} == {r['source_id'] for r in raw}
    by_id = {r['source_id']: r for r in raw}
    for record in records:
        validated = OpportunityRecord.model_validate(record)
        assert len(validated.sources) == 1
        for evidence in record['sources'] + record['related_sources']:
            assert all(value == by_id[evidence['source_id']][key] for key, value in evidence.items())
    assert records == json.loads((DATA / 'prepared/opportunities.json').read_text(encoding='utf-8'))
    assert report == json.loads((DATA / 'prepared/preparation_report.json').read_text(encoding='utf-8'))


def test_conflicting_fees_and_dates_remain_unknown(preparer):
    records = {r['opportunity_id']: r for r in preparer.prepare()}
    for record_id in ['OPP-0032', 'OPP-0033']:
        record = records[record_id]
        assert record['cost'] is None and 'cost' in record['conflicting_fields']
        evidence = record['sources'] + record['related_sources']
        assert any('رسوم المشاركة 75' in s['content'] for s in evidence)
        assert any('رسوم المشاركة 135' in s['content'] for s in evidence)
    for record_id in ['OPP-0104', 'OPP-0105']:
        record = records[record_id]
        assert record['start_date'] is None and record['duration'] is None and record['deadline'] is None
        assert record['related_sources'] and record['conflicting_fields']
        assert any('مُدد آخر يوم للتقديم' in s['content'] for s in record['sources'] + record['related_sources'])


def test_only_explicit_source_facts_are_extracted(preparer):
    record = extract(preparer, 'ورشة تطبيقية بعنوان «تحديد السؤال». المحتوى مناسب للمبتدئين، والخبرة السابقة ليست شرطاً.')
    assert record['opportunity_type'] == 'workshop'  # تحديد is not تحدي.
    for field in ['location','delivery_mode','remote_participation_allowed','physical_attendance_required',
                  'cost','start_date','end_date','deadline','duration','required_skills','session_count']:
        assert record[field] is None
    assert record['eligibility_requirements']['mandatory'] is None
    record = extract(preparer, 'إعلان بعنوان «نشاط جديد».')
    assert record['opportunity_type'] is None and record['delivery_mode'] is None
    assert OpportunityRecord.model_validate(record).opportunity_type is None


def test_update_does_not_invent_end_date_or_duration(preparer):
    record = extract(preparer, 'تحديث من جهة تدريبية لفرصة «نشاط جديد»: تغيّر الموعد من 1 نوفمبر 2026 إلى 8 نوفمبر 2026، ومُدد آخر يوم للتقديم من 20 أكتوبر 2026 إلى 24 أكتوبر 2026.')
    assert record['start_date'] == '2026-11-08' and record['deadline'] == '2026-10-24'
    assert record['end_date'] is None and record['duration'] is None


def test_dates_and_hours_keep_separate_meanings(preparer):
    record = extract(preparer, 'دورة تدريبية بعنوان «تعلم جديد». يبدأ البرنامج يوم 1 نوفمبر 2026 وينتهي يوم 15 نوفمبر 2026. يشمل 5 لقاءات، مدة كل لقاء ساعتان. آخر يوم للتقديم 20 أكتوبر 2026.')
    assert record['duration'] == 15
    assert record['session_count'] == 5 and record['hours_per_session'] == 2 and record['total_hours'] == 10
    assert record['deadline'] == '2026-10-20'  # No invented midnight cutoff.
    assert record['cost'] is None


def test_preferred_skills_are_not_requirements(preparer):
    record = extract(preparer, 'مسابقة فرق بعنوان «Python». تُعد معرفة Python ميزة إضافية وليست شرطاً. المشاركة مجانية.')
    assert record['required_skills'] is None
    assert record['eligibility_requirements']['preferred'] == ['تُعد معرفة Python ميزة إضافية وليست شرطاً']
    assert record['cost'] == 0
    record = extract(preparer, 'ورشة عمل بعنوان «SQL». معرفة SQL مطلوبة. لم تُحدد رسوم المشاركة.')
    assert record['required_skills'] == ['SQL'] and record['cost'] is None


def test_invalid_optional_date_remains_unknown_without_losing_source(preparer):
    text = 'ورشة عمل بعنوان «نشاط جديد». يبدأ البرنامج يوم 32 أكتوبر 2026 وينتهي يوم 1 نوفمبر 2026.'
    issues = []
    record = preparer.extract_record(source(text), ['نشاط جديد'], issues)
    assert record['start_date'] is None and record['duration'] is None and issues
    assert record['sources'][0]['content'] == text


def test_invalid_rows_have_explicit_skip_reasons(preparer, tmp_path):
    rows = [source('ورشة عمل بعنوان «صالح».', 'SRC-9001'),
            source('إعلان بلا عنوان قابل للاستخلاص.', 'SRC-9002'),
            source('ورشة عمل بعنوان «توقيت ناقص».', 'SRC-9003')]
    rows[2]['published_at'] = '2026-10-01T12:00:00'
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=preparer.RAW_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    path = tmp_path / 'input.csv'
    path.write_text(stream.getvalue(), encoding='utf-8')
    records, report = preparer.prepare_with_report(path)
    assert len(records) == 1 and report['skipped_invalid_record_count'] == 2
    assert {item['source_id'] for item in report['skipped_records']} == {'SRC-9002','SRC-9003'}
    assert all(item['reason'] for item in report['skipped_records'])


def test_source_whitespace_is_not_changed(preparer):
    text = '  ورشة عمل بعنوان «عنوان محفوظ».\n'
    record = extract(preparer, text)
    assert record['sources'][0]['content'] == text
    assert OpportunityRecord.model_validate(record).sources[0].content == text


def test_registration_conflicts_cannot_reopen_a_related_closed_record(preparer):
    from datetime import datetime
    from app.internal import SearchPlan
    from app.schemas import UserProfile
    from app.validation import check_candidate
    records = preparer.prepare()
    conflicts = [r for r in records if 'application_status' in r['conflicting_fields']]
    assert conflicts
    for data in conflicts:
        record = OpportunityRecord.model_validate(data)
        assert record.application_status is None
        result = check_candidate(record, SearchPlan().constraints, SearchPlan(), UserProfile(),
                                 datetime.fromisoformat('2026-10-21T12:00:00+03:00'))
        assert not result.accepted
