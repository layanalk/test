"""Reproducibly prepare every raw announcement without an LLM or credentials.

One record per source announcement; shared links identify review candidates, not
proven duplicate activities. Extraction is deliberately limited to explicit corpus
phrases. Unknown/contradictory facts remain null. No newest-source-wins policy.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from preparation_helpers import _MONTHS, extract_facts, related_sources

ROOT = Path(__file__).resolve().parent
RAW_COLUMNS = ['source_id', 'source_type', 'source_name', 'published_at', 'content']
DATE = r'(\d{1,2})\s+(' + '|'.join(_MONTHS) + r')\s+(\d{4})'
TYPES = {
    'workshop': ('ورشة تطبيقية', 'ورشة عمل', 'جلسة تدريبية عملية'),
    'course': ('برنامج تعليمي متسلسل', 'دورة تدريبية', 'دورة من عدة لقاءات'),
    'internship': ('تدريب مهني داخل المؤسسة', 'تدريب عملي داخل مقر العمل', 'فرصة تدريب ميداني بإشراف الفريق'),
    'competition': ('منافسة لتطوير فكرة', 'مسابقة فرق', 'تحدي تنافسي'),
    'volunteering': ('نشاط تطوعي', 'فرصة للتطوع', 'مهمة تطوعية مجتمعية'),
    'event': ('أمسية معرفية', 'لقاء مفتوح', 'فعالية حوارية'),
}
IMPORTANT_FIELDS = ('opportunity_type', 'delivery_mode', 'location', 'cost', 'start_date',
                    'end_date', 'duration', 'deadline', 'required_skills')
CONFLICT_FIELDS = IMPORTANT_FIELDS + ('application_status', 'remote_participation_allowed',
                                      'physical_attendance_required', 'session_count',
                                      'hours_per_session', 'total_hours')
FIELD_LABELS = {
    'opportunity_type': 'نوع الفرصة', 'delivery_mode': 'طريقة المشاركة',
    'location': 'الموقع', 'cost': 'رسوم المشاركة', 'start_date': 'تاريخ البداية',
    'end_date': 'تاريخ النهاية', 'duration': 'المدة التقويمية', 'deadline': 'آخر موعد للتقديم',
    'required_skills': 'المهارات المطلوبة', 'application_status': 'حالة التسجيل',
    'remote_participation_allowed': 'إمكانية المشاركة الكاملة عن بعد',
    'physical_attendance_required': 'إلزامية الحضور', 'session_count': 'عدد اللقاءات',
    'hours_per_session': 'ساعات اللقاء', 'total_hours': 'إجمالي ساعات المشاركة',
}


def unique(values):
    return list(dict.fromkeys(values))


def one(values):
    values = unique(values)
    return values[0] if len(values) == 1 else None


def dates(text, prefix, issues, field):
    found = []
    for match in re.finditer(prefix + DATE, text):
        day, month, year = match.groups()[-3:]
        try:
            found.append(date(int(year), _MONTHS[month], int(day)).isoformat())
        except ValueError:
            issues.append(f'{field}: invalid calendar date in source')
    return unique(found)


def extract_record(row, titles, issues):
    text = row['content']
    warnings = ['إعلانات وعناوين تقديم خيالية للتدريب؛ لا تستقبل طلبات حقيقية.',
                'لم يتم التحقق من حالة التسجيل الحالية مع الجهة.']
    candidates = unique(title for title in titles if title != row['source_name'])
    if len(candidates) != 1:
        raise ValueError('Missing or ambiguous activity title (publisher names excluded)')
    kind = one([kind for kind, phrases in TYPES.items() if any(phrase in text for phrase in phrases)])
    online = 'المشاركة عن بعد بالكامل عبر قاعة إلكترونية' in text
    hybrid_cities = re.findall(r'البداية حضورية في ([^.\n،]+)، ثم تُعقد جلسة متابعة عن بعد', text)
    onsite_cities = re.findall(r'الحضور في ([^.\n]+?) داخل مقر', text)
    onsite_cities += re.findall(r'العمل حضوري داخل مقر [^.\n]+? في ([^.\n]+)', text)
    modes = (['online'] if online else []) + (['hybrid'] if hybrid_cities else []) + (['in_person'] if onsite_cities else [])
    mode = one(modes)
    location = one(hybrid_cities + onsite_cities)
    remote = True if online else (False if onsite_cities or hybrid_cities else None)
    physical = True if onsite_cities or hybrid_cities else (False if online else None)
    if len(unique(modes)) > 1:
        mode = remote = physical = None
        issues.append('delivery_mode: conflicting attendance statements in source')
    if len(unique(hybrid_cities + onsite_cities)) > 1:
        location = None
        issues.append('location: multiple attendance locations in source')

    # Full explicit activity dates, not publication/opening/update dates.
    starts = dates(text, r'يبدأ البرنامج يوم\s+', issues, 'start_date')
    ends = dates(text, r'وينتهي يوم\s+', issues, 'end_date')
    single_days = dates(text, r'هذا نشاط ليوم واحد بتاريخ\s+', issues, 'start_date')
    starts = unique(starts + single_days)
    ends = unique(ends + single_days)
    # Updates explicitly state a new start/deadline; no inferred shift of end date.
    for match in re.finditer(r'تغيّر الموعد من\s+' + DATE + r'\s+إلى\s+' + DATE, text):
        starts += dates('new ' + ' '.join(match.groups()[3:]), r'new\s+', issues, 'start_date')
    deadlines = dates(text, r'آخر يوم للتقديم\s+', issues, 'deadline')
    for match in re.finditer(r'مُدد آخر يوم للتقديم من\s+' + DATE + r'\s+إلى\s+' + DATE, text):
        deadlines += dates('new ' + ' '.join(match.groups()[3:]), r'new\s+', issues, 'deadline')
    start, end, deadline = one(starts), one(ends), one(deadlines)
    for field, values in [('start_date', starts), ('end_date', ends), ('deadline', deadlines)]:
        if len(unique(values)) > 1:
            issues.append(f'{field}: conflicting explicit dates in source')
    if start and end and start > end:
        issues.append('start_date/end_date: activity ends before it starts')
        start = end = None
    duration = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1 if start and end else None

    amounts = [float(v) for v in re.findall(r'رسوم المشاركة\s+(\d+(?:\.\d+)?)\s+ريال', text)]
    if 'المشاركة مجانية' in text:
        amounts.append(0.0)
    cost = one(amounts)
    if len(unique(amounts)) > 1:
        issues.append('cost: conflicting fees in source')
    if 'لم تُحدد رسوم المشاركة' in text and amounts:
        cost = None
        issues.append('cost: stated fee conflicts with an unknown-fee statement')

    status_claims = []
    if any(s in text for s in ('باب التقديم مغلق حالياً', 'انتهى استقبال الطلبات لهذه النسخة', 'أُغلق التسجيل لاكتمال المقاعد')):
        status_claims.append('closed')
    if any(s in text for s in ('تم إلغاء النشاط', 'اعتذر المنظمون عن إقامة البرنامج وألغوه', 'أُلغيت هذه الفرصة')):
        status_claims.append('cancelled')
    if any(s in text for s in ('التسجيل مفتوح حالياً', 'باب المشاركة مفتوح وقت نشر الإعلان', 'تُستقبل الطلبات الآن',
                               'لم يبدأ التسجيل بعد', 'أوقفنا استقبال الطلبات مؤقتاً')):
        status_claims.append('requires_verification')
    status = one(status_claims)
    if len(unique(status_claims)) > 1:
        issues.append('application_status: conflicting source registration statements')

    skills = unique(re.findall(r'معرفة ([A-Za-z][A-Za-z ]*?) مطلوبة', text)) or None
    mandatory = unique(re.findall(r'يشترط[^.\n]+|معرفة [A-Za-z][A-Za-z ]*? مطلوبة', text)) or None
    preferred = unique(re.findall(r'تُعد معرفة [^.\n]+ليست شرطاً', text)) or None
    # A beginner-friendly invitation does not establish absence of prerequisites.
    if skills is None and re.search(r'لا توجد مهارات مطلوبة|لا يشترط امتلاك أي مهارات', text):
        skills = []
    session_values = [int(v) for v in re.findall(r'يشمل (\d+) لقاءات', text)]
    sessions = one(session_values)
    hours = 2.0 if 'مدة كل لقاء ساعتان' in text else None
    url = one(re.findall(r'https://fursah-\d+\.example/info', text))
    if 'تحديث من' in text or 'تغيّر الموعد' in text:
        warnings.append('إعلان تحديث: احتفظنا بنصه ولم نفترض أنه يلغي إعلانًا آخر أو يغير تاريخ النهاية غير المذكور.')
    for issue in issues:
        warnings.append('تحتاج معلومة مصدرية إلى مراجعة: ' + issue)
    return {
        'opportunity_id': 'OPP-' + row['source_id'].removeprefix('SRC-'),
        'title': candidates[0], 'description': text, 'opportunity_type': kind,
        'location': location, 'delivery_mode': mode,
        'remote_participation_allowed': remote, 'physical_attendance_required': physical,
        'start_date': start, 'end_date': end, 'duration': duration,
        'cost': cost, 'deadline': deadline, 'application_status': status,
        'required_skills': skills,
        'eligibility_requirements': {'mandatory': mandatory, 'preferred': preferred},
        'session_count': sessions, 'hours_per_session': hours,
        'total_hours': sessions * hours if sessions and hours else None,
        'topics': [candidates[0]], 'source_ids': [row['source_id']], 'application_url': url,
        'data_warnings': warnings,
        'sources': [{k: row[k] for k in ('source_id', 'source_name', 'published_at', 'content')}],
        'related_sources': [], 'conflicting_fields': [],
    }


def prepare_with_report(csv_path=None):
    path = Path(csv_path) if csv_path is not None else ROOT / 'raw/raw_opportunities.csv'
    # Match DatasetLoader's universal-newline decoding, including quoted cells.
    # Only transport line endings are canonicalized; no source text is stripped.
    with path.open(encoding='utf-8-sig', newline=None) as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != RAW_COLUMNS:
            raise ValueError('Unexpected raw CSV headers; refusing to overwrite prepared data')
        raw = list(reader)
    id_counts = Counter(row.get('source_id') for row in raw)
    skipped, records, field_issues = [], [], {}
    valid_rows = []
    for number, row in enumerate(raw, 2):
        try:
            if set(row) != set(RAW_COLUMNS) or any(not isinstance(v, str) or not v.strip() for v in row.values()):
                raise ValueError('Missing required raw column/value or malformed CSV row')
            if not re.fullmatch(r'SRC-\d+', row['source_id']):
                raise ValueError('Invalid source ID format')
            if id_counts[row['source_id']] > 1:
                raise ValueError('Duplicate source ID; no arbitrary first/last row selected')
            if datetime.fromisoformat(row['published_at']).tzinfo is None:
                raise ValueError('Publication timestamp lacks a timezone')
            valid_rows.append(row)
        except (ValueError, TypeError) as exc:
            skipped.append({'csv_line': number, 'source_id': row.get('source_id'), 'reason': str(exc)})
    # Reuse the lesson helpers for candidate titles, links and related-source discovery.
    frame = pd.DataFrame(valid_rows, columns=RAW_COLUMNS)
    frame['cleaned_content'] = frame['content'].str.strip()
    facts = extract_facts(frame)
    related = set(related_sources(facts)['source_id'])
    for row, titles in zip(valid_rows, facts['title_candidates']):
        issues = []
        try:
            record = extract_record(row, titles, issues)
            records.append(record)
            if issues:
                field_issues[row['source_id']] = issues
        except ValueError as exc:
            skipped.append({'source_id': row['source_id'], 'reason': str(exc)})

    linked = defaultdict(list)
    for record in records:
        if record['sources'][0]['source_id'] in related and record['application_url']:
            linked[record['application_url']].append(record)
    # Compare untouched extracted snapshots, so processing order cannot resolve conflicts.
    conflicts = {}
    for url, group in linked.items():
        fields = []
        for field in CONFLICT_FIELDS:
            known = {json.dumps(r[field], ensure_ascii=False, sort_keys=True) for r in group if r[field] is not None}
            if len(known) > 1:
                fields.append(field)
        if fields:
            conflicts[url] = fields
        ids = [r['source_ids'][0] for r in group]
        evidence = [r['sources'][0] for r in group]
        for record in group:
            own_id = record['source_ids'][0]
            record['related_sources'] = [s.copy() for s in evidence if s['source_id'] != own_id]
            record['data_warnings'].append('إعلانات مرتبطة بالرابط تحتاج إلى مراجعة دون دمج أو افتراض تطابق الهوية: ' + '، '.join(ids))
            record['conflicting_fields'] = fields.copy()
            for field in fields:
                record[field] = None
                record['data_warnings'].append('اختلاف بين أدلة المصادر في ' + FIELD_LABELS[field] + '؛ أبقينا القيمة مجهولة. المصادر: ' + '، '.join(ids))
            if any(field in fields for field in ('start_date', 'end_date')):
                record['duration'] = None
            # Do not leave attendance flags implying certainty after a mode conflict.
            if 'delivery_mode' in fields:
                record['remote_participation_allowed'] = record['physical_attendance_required'] = None
            if any(field in fields for field in ('session_count', 'hours_per_session')):
                record['total_hours'] = None

    # Full output must conform to the actual repository model before it is written.
    import sys
    backend = str(ROOT.parent / 'backend')
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from app.internal import OpportunityRecord
    validated = []
    for record in records:
        try:
            OpportunityRecord.model_validate(record)
            validated.append(record)
        except ValueError as exc:
            skipped.append({'source_id': record['source_ids'][0], 'reason': 'Prepared schema validation failed: ' + str(exc)})
    unknown = {field: [r['source_ids'][0] for r in validated if r[field] is None] for field in IMPORTANT_FIELDS}
    incomplete = [r['source_ids'][0] for r in validated if any(r[field] is None for field in IMPORTANT_FIELDS)]
    report = {
        'raw_row_count': len(raw), 'prepared_record_count': len(validated),
        'skipped_invalid_record_count': len(skipped), 'skipped_records': skipped,
        'duplicate_source_id_row_count': sum(n - 1 for n in id_counts.values() if n > 1),
        'exact_duplicate_text_count': len(raw) - len({r.get('content') for r in raw}),
        'shared_link_group_count': len(linked),
        'shared_link_announcement_count': sum(len(group) for group in linked.values()),
        'extra_shared_link_announcements': sum(len(group) - 1 for group in linked.values()),
        'conflicting_link_group_count': len(conflicts),
        'records_with_conflicting_evidence_count': sum(bool(r['conflicting_fields']) for r in validated),
        'conflicting_fields_by_link': conflicts,
        'records_with_incomplete_important_fields_count': len(incomplete),
        'unknown_important_field_counts': {field: len(ids) for field, ids in unknown.items()},
        'unknown_field_source_ids': unknown,
        'source_extraction_issues': field_issues,
        'identity_policy': 'One record per announcement. Shared links are review candidates, not proven duplicate opportunities. No automatic merge or latest-source resolution.',
    }
    return validated, report


def prepare():
    return prepare_with_report()[0]


if __name__ == '__main__':
    records, report = prepare_with_report()
    if not records:
        raise SystemExit('No valid records; refusing to overwrite prepared data')
    destination = ROOT / 'prepared'
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in [('opportunities.json', records), ('preparation_report.json', report)]:
        temporary = destination / (name + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
        temporary.replace(destination / name)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))}, ensure_ascii=False, indent=2))
    print('Unknown important fields:', json.dumps(report['unknown_important_field_counts']))
