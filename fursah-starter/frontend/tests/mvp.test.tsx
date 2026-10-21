import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { renderToStaticMarkup } from 'react-dom/server';
import { discover } from '../src/api/client';
import SearchResults from '../src/opportunities/SearchResults';
import OpportunityCard from '../src/opportunities/OpportunityCard';
import type { DiscoveryRequest, DiscoveryResponse, Recommendation, SimilarAlternative } from '../src/types';

const request: DiscoveryRequest = {
  request_id: 'test-ui', query: 'Python',
  profile: { user_id: null, city: null, education_level: null, field_of_study: null, interests: ['Python'], skills: [] },
  constraints: { location_preference: null, opportunity_types: [], delivery_modes: ['online'],
    available_from: null, available_to: null, max_duration_days: null, max_budget_sar: 100 },
};
const recommendation = (id: number): Recommendation => ({
  opportunity_id: `TEST-${id}`, title: `Python workshop ${id}`, eligibility_status: 'uncertain',
  reason: 'من نص الإعلان: «Original source quote.»', match_summary: 'التكلفة المعلنة 50 ريال ضمن ميزانيتك.',
  source_ids: [`SRC-${id}`], evidence_version: `version-${id}`, caveats: [],
});
const response = (count = 3): DiscoveryResponse => ({
  request_id: 'test-ui', status: 'ok', message: 'Matched records', error_code: null,
  missing_fields: [], recommendations: Array.from({ length: count }, (_, index) => recommendation(index)),
  similar_alternatives: [], batch_index: 0, batch_token: null, has_more: false,
});
const alternative = (): SimilarAlternative => ({
  ...recommendation(10), eligibility_status: 'uncertain', match_summary: 'يطابق موضوع بحثك: Python.',
  failed_constraints: [{ field: 'max_budget_sar', status: 'not_met', message: 'التكلفة 120 ريال تتجاوز ميزانيتك 100 ريال.' }],
});
const noMatch = (): DiscoveryResponse => ({
  ...response(0), status: 'no_match', similar_alternatives: [alternative()],
});
const noNetwork = async () => { throw new Error('Real network forbidden in frontend tests'); };
globalThis.fetch = noNetwork;
afterEach(() => { globalThis.fetch = noNetwork; });

function mockResponse(body: unknown, status = 200) {
  const calls: { url: string; body: Record<string, unknown> }[] = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), body: JSON.parse(String(init?.body)) });
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
  };
  return calls;
}

function renderResults(res: DiscoveryResponse) {
  return renderToStaticMarkup(<SearchResults res={res} view={res.status}
    onBatch={() => {}} onResolve={() => {}} onEditProfile={() => {}} onRetry={() => {}}
    onFocusQuery={() => {}} onOpenDetails={() => {}} />);
}

for (const count of [1, 2, 3]) {
  test(`accepts and renders ${count} normal API recommendations`, async () => {
    const calls = mockResponse(response(count));
    const result = await discover(request);
    assert.equal(result.status, 'ok');
    assert.equal(result.recommendations.length, count);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, 'http://localhost:8000/api/v1/discover');
    const html = renderResults(result);
    assert.equal((html.match(/<article/g) ?? []).length, count);
    assert.match(html, /أوجه المطابقة المثبتة/);
    assert.match(html, /اقتباس من المصدر/);
  });
}

test('old API responses without additive fields remain compatible', async () => {
  const old = response(1);
  delete old.similar_alternatives; delete old.batch_index; delete old.batch_token; delete old.has_more;
  delete old.recommendations[0].match_summary;
  mockResponse(old);
  assert.equal((await discover(request)).status, 'ok');
});

test('no_match alternatives show explicit failures and never pretend to be matches', async () => {
  const body = noMatch();
  body.similar_alternatives![0].failed_constraints.push({ field: 'max_duration_days', status: 'unknown', message: 'المدة غير معروفة.' });
  mockResponse(body);
  const result = await discover(request);
  assert.equal(result.status, 'no_match');
  assert.equal(result.recommendations.length, 0);
  const html = renderResults(result);
  assert.match(html, /لا تستوفي كل القيود/);
  assert.match(html, /غير مطابق:/);
  assert.match(html, /غير معروف:/);
  assert.match(html, /120 ريال تتجاوز ميزانيتك 100 ريال/);
  assert.match(html, /Original source quote/);
  assert.doesNotMatch(html, /فرصة واحدة تطابق/);
});

test('plain no_match shows no alternative cards or navigation', () => {
  const body = noMatch(); body.similar_alternatives = [];
  const html = renderResults(body);
  assert.doesNotMatch(html, /<article|بدائل قريبة|الدفعة التالية/);
});

for (const defect of ['missing_failures', 'scalar_failures', 'empty_failures', 'unknown_field', 'array_field', 'array_status', 'confirmed', 'too_many', 'mixed_success']) {
  test(`rejects unsafe alternative contract: ${defect}`, async () => {
    const body = structuredClone(noMatch()) as any;
    const item = body.similar_alternatives[0];
    if (defect === 'missing_failures') delete item.failed_constraints;
    if (defect === 'scalar_failures') item.failed_constraints = 'over budget';
    if (defect === 'empty_failures') item.failed_constraints = [];
    if (defect === 'unknown_field') item.failed_constraints[0].field = 'invented';
    if (defect === 'array_field') item.failed_constraints[0].field = ['max_budget_sar'];
    if (defect === 'array_status') item.failed_constraints[0].status = ['not_met'];
    if (defect === 'confirmed') item.eligibility_status = 'confirmed';
    if (defect === 'too_many') body.similar_alternatives = Array(4).fill(item);
    if (defect === 'mixed_success') { body.status = 'ok'; body.recommendations = [recommendation(1)]; }
    mockResponse(body);
    assert.equal((await discover(request)).error_code, 'FURSAH_BAD_RESPONSE');
  });
}

test('next batch sends original inputs and session/index; first and final navigation states render', async () => {
  const body = { ...response(2), batch_token: '0123456789abcdef0123456789abcdef', batch_index: 1, has_more: false };
  const calls = mockResponse(body);
  const result = await discover({ ...request, batch_token: body.batch_token, batch_index: 1 });
  assert.equal(result.status, 'ok');
  assert.equal(calls[0].body.batch_token, body.batch_token);
  assert.equal(calls[0].body.batch_index, 1);
  assert.deepEqual(calls[0].body.profile, request.profile);
  assert.deepEqual(calls[0].body.constraints, request.constraints);
  const html = renderResults(result);
  assert.match(html, /الدفعة السابقة/);
  assert.match(html, /الدفعة 2/);
  assert.match(html, /disabled="">الدفعة التالية/);
  const first = renderResults({ ...body, batch_index: 0, has_more: true });
  assert.match(first, /disabled="">الدفعة السابقة/);
});

test('pagination preserves explicit conflict resolution choice', async () => {
  const calls = mockResponse({ ...response(), batch_token: '0123456789abcdef', has_more: true });
  await discover({ ...request, constraint_resolution: 'query', batch_token: '0123456789abcdef', batch_index: 1 });
  assert.equal(calls[0].body.constraint_resolution, 'query');
});

for (const paging of [{ batch_index: -1 }, { batch_index: 1.5 }, { batch_index: 1 }, { batch_index: 0, has_more: true }]) {
  test(`rejects invalid response pagination ${JSON.stringify(paging)}`, async () => {
    mockResponse({ ...response(), ...paging });
    assert.equal((await discover(request)).error_code, 'FURSAH_BAD_RESPONSE');
  });
}

test('later batch without token is rejected locally without fetch', async () => {
  const calls = mockResponse(response());
  assert.equal((await discover({ ...request, batch_index: 1 })).error_code, 'FURSAH_REQUEST_VALIDATION_ERROR');
  assert.equal(calls.length, 0);
});

test('expired session has a clear restart action instead of silently running interpretation again', async () => {
  mockResponse({ error_code: 'FURSAH_BATCH_EXPIRED', message: 'Expired', request_id: 'test-ui' }, 409);
  const result = await discover(request);
  assert.equal(result.status, 'error');
  assert.equal(result.error_code, 'FURSAH_BATCH_EXPIRED');
  assert.match(renderResults(result), /انتهت جلسة النتائج/);
});

test('needs_info, conflict and provider errors retain existing UI branches', () => {
  const needs = { ...response(0), status: 'needs_info' as const, missing_fields: ['profile.skills'] };
  assert.match(renderResults(needs), /تحديث الملف الشخصي/);
  const conflict = renderResults({ ...needs, constraint_conflict: true, missing_fields: ['constraints.location_preference'] });
  assert.match(conflict, /استخدم قيود ملفي/);
  assert.match(conflict, /استخدم قيود نص البحث/);
  const error = renderResults({ ...response(0), status: 'error', error_code: 'FURSAH_MODEL_ERROR' });
  assert.match(error, /FURSAH_MODEL_ERROR/);
  assert.match(error, /إعادة المحاولة/);
});

test('summary and original quote have independent visible sections and user text is escaped', () => {
  const rec = { ...recommendation(1), match_summary: '<script>unsafe</script>' };
  const html = renderToStaticMarkup(<OpportunityCard rec={rec} onOpen={() => {}} />);
  assert.match(html, /أوجه المطابقة المثبتة/);
  assert.match(html, /اقتباس من المصدر/);
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /<script>/);
});
