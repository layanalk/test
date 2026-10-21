# FURSAH MVP

This version implements one main OpenRouter-backed agent inside one LangGraph workflow:

`React → FastAPI/Pydantic → interpret (one LLM call) → lexical search → deterministic validation → ranking/batch of ≤3 → match summary + verified source excerpt → React`

The existing frontend and endpoints are reused. Optional batching, match-summary and similar-alternative fields extend the public contracts; existing request bodies and strict-result statuses remain valid. No vector database, extra agents, persistent graph checkpoints, or automatic applications are involved.

## Run locally

Prerequisites: Python 3.11+ and Node.js/npm. Python 3.12 and Node 24 were verified.

From the repository checkout (replace `/path/to/test` with its location):

```bash
cd /path/to/test/fursah-starter/backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Set the key in this terminal without echoing it or placing it in shell history:

```bash
read -r -s -p 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
printf '\n'
```

The backend reads the key **only from `os.environ['OPENROUTER_API_KEY']`**. It does not read a key from `.env`, `FURSAH_MODEL_API_KEY`, frontend configuration, or source code. Do not put a key in React/Vite variables or commit it. No real key is needed to install packages, run tests, start the API, or fetch details. Discovery requires a configured key when using the real provider.

Optional non-secret configuration:

```bash
# Default model; JSON Schema and reasoning-off are requested, validated locally.
export FURSAH_OPENROUTER_MODEL='openai/gpt-4o-mini'
# Reproducible demo date for the supplied 2026 fictional announcements.
# Unset this variable to evaluate against actual request time instead.
export FURSAH_EVALUATION_TIME='2026-10-21T12:00:00+03:00'
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd /path/to/test/fursah-starter/frontend
npm ci
npm run dev
```

Open the local React development server on port 5173. It calls the API on port 8000 by default. The API documentation is served at `/docs`.

Try these requests in the UI:

- `أبحث عن فرص مناسبة` — broad search across all prepared announcements.
- `أبحث عن ورشة عن بعد في تجربة المستخدم` — subject/type/mode filters.
- `أبحث عن تدريب Python في نجران` — asks for skills if they are unanswered; the study-year condition remains uncertain.

If a saved constraint conflicts with the query, the existing UI asks which to use for this request. Saving a profile or editing the query invalidates previous search results. Search constraints are never relaxed automatically.

## Data

`data/prepared/opportunities.json` now contains **1,000 source-level records from all 1,000 raw announcements**. Each record retains its own original source ID, publisher, timestamp and complete text. These are announcements, not 1,000 independently verified unique activities. The repository checks original and related evidence against the raw CSV and computes an evidence fingerprint for recommendations and details.

Preparation is deterministic and offline. Install the preparation requirements and rebuild from any working directory:

```bash
python -m pip install -r /path/to/test/fursah-starter/data/requirements.txt
python /path/to/test/fursah-starter/data/prepare_mvp.py
```

On Windows, Git may convert physical CSV newlines to CRLF, including inside quoted announcement text. Both preparation and loading decode LF/CRLF/CR to LF before exact evidence validation; no other whitespace or Unicode characters are normalized. Escaped newlines inside prepared JSON remain consistent with that decoding. The dataset's Git attributes and explicit UTF-8/LF output keep future checkouts and regeneration consistent. Existing CRLF checkouts work without changing the raw file or regenerating the prepared records.

The backend development requirements also install these preparation dependencies. No OpenRouter key is read or required for preparation. The script reuses the supplied helpers for title/link candidates and related-source discovery, and uses explicit corpus phrases to extract types, attendance, dates, fees, skills and eligibility statements. Unsupported or ambiguous information stays null; source publication time is never used as an activity or deadline date. Calendar duration and session hours are computed only from explicitly stated components.

`data/prepared/preparation_report.json` records coverage, skipped-record reasons, unknown-field counts/source IDs, exact duplicates, shared links and conflicts. Current results: 1,000 prepared, zero skipped, zero duplicate IDs or exact duplicate texts. There are 100 shared-link pairs (200 announcements); 30 pairs contain conflicting facts. Related evidence is retained in `related_sources` without merging activity identities, filling absent facts from another announcement, or assuming that the newest announcement resolves a disagreement. Conflicting structured fields stay null; dependent durations/hours are cleared when their inputs conflict. Registration conflicts are excluded from recommendations until reviewed. The original source statements remain available as evidence. Reposts can therefore remain separate search records; a shared link alone is not treated as proof of duplicate identity.

978 records have at least one null important field. These counts include 259 records without a physical location (all are online), 206 unknown/conflicting fees, 393 unknown/conflicting deadlines and 958 unknown skill requirements. Missing skill requirements are not an assertion that no skills are required. All 1,000 records have an explicitly recognized opportunity type and delivery mode. The original CSV and learner notebooks are unchanged.

Missing costs are not treated as free; unknown dates/durations cannot satisfy a strict matching constraint. Closed, cancelled, clearly expired and finished activities are excluded. Exact deadline equality is expired; a date-only deadline on the evaluation date remains uncertain. Source registration statements are not treated as live confirmation.

Unknown skill profiles (`null`) trigger `needs_info` only for otherwise-matching opportunities with a known skill requirement. An explicit empty skill list (`[]`) means those required skills are absent. Age, study year and other unstructured eligibility conditions remain uncertain rather than being invented or confirmed. The MVP intentionally reports conservative eligibility.

## Agent and graph

`workflow.py` defines one compiled graph and `agent.py` supplies one main agent, used **only to interpret the query**. A new successful recommendation, no-match, or needs-info search normally makes **one** OpenRouter call. Next/previous batches reuse that validated interpretation and make **zero** model calls. Python retrieves, filters, ranks, assigns IDs and selects source excerpts; no opportunity evidence is sent to the model and no second explanation call is made.

The interpretation request uses OpenRouter Chat Completions with:

- `response_format: {"type": "json_schema", "json_schema": {"name": "SearchPlan", "strict": true, "schema": ...}}`, generated from the existing Pydantic schema. Every object field is required in the provider schema, unknown properties are forbidden, and absent values are explicit arrays/nulls. Local Pydantic validation remains unchanged and still checks dates, enums and numeric constraints.
- `reasoning: {"enabled": false}`. `provider.require_parameters` is intentionally omitted: forcing every endpoint to support every requested parameter produced the live HTTP 404 "No compatible endpoints found." JSON Schema/strict and reasoning-off remain requested, but a routed endpoint may not enforce every parameter. Local finish-reason checks and Pydantic validation remain mandatory; no fallback call or JSON repair is added.
- `temperature: 0`, unchanged `max_tokens: 1800`, no tools, no streaming, no extra agents. The prompt retains explicit rules for all eight array fields and asks for short keywords rather than explanations.

Only `finish_reason: "stop"` with final message content that passes Pydantic is accepted. Length-limited output is rejected even when its prefix is parseable JSON. Reasoning text never substitutes for final content; fenced, malformed or incomplete JSON is not repaired or coerced. Diagnostics log schema/attempt/finish/content type/length/error types, never responses, queries, source contents, authorization headers or credentials.

HTTP failures additionally log status, model, attempt, requested JSON Schema/strict/reasoning-off/provider controls, token limit and temperature. Only recognized error codes/types/parameter identifiers and canonical message summaries are emitted. Bounded nested provider errors are inspected for these allowlisted fields; arbitrary provider text, metadata, headers, prompts, response content and exception strings are never logged. Unknown messages are explicitly omitted rather than trusting redaction to remove every possible secret. Transport failures log an allowlisted exception category instead of exception text; a missing process credential is distinguished without sending a request. These diagnostics do not change retries or public API errors and do not prove live Qwen endpoint compatibility. After relaxing routing, diagnostics report `require_parameters=False`; all other requested controls remain visible.

The shared HTTP/output retry budget remains `min(model_max_retries, model_output_retries, 2)`: **two attempts by default, at most three with configuration**, across a new discovery request; pagination makes no model attempt. There are no separate explanation attempts and the graph is not restarted. Transient HTTP 408/429/500/502/503/504, transport errors/timeouts and invalid output use that shared budget. Permanent provider errors (including unsupported parameters, authentication and credit errors) stop after one attempt. Defaults remain 40 seconds per HTTP operation, 90 seconds for the whole workflow and 8 graph steps. Workflow timeout cancels an outstanding asynchronous provider call.

OpenRouter documents that reasoning and final output share the `max_tokens` budget; `max_completion_tokens` uses the same budget. Disabling reasoning removes unnecessary work rather than just hiding reasoning text. Support for reasoning-off and strict structured output varies **by provider endpoint**, including endpoints serving the same Qwen model. Mocks verify our request and error handling, not current live endpoint availability or compliance. A provider error remains controlled without runtime parameter switching or extra fallback calls. Omitting the all-parameter routing restriction addresses the reported route exclusion, but one local live request is still needed to confirm endpoint availability, reasoning-off behavior and compliant final JSON. See [reasoning controls](https://github.com/OpenRouterTeam/docs/blob/d893b1e7ae2384197b9a023d0ad87d4ec1d3ca7b/guides/best-practices/reasoning-tokens.mdx), [structured outputs](https://github.com/OpenRouterTeam/docs/blob/d893b1e7ae2384197b9a023d0ad87d4ec1d3ca7b/guides/features/structured-outputs.mdx), and [parameter routing](https://github.com/OpenRouterTeam/docs/blob/d893b1e7ae2384197b9a023d0ad87d4ec1d3ca7b/guides/routing/provider-selection.mdx).

Search remains lexical and uses structured constraints. Ranking scores every eligible candidate with its final deterministic match percentage, then sorts descending by that exact percentage, with stable ID ordering for ties. The complete order is retained as IDs/evidence fingerprints/percentages for an ephemeral search session; each response contains maximum three recommendations. There is no vector search or claim of semantic-retrieval accuracy. `explanations.py` chooses one original excerpt of 8–160 characters per selected record, preferring passages containing query keywords and using stable source order to break ties. It constructs the unchanged `ExplanationBatch` Pydantic model, then the existing evidence verifier checks exact ID coverage, uniqueness and verbatim membership in the corresponding original sources. No paraphrases or invented facts enter the explanation, and eligibility remains deterministic and conservative. All 1,000 records are covered by the excerpt-verification regression test.


### Deterministic match percentage before selection

`scoring.py` computes a whole-number `relevance_score` for **every eligible candidate before sorting or slicing**. Only validated record fields and the submitted profile/effective search constraints are used. The LLM never calculates percentages. Three factor groups have explicit weights:

| Supplied factor group | Weight | Coverage |
| --- | --- | --- |
| Query subject keywords | 60 | Fraction of unique normalized query terms found in title/topics/description |
| Profile interests | 25 | Fraction of unique normalized interests found in those same fields |
| Explicit positive constraints | 15 | Fraction of requested location/type/mode/budget/duration/availability constraints supported by known matching facts |

Absent groups are excluded from both numerator and denominator. Percentage = `100 × sum(weight × coverage) / sum(active weights)`, rounded half up using exact rational arithmetic. Unknown facts earn no credit; repeated/case-equivalent/blank terms do not inflate scores. A search with no supported scoring criteria receives 0%, rather than an invented personalized percentage. Hard filtering still happens first, so every accepted candidate satisfies its supplied hard constraints; excluded/closed/expired/ineligible records cannot outrank eligible ones.

`score_and_rank()` first materializes all scores, sorts by the **final rounded percentage** descending and ID ascending, and only then does the graph select a three-record page. It never normalizes against the best/worst candidate in the current batch. The already-computed value is copied into each returned recommendation's existing `relevance_score`; explanation generation does not recalculate it. One/two eligible records produce one/two scored recommendations. No eligible records continue to the existing no-match/similar flow.

React prominently displays `94% مطابقة`, independently of the match summary, source quotation and eligibility badge. This percentage measures supplied-criteria coverage, **not probability of success or confirmation of eligibility**. Similar alternatives remain separately labeled near misses and do not display an eligible-match percentage. Their existing fewest-failed-constraints policy is retained, with subject/interest coverage and stable ID ordering for ties.

All subsequent batches use the same complete ranking and revalidate its IDs, evidence fingerprints and percentages. Next/Previous navigation never calls the LLM. A scoring/evidence/cohort change invalidates the session instead of shifting page boundaries or repeating IDs across forward batches.

### Similar opportunities without changing strict matches

When strict filtering yields no accepted candidates and no actionable missing-skills request, `alternatives.py` evaluates the same lexical candidates. It keeps explicit exclusions, closed/cancelled/not-yet-open registration rejection, expired deadlines/ended activities, attendance consistency and required-skill checks. It may expose a near miss for positive location/type/mode/budget/duration/availability preferences, but never silently labels it a match.

A meaningful alternative must share an extracted subject; if the query has no subjects, it must match a profile interest, or (with no interests) a requested location/type/mode. At most two positive preferences may fail or be unknown. Candidates are ordered by fewest such issues, then the existing subject/interest ranking and stable ID. Maximum three are returned in `similar_alternatives`, **only with `status: "no_match"` and empty `recommendations`**. Each includes all `failed_constraints` with a field name, `not_met`/`unknown` status and explanation. Unknown prices/dates are never treated as proven matches. An unrelated or unsafe pool still returns ordinary no-match. Alternatives are one small list, not a second paginated search.

### Next and previous batches

`batching.py` holds at most 64 disposable sessions for 10 minutes in the current backend process. A session retains the validated plan, effective constraints, filter labels, input fingerprint and ordered IDs/evidence versions/percentages—not persistent recommendation history or source text. No database, extra agent or LangGraph checkpoint is involved. Tokens are random opaque IDs; they are not credentials for the model and are not logged or saved to browser storage.

A successful search with more than three matches returns `batch_token`, `batch_index: 0`, and `has_more`. Navigation posts the original query/profile/constraints/resolution plus that token and the desired zero-based `batch_index` to the same `/api/v1/discover` endpoint. The same LangGraph runs, but `interpret` restores the validated plan without calling the model. Search/filter/ranking and original-evidence checks run again. The ordered IDs, fingerprints and percentages must match the session before fixed three-record page boundaries are used. Forward batches do not duplicate IDs; returning to a previous batch deliberately revisits those same IDs.

Changed inputs, changed eligible results/evidence, expiration, eviction, invalid pages or backend restarts return HTTP 409 `FURSAH_BATCH_EXPIRED`; they never silently rerun the LLM or shift page boundaries. React offers a fresh-search retry; transient navigation errors retain the page request for a zero-LLM retry. Editing the query/profile clears the current navigation state. This intentionally targets a **single backend process**: multiple workers need shared state or sticky routing, outside this MVP. Sessions are not persisted and their TTL does not extend on reads.

### Deterministic personalized summaries

`build_match_summary()` adds `match_summary` independently of the unchanged verified quotation in `reason`. It uses at most five factual clauses: actual keyword/interest matches, matching requested mode/location/type, a known fee/duration within the requested maximum, matching known required skills or known dates satisfying the corresponding availability bound. It does not infer suitability from profile city, education or skill levels, and never claims complete eligibility. Unknown or failing fields do not produce positive claims. Broad searches without supported personal matches say so explicitly.

React shows match summaries and original-source quotations separately. Alternative cards and their details retain the non-match warning and failed/unknown constraints; evidence fingerprint checks still suppress stale assessments.

### Alignment with the intended system design

The actual graph remains `START → interpret → search → validate → rank → explain → END`, with conditional exits after interpretation and validation. Similar alternatives now continue through deterministic ranking/output and the same evidence verifier. Clarification/conflict decisions remain application-level edit/choice and resubmission, not paused graph executions. Batch navigation reuses short-lived interpretation state. The explanation box should be labeled **Python match summary + original excerpt**, not an explanation LLM.

Still outside scope: automatic applications, permanent application/interest tracking, permanent recommendation history, broad semantic similarity, full age/education/study-year eligibility evaluation and automatic duplicate-identity merging. The supplied activities and destinations remain fictional course data.

Missing information returns through the existing `needs_info` API response. The user edits their profile and resubmits; the graph runs again. No server-side conversation history is required.

Provider failures, invalid model JSON and invalid evidence return controlled structured errors. Model attempts and the whole workflow are bounded. A failed provider call does not fall back to pretend recommendations.

## Verification

From `backend/` with the virtual environment active:

```bash
python -m pytest -q
python -m pip check
```

Tests run the actual LangGraph and FastAPI with `httpx.MockTransport` replacing only OpenRouter's HTTP boundary. They inject a fake test credential and never need a real key or make real provider calls. Coverage includes next/previous batches without extra model calls, stable ordering and duplicate prevention, expiration/input/evidence changes, labeled near misses, factual summaries, scoring every eligible record before selection, final-percentage ordering/rounding/ties, cross-batch percentage consistency, complete one/two/three-result API flows, maximum result count, source grounding and hallucination rejection, all-1,000-record provenance and excerpts, missing skills/resubmission, hard filters, conflicts/exclusions, date boundaries, array shapes, native schema requests, malformed/truncated/refused output, reasoning-heavy responses, unsupported parameters, shared bounded retries, HTTP errors, provider cancellation and graph/workflow timeouts. Explanation JSON schemas remain tested for malformed and truncated data even though explanations no longer require provider JSON.

Frontend tests use the existing Vite compiler and Node test runner with mocked fetch and server rendering; they do not require a real API or new testing dependencies. From `frontend/`:

```bash
npm test
npm run typecheck
npm run build
```

`/health` checks process health. `/ready` validates the prepared store, checks that the environment variable is present and that the graph can be constructed. It makes no billable provider request and cannot prove provider authorization, account balance or model availability. Without a key it returns 503, while health and details remain usable.

## MVP limits

Preparation covers all 1,000 source announcements; this is extraction coverage, not proof that every fact is known or that recommendations are accurate. Retrieval and request interpretation still need broader evaluation. No automated eligibility confirmation, automatic source-conflict reconciliation, real applications, automatic search broadening, permanent alternatives history, persistent checkpoints or backend user accounts are included. The supplied destinations and opportunities are fictional course material. Live OpenRouter authorization and output quality must be checked locally after configuring your key.
