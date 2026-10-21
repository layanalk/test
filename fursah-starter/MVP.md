# FURSAH MVP

This version implements one main OpenRouter-backed agent inside one LangGraph workflow:

`React → FastAPI/Pydantic → interpret → search → deterministic validation → rank/top 3 → evidence explanations → React`

The existing frontend and public request/response contracts are reused unchanged. No vector database, extra agents, persistent graph checkpoints, or automatic applications are involved.

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
# Default model; replace with another JSON-capable OpenRouter model if desired.
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

`workflow.py` defines one compiled graph. `agent.py` supplies the same main agent for request interpretation and source-passage selection. OpenRouter is called directly with `httpx`; Pydantic validates its JSON responses. Raw announcement text is treated as evidence, not instructions.

Explanation requests send `reasoning: {"enabled": false}` and `provider: {"require_parameters": true}` so the provider must support reasoning-off and the other requested parameters. Interpretation requests retain their existing settings. The completion budget stays at `max_tokens: 1800`; OpenRouter documents that `max_completion_tokens` shares the same reasoning-plus-output budget, so renaming it would not resolve reasoning exhaustion. Merely excluding reasoning from the response would still consume that budget. If the selected model requires reasoning or no provider supports the requested controls, the existing controlled provider error is returned; the app does not silently remove controls, increase the budget, or add fallback requests. Live provider compatibility must be verified locally. See [OpenRouter's reasoning documentation](https://github.com/OpenRouterTeam/docs/blob/d893b1e7ae2384197b9a023d0ad87d4ec1d3ca7b/guides/best-practices/reasoning-tokens.mdx) and [provider parameter routing](https://github.com/OpenRouterTeam/docs/blob/d893b1e7ae2384197b9a023d0ad87d4ec1d3ca7b/guides/routing/provider-selection.mdx#requiring-providers-to-support-all-parameters).

Search is lexical and uses structured constraints. Ranking is deterministic: subject matches first, profile interests as tie-breakers, stable ID ordering afterward. There is no vector search or claim of semantic-retrieval accuracy. Explanations are assembled from the agent's selected **verbatim source excerpts**, with IDs and quotations checked against the selected records before returning them. The model cannot set eligibility, invent a source, change selected IDs, or override hard filters.

Missing information returns through the existing `needs_info` API response. The user edits their profile and resubmits; the graph runs again. No server-side conversation history is required.

Provider failures, invalid model JSON and invalid evidence return controlled structured errors. Model attempts and the whole workflow are bounded. A failed provider call does not fall back to pretend recommendations.

## Verification

From `backend/` with the virtual environment active:

```bash
python -m pytest -q
python -m pip check
```

Tests run the actual LangGraph and FastAPI with `httpx.MockTransport` replacing only OpenRouter's HTTP boundary. They inject a fake test credential and never need a real key or make real provider calls. Coverage includes recommendations/details, maximum result count, source grounding, provenance, missing skills and resubmission, hard filters, conflicts and exclusions, deadline boundaries, missing credentials, malformed model output, errors and timeouts.

From `frontend/`:

```bash
npm run typecheck
npm run build
```

`/health` checks process health. `/ready` validates the prepared store, checks that the environment variable is present and that the graph can be constructed. It makes no billable provider request and cannot prove provider authorization, account balance or model availability. Without a key it returns 503, while health and details remain usable.

## MVP limits

Preparation covers all 1,000 source announcements; this is extraction coverage, not proof that every fact is known or that recommendations are accurate. Retrieval and request interpretation still need broader evaluation. No automated eligibility confirmation, automatic source-conflict reconciliation, real applications, search-broadening approval, alternatives history, persistent checkpoints or backend user accounts are included. The supplied destinations and opportunities are fictional course material. Live OpenRouter authorization and output quality must be checked locally after configuring your key.
