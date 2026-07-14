# Claims-Cost Intelligence Platform

A governed semantic layer + AI agent for claims-cost analytics. dbt MetricFlow
defines every metric once; a Tableau dashboard and a natural-language query
agent both read that same definition, so they can never quietly disagree
about what "total spend" means.

---

## The problem this solves

Most companies eventually build two systems that both claim to know "total
spend": a BI dashboard for analysts, and — increasingly — an AI assistant
that answers plain-English questions about the same data. When those two
systems compute metrics independently, the numbers drift. That's not a
hypothetical risk; it's a common, real governance failure.

The differentiator here isn't "an AI that answers questions about data." It's
an agent that is **structurally prevented from computing an answer outside
governed logic** — not by asking it nicely, but by routing every query
through a deterministic guardrail that checks it against the semantic
layer's own schema before anything executes.

---

## Architecture

```
Synthetic claims data (DuckDB, local)
              │
              v
        dbt models ──> MetricFlow semantic layer
              │                    │
              │         (single definition of every metric:
              │          total spend, PMPM, denial rate,
              │          days-to-adjudication)
              │                    │
      ┌───────┴────────┐   ┌───────┴────────┐
      v                v   v                v
  Tableau           FastAPI /ask endpoint + chat UI
  dashboard         (NL question -> agent -> guardrail -> semantic layer)
```

**NL query flow:**

```
Question
   │
   v
LangGraph agent: propose a structured query (metric + dimensions +
filters + time_grain + start/end date) — NOT raw SQL
   │
   v
Guardrail: validate the proposal against the semantic layer's actual
schema — does this metric exist? these dimensions? this time grain? —
reject with a specific reason if not
   │
   v
Cache check → execute the validated query on miss (same code path
Tableau uses) → summarize the result in plain English
```

---

## The guardrail: three layers of protection

The guardrail (`app/guardrail.py`) is deliberately plain Python reading
MetricFlow's own `semantic_manifest.json` — not another LLM call. The thing
checking the AI isn't itself an AI you have to trust blindly.

Three genuinely distinct failure modes get caught, each demonstrated live
against a real local model (not scripted):

**1. Fabricated metric name.** Ask about something that doesn't exist
("readmission rate") and the guardrail rejects it by name, listing what
actually exists.

**2. Guardrail-approved but MetricFlow-unresolvable.** `pmpm` is a *ratio*
metric spanning two semantic models (`claims` and `member_months`) with
different primary entities. The guardrail approves `region` as a dimension
for `pmpm` (it exists on both underlying models independently) — but
MetricFlow's join resolution can't actually group a ratio metric by it. This
needed a second, execution-layer check (`app/query_executor.py`) asking
MetricFlow itself which dimensions are *actually* resolvable per metric,
since a schema-level check alone over-approves.

**3. No confident match — verified candidates instead of a wrong guess.**
Originally, the system prompt told the model to "propose your best guess
anyway" when nothing matched, on the theory that the guardrail would catch
anything invalid. In practice this backfired: a real local model, asked
about "therapy adherence," picked `pmpm` — a real, existing metric — as its
best guess. That passed schema validation (region exists, pmpm exists) and
would have silently returned a confident, wrong answer. The fix: the model
is now told to decline explicitly (`metric: null`) rather than substitute an
unrelated real metric, and offer `candidate_metrics` instead. Those
candidates are then filtered against the real manifest before ever being
shown to the user — never trusting the model's suggestion list any more than
its primary answer.

`![guardrail catching a fabricated metric, an unjoinable dimension, and a no-match case](screenshots/q5-and-q6-rejections.png)`

A fourth refinement worth calling out: local models repeatedly tried to
express time bucketing through the `dimensions` list rather than
`time_grain` — first by misusing a real dimension (`submitted_date`), then
by inventing a fake one (`month`, a time-grain word with no corresponding
dimension at all). Both get caught, and the second case gets a pointed hint
("looks like a time granularity, not a dimension — use `time_grain='month'`
instead") rather than a generic dimension list, since the failure mode is
specific and recurring enough to deserve a specific message.

---

## Why local_sim mode matters (three real bugs it caught)

The LLM client (`app/llm_client.py`) has three modes, same response shape
from each: `mock` (deterministic, $0, no model call — proves the plumbing),
`local_sim` (a real Ollama call for varied output, with token counts run
through Anthropic's free `count_tokens` endpoint to simulate real cost under
Claude's actual tokenizer), and `live` (real hosted API calls — intentionally
not built yet, see below).

91 tests pass in mock mode. None of the following three bugs were visible in
that suite — they only surfaced by actually running the agent against
`llama3.1:8b` and watching it produce real, unpredictable output:

**Bug 1 — silent error-swallowing.** `query_executor.py`'s subprocess error
handling only captured `stderr`. MetricFlow writes its actual error text to
`stdout`. A failing query returned `QueryExecutionError: mf query failed
(exit 1):` with nothing after the colon — the real reason was being thrown
away. Fixed to capture both streams.

**Bug 2 — unvalidated filter operators.** The proposal schema was designed
around `"equals"` filters only, but nothing enforced that. A local model,
asked about "last quarter," invented a `"during quarter"` operator with a
relative-time value. It slipped past validation and hit MetricFlow's raw
engine, producing exactly the same swallowed-error problem above. Fixed with
an explicit allow-list, checked before any subprocess call.

**Bug 3 — time dimensions leaking into filters and group-by.** Even after
fixing the operator, the same underlying confusion resurfaced: a model
correctified the *operator* to `"equals"` while still putting a relative-time
*string* ("last quarter") as the filter *value* for a time-typed dimension.
Checking operator alone wasn't enough — dimension *type* had to be checked
too. Extended the guardrail to expose dimension types from the manifest and
reject time-typed dimensions in both filters and the group-by list,
regardless of how they're phrased.

The actual fix for expressing time windows correctly: the proposal schema
gained explicit `start_date`/`end_date` fields, with the model instructed to
resolve relative expressions ("last quarter") against a configurable
reference date (`AGENT_REFERENCE_DATE`) into explicit dates — never through
filters or dimensions.

---

## Summarization reliability: query correctness vs. answer faithfulness

This is the most important finding from end-to-end testing, and worth being
fully honest about rather than glossing over.

Across six demo questions run through the live chat UI in `local_sim` mode:

`![chat flow: total spend by month, Region 3 last quarter, denial rate by region](screenshots/q1-q3-chat-flow.png)`
`![chat flow continued: days to adjudication, PMPM rejection, no-match rejection](screenshots/q5-and-q6-rejections.png)`


| Question | Query layer | Summary accuracy |
|---|---|---|
| Total spend by month | ✅ correct, guardrail-passed | ❌ **fabricated** — reported figures that don't match the actual rows at all, and erased the April anomaly entirely ($57.5K reported vs. $166K actual) |
| Total spend, Region 3, last quarter | ✅ correct | ✅ exact match ($172,182) |
| Denial rate by region | ✅ correct | ✅ exact match, all four regions |
| Days to adjudication | ✅ correct (query silently narrowed to June-only) | ⚠️ correct number (10.6 days), wrong comparative claim ("Region 3 shortest" — actually Region 4) |
| PMPM by region | — rejected: guardrail-approved, execution-unjoinable | (see guardrail section) |
| Therapy adherence rate | — rejected: no confident match | (see guardrail section) |

Every query that reached execution was independently verified against raw
DuckDB output and was correct, every time, no exceptions. **The guardrail's
job — validating that a proposed query is legitimate — is fully proven and
reliable.** What's *not* currently guaranteed is that the natural-language
summary faithfully reports the numbers it was given. `summarize_result`
hands the correct raw rows to `llama3.1:8b` and asks for plain English; the
model is capable of inventing plausible-looking numbers instead of reporting
the real ones, especially when the task requires arithmetic (summing 24 rows
across 4 regions into 6 monthly totals) rather than just reading a single
value.

This is a genuine, distinct trust boundary: **schema validity and semantic
relevance are different guarantees, and so are query correctness and answer
faithfulness.** The guardrail was designed to solve the first pair. The
second pair — whether the LLM's prose response accurately reflects the data
it was handed — has no corresponding safeguard today. That's a concrete,
evidence-backed reason to expect a hosted frontier model to perform
meaningfully better specifically at the summarization step, independent of
the query-proposal reliability improvements discussed below.

---

## Path to a hosted model (live mode)

`live` mode exists in the code today as an intentional, clearly-marked stub
— `_live_propose_query` and `_live_summarize_result` both raise
`NotImplementedError` with an explanation, and the agent graph catches that
gracefully (a clean in-chat rejection message, not a crash) rather than
half-implementing something untested. The chat UI's `MODE` badge
(`GUARDRAIL: ARMED` / `MODE: LOCAL_SIM · llama3.1:8b`) already reflects
whichever mode is active via a `/mode` endpoint, so switching to live mode
requires no UI changes — the badge would simply read `MODE: LIVE ·
claude-sonnet-5`.

What actually changes when wiring up a real API key:

- **Structured JSON reliability.** `local_sim` forces Ollama's `format:
  "json"` mode for the query-proposal step but deliberately *not* for
  summarization (an earlier bug — see git history — showed forcing JSON on a
  free-text prompt produces degenerate `{}` output). Hosted frontier models
  are markedly more reliable at strict JSON than local Llama-class models,
  so production should see far fewer malformed-output retries than local dev
  does.
- **Summarization faithfulness.** Per the finding above, this is the
  strongest argument for live mode specifically — not just cost or
  convenience, but demonstrated accuracy on the exact task (faithfully
  reporting multi-row structured data in prose) that a small local model
  measurably struggled with.
- **Real cost tracking, not simulated.** The `count_tokens`-based cost
  simulation in `local_sim` mode becomes real dollar tracking in `live` mode
  with zero structural change — the same cumulative-cost-log pattern applies,
  just against real API responses instead of a free token-count endpoint.
- **What doesn't change:** the guardrail, query executor, cache, and agent
  orchestration are all completely decoupled from which LLM mode is active.
  Implementing `live` mode is a scoped change to two functions in
  `llm_client.py` — nothing else in the architecture needs to move.

---

## Caching

`app/cache.py` is a deliberately simple in-memory dict, keyed by a canonical
serialization of `(metric, dimensions, filters, time_grain, start_date,
end_date)` — explicitly excluding `source`, so a `mock`-mode and
`local_sim`-mode run of "the same" query hit the same cache entry. Wired into
the agent as its own explicit LangGraph node (`check_cache`), matching the
architecture diagram's own drawn flow rather than hiding inside the executor.

Measured effect on a repeated query: **11.9 seconds → 0.006 seconds**
(cache miss vs. cache hit, same question, same session).

This is a portfolio-scale choice, not a production one, and the README says
so rather than pretending otherwise: this cache is process-local (doesn't
survive a restart, isn't shared across multiple API workers). Redis is the
natural production upgrade path.

---

## Tableau: proving single source of truth

Two dashboards connect directly to the same `claims_metrics.duckdb` file the
API queries — not a CSV export, not a copy, the identical physical storage —
via the community-maintained DuckDB JDBC connector
([motherduckdb/duckdb-tableau-connector](https://github.com/motherduckdb/duckdb-tableau-connector)),
since Tableau has no first-party DuckDB connector.

`![overview dashboard](screenshots/overview-dashboard.png)`
`![anomaly dashboard](screenshots/anomaly-dashboard.png)`

Every KPI on the Overview dashboard was independently cross-checked against
the semantic layer's own output:

| KPI | Dashboard | Semantic layer | Match |
|---|---|---|---|
| Total Spend | $361,032.94 | $361,032.94 | ✅ exact |
| PMPM | 124.88 | 124.88168 | ✅ exact |
| Denial Rate | 7.90% | 7.90% | ✅ exact |
| Days to Adjudication | 10.88 | 10.8775 | ✅ exact |

**One open discrepancy, noted honestly rather than hidden:** the Anomaly
dashboard's Region 3 / April tooltip shows $151K total spend and $739.82
cost-per-claim, versus $153,739.11 / $746.31 confirmed independently through
both the semantic layer and raw DuckDB. A consistent ~1.8% gap in the same
direction on both figures — likely a date-boundary difference between how a
Tableau field truncates `submitted_date` versus the SQL `date_trunc('month',
...)` used everywhere else, though the exact cause wasn't pinned down.
Everything else — claim volume, chart shapes, region/denial/adjudication
orderings — matched exactly. Worth resolving before treating this dashboard
as a fully audited artifact.

---

## Tool choices and rationale

| Layer | Tool | Why |
|---|---|---|
| Local dev warehouse | **DuckDB** | Free, zero cloud cost, fast local iteration, embedded engine — no separate DB server. |
| Transformation + semantic layer | **dbt Core + MetricFlow** | Define a metric once, query it consistently from anywhere. The actual differentiator. |
| Dashboard | **Tableau** | Certified in this tool; matches JD requirements seen across other applications. |
| API layer | **FastAPI** | Thin wrapper — no new logic in the API layer itself, just HTTP around the agent. |
| Agent orchestration | **LangGraph** | Small, purposeful graph: propose → validate → cache-check → execute-or-reject → summarize. Not a framework showcase. |
| Guardrail validation | **Plain Python against the semantic manifest** | Deliberately not another LLM call — deterministic and auditable, not itself probabilistic. |
| Query caching | **In-memory dict** | Proportionate to portfolio scale; Redis is the documented production path. |
| Observability | **structlog** (planned) | Consistent with the sibling `cost-anomaly-monitor` project's logging pattern. |

---

## Known limitations

- **Summarization faithfulness is not guarded.** See the dedicated section
  above — this is the most important limitation in the project and the
  primary argument for live mode.
- **Cache is process-local**, not shared across restarts or workers.
- **Synthetic data only** — no real claims data was used or appropriate for
  this project; the Region 3/April anomaly is deliberately baked in to give
  the demo something concrete to explain, not evidence of real-world
  utilization patterns.
- **Filters only support `"equals"`** — no ranges, no comparisons. Time
  windows are the one exception, expressed via `start_date`/`end_date`.
- **`live` mode is unimplemented** — intentionally, per the build order
  (prove `local_sim` first). Stubbed gracefully, not half-built.
- **One unresolved Tableau/semantic-layer discrepancy** (Region 3/April, ~1.8%
  gap) — see the Tableau section.

---

## Setup

```bash
# environment
cd claims_metrics
python3 -m venv venv && source venv/bin/activate
pip install dbt-metricflow dbt-duckdb fastapi "uvicorn[standard]" httpx langgraph requests anthropic pytest

# add DBT_PROFILES_DIR to the venv so it's always set
echo 'export DBT_PROFILES_DIR=~/.dbt' >> venv/bin/activate

# profiles.yml
mkdir -p ~/.dbt
cat > ~/.dbt/profiles.yml << 'EOF'
claims_metrics:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: 'claims_metrics.duckdb'
      threads: 4
EOF

# build the semantic layer
dbt deps
python3 scripts/generate_synthetic_data.py
dbt build

# run the full test suite
cd ..
python3 -m pytest tests/ -v

# start the API + chat UI
LLM_MODE=local_sim OLLAMA_MODEL=llama3.1:8b MF_BINARY=$(pwd)/venv/bin/mf \
  uvicorn app.api:app --reload
# visit http://localhost:8000
```

**Note:** the app deliberately uses `CLAIMS_DBT_PROJECT_DIR`, not
`DBT_PROJECT_DIR` — the latter collides with a dbt-core-native environment
variable and caused a real, confusing bug during development (dbt's own
project-directory resolution silently interfered with the subprocess `cwd`
override). Also: on systems with LaTeX installed, `mf` collides with
METAFONT's binary of the same name — the venv's `activate` script puts the
correct one first on `PATH`, but this is worth knowing if `mf` commands ever
behave strangely outside an activated venv.

---

## What's next

- Implement `live` mode against a real Anthropic API key, specifically to
  test whether the summarization-faithfulness finding above actually
  resolves with a hosted frontier model.
- Resolve the Region 3/April Tableau discrepancy.
- Redis for production-scale caching.
- structlog observability, matching the sibling `cost-anomaly-monitor`
  project.
