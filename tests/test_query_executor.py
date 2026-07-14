"""
Tests for app.query_executor.

These are integration tests, not isolated unit tests — execute_query's
entire job is talking to the real `mf` CLI and a real dbt project, so
there's no meaningful way to test it against a fixture the way
test_guardrail.py does. Tests skip cleanly if `mf` isn't on PATH or the
dbt project's semantic_manifest.json hasn't been generated yet, rather
than failing confusingly in environments where they can't run.
"""

import shutil
from pathlib import Path

import pytest

from app.guardrail import SemanticManifest
from app.query_executor import QueryExecutionError, execute_query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DBT_PROJECT_DIR = str(PROJECT_ROOT / "claims_metrics")
DBT_PROFILES_DIR = str(Path.home() / ".dbt")
MANIFEST_PATH = str(PROJECT_ROOT / "claims_metrics" / "target" / "semantic_manifest.json")

# NOTE: 'mf' is a generic name that can collide with unrelated tools on
# PATH (e.g. METAFONT, a LaTeX typesetting tool, is also called `mf` on
# some systems). Prefer the venv's own mf explicitly rather than trusting
# shutil.which to find the right one.
_VENV_MF = PROJECT_ROOT / "venv" / "bin" / "mf"
MF_BINARY = str(_VENV_MF) if _VENV_MF.exists() else (shutil.which("mf") or "mf")


def _mf_available() -> bool:
    return Path(MF_BINARY).exists() and (
        Path(DBT_PROJECT_DIR) / "target" / "semantic_manifest.json"
    ).exists()


pytestmark = pytest.mark.skipif(
    not _mf_available(),
    reason="mf CLI or semantic_manifest.json not available — run `dbt parse` first",
)


def test_execute_simple_metric_no_dimensions():
    proposal = {"metric": "claim_count", "dimensions": [], "filters": [], "time_grain": None}
    rows = execute_query(proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY)
    assert len(rows) == 1
    assert "claim_count" in rows[0]


def test_execute_with_region_dimension_and_filter():
    proposal = {
        "metric": "total_spend",
        "dimensions": ["region"],
        "filters": [{"dimension": "region", "operator": "equals", "value": "Region 3"}],
        "time_grain": "month",
    }
    rows = execute_query(proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY)
    assert len(rows) > 0
    for row in rows:
        assert row["claim__region"] == "Region 3"


def test_execute_pmpm_by_region_raises_clear_error_not_raw_traceback():
    """
    pmpm spans two semantic models (claims, member_months) with different
    primary entities. The guardrail approves 'region' as a dimension for
    pmpm (it exists on both underlying models independently), but
    MetricFlow's join resolution can't actually group a ratio metric by
    it. This must surface as a specific QueryExecutionError, not an
    unhandled MetricFlow engine exception.
    """
    proposal = {"metric": "pmpm", "dimensions": ["region"], "filters": [], "time_grain": "month"}
    with pytest.raises(QueryExecutionError, match="join resolution"):
        execute_query(proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY)


def test_execute_pmpm_with_only_time_grain_succeeds():
    """pmpm CAN be grouped by metric_time — that's resolved separately from named dimensions."""
    proposal = {"metric": "pmpm", "dimensions": [], "filters": [], "time_grain": "month"}
    rows = execute_query(proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY)
    assert len(rows) == 6  # Jan-Jun 2025
    assert "pmpm" in rows[0]


def test_unsupported_filter_operator_rejected_before_calling_mf():
    """
    Found via real local_sim testing: a local model proposed a relative-time
    filter ('last quarter') using a non-'equals' operator our schema was
    never designed to support. This must fail fast with a specific reason,
    not reach MetricFlow's engine and produce a raw, hard-to-read error
    (or worse, silently misinterpret the filter).
    """
    proposal = {
        "metric": "total_spend",
        "dimensions": ["region", "submitted_date"],
        "filters": [
            {"dimension": "region", "operator": "equals", "value": "Region 3"},
            {"dimension": "submitted_date", "operator": "during quarter", "value": "last quarter"},
        ],
        "time_grain": "quarter",
    }
    with pytest.raises(QueryExecutionError, match="not supported"):
        execute_query(proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY)


def test_time_dimension_filter_rejected_even_with_equals_operator():
    """
    A follow-up gap found via real local_sim testing: a local model can
    correct the OPERATOR to 'equals' while still putting a relative-time
    string ('last quarter') as the VALUE for a time-typed dimension. The
    operator check alone doesn't catch this — dimension TYPE must be
    checked too. This should fail instantly, without ever calling mf.

    NOTE: submitted_date deliberately does NOT appear in `dimensions` here
    — only in `filters` — so this test isolates the filter-side check from
    the separate group-by-list check (see
    test_time_dimension_in_group_by_list_rejected below).
    """
    manifest = SemanticManifest(MANIFEST_PATH)
    proposal = {
        "metric": "total_spend",
        "dimensions": ["region"],
        "filters": [
            {"dimension": "region", "operator": "equals", "value": "Region 3"},
            {"dimension": "submitted_date", "operator": "equals", "value": "last quarter"},
        ],
        "time_grain": "quarter",
    }
    with pytest.raises(QueryExecutionError, match="Filter on 'submitted_date'"):
        execute_query(
            proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY, manifest=manifest
        )


def test_execute_with_explicit_date_range():
    """
    The correct way to express 'last quarter' etc: the LLM resolves it to
    explicit start_date/end_date rather than putting it in filters.
    """
    proposal = {
        "metric": "total_spend",
        "dimensions": ["region"],
        "filters": [{"dimension": "region", "operator": "equals", "value": "Region 3"}],
        "time_grain": "month",
        "start_date": "2025-04-01",
        "end_date": "2025-06-30",
    }
    rows = execute_query(proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY)
    assert len(rows) == 3  # April, May, June only
    months = {row["metric_time__month"][:7] for row in rows}
    assert months == {"2025-04", "2025-05", "2025-06"}


def test_malformed_start_date_rejected_before_calling_mf():
    """
    Defends against the model failing to actually resolve a relative time
    expression despite being instructed to — e.g. leaving 'last quarter'
    in start_date instead of an explicit date.
    """
    proposal = {
        "metric": "total_spend",
        "dimensions": [],
        "filters": [],
        "time_grain": "quarter",
        "start_date": "last quarter",
        "end_date": None,
    }
    with pytest.raises(QueryExecutionError, match="not a valid YYYY-MM-DD date"):
        execute_query(proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY)


def test_start_date_after_end_date_rejected():
    proposal = {
        "metric": "total_spend",
        "dimensions": [],
        "filters": [],
        "time_grain": "quarter",
        "start_date": "2025-06-01",
        "end_date": "2025-01-01",
    }
    with pytest.raises(QueryExecutionError, match="after end_date"):
        execute_query(proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY)


def test_time_dimension_in_group_by_list_rejected():
    """
    Found via real local_sim testing: a model can add a time dimension
    (e.g. submitted_date) to the group-by dimensions list even when
    time_grain and start_date/end_date already scope the query correctly.
    This silently returns a far more granular breakdown than intended
    (day-by-day instead of a single quarterly total) rather than erroring —
    so it must be caught explicitly, not just left as a "technically valid"
    but unintended result shape.
    """
    manifest = SemanticManifest(MANIFEST_PATH)
    proposal = {
        "metric": "total_spend",
        "dimensions": ["region", "submitted_date"],
        "filters": [{"dimension": "region", "operator": "equals", "value": "Region 3"}],
        "time_grain": "quarter",
        "start_date": "2025-04-01",
        "end_date": "2025-06-30",
    }
    with pytest.raises(QueryExecutionError, match="group-by dimensions list"):
        execute_query(
            proposal, DBT_PROJECT_DIR, DBT_PROFILES_DIR, mf_binary=MF_BINARY, manifest=manifest
        )
