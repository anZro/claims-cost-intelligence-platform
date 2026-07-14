"""
Tests for app.agent — the full LangGraph flow.

Integration tests (real mf, real dbt project), not isolated unit tests,
same reasoning as test_query_executor.py. Forces LLM_MODE=mock so these
run deterministically with no Ollama dependency; local_sim mode isn't
covered here since it requires a running local model.
"""

import shutil
from pathlib import Path

import pytest

from app import config
from app import cache as cache_module
from app.agent import ask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DBT_PROJECT_DIR = str(PROJECT_ROOT / "claims_metrics")
DBT_PROFILES_DIR = str(Path.home() / ".dbt")
MANIFEST_PATH = str(PROJECT_ROOT / "claims_metrics" / "target" / "semantic_manifest.json")

_VENV_MF = PROJECT_ROOT / "venv" / "bin" / "mf"
MF_BINARY = str(_VENV_MF) if _VENV_MF.exists() else (shutil.which("mf") or "mf")


def _mf_available() -> bool:
    return Path(MF_BINARY).exists() and Path(MANIFEST_PATH).exists()


pytestmark = pytest.mark.skipif(
    not _mf_available(),
    reason="mf CLI or semantic_manifest.json not available — run `dbt parse` first",
)


@pytest.fixture(autouse=True)
def force_mock_mode(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "mock")


@pytest.fixture(autouse=True)
def clear_cache():
    # Prevents test-order dependencies: without this, a test could get a
    # cache hit from a proposal an earlier test executed, which produces
    # correct data either way but means these tests wouldn't reliably
    # exercise the same code path in isolation.
    cache_module.clear()
    yield
    cache_module.clear()


def _ask(question: str):
    return ask(
        question,
        manifest_path=MANIFEST_PATH,
        dbt_project_dir=DBT_PROJECT_DIR,
        dbt_profiles_dir=DBT_PROFILES_DIR,
        mf_binary=MF_BINARY,
    )


def test_valid_question_returns_rows_and_summary():
    result = _ask("What is total spend by region?")
    assert result["valid"] is True
    assert result["rows"] is not None
    assert len(result["rows"]) > 0
    assert result["summary"] is not None
    assert "I can't answer" not in result["summary"]


def test_guardrail_rejects_fabricated_metric_with_clear_reason():
    """The step 8 thesis: a fabricated metric is caught, not silently answered."""
    result = _ask("What is the readmission rate for Region 2?")
    assert result["valid"] is False
    assert result.get("rows") is None
    assert "readmission_rate" in result["rejection_reason"]
    assert "does not exist" in result["rejection_reason"]
    assert "I can't answer that as asked" in result["summary"]


def test_no_match_question_gets_verified_candidate_suggestions_not_a_wrong_answer():
    """
    A DIFFERENT failure mode than a fabricated metric: the question doesn't
    clearly match anything, so the model should decline (metric=None) and
    offer real, verified candidate metrics — never a confident-but-wrong
    answer using an existing-but-unrelated metric.
    """
    result = _ask("What is the average therapy adherence rate?")
    assert result["valid"] is False
    assert result["proposal"]["metric"] is None
    assert result.get("rows") is None
    assert "Did you mean one of" in result["rejection_reason"]
    assert "I can't answer that as asked" in result["summary"]


def test_execution_layer_rejects_unjoinable_dimension_with_clear_reason():
    """
    pmpm grouped by region passes the guardrail (region exists on both
    underlying semantic models) but MetricFlow can't actually join it —
    the agent must still fail closed with a specific reason, not crash.
    """
    result = _ask("What is PMPM by region this year?")
    assert result["valid"] is False
    assert result.get("rows") is None
    assert "join resolution" in result["rejection_reason"]


def test_valid_question_with_no_dimensions_or_filters():
    # mock mode defaults time_grain to "month" when nothing else matches,
    # so this returns one row per month (Jan-Jun 2025), not a single total.
    result = _ask("What is the denial rate?")
    assert result["valid"] is True
    assert result["rows"] is not None
    assert len(result["rows"]) == 6


def test_proposal_is_always_present_regardless_of_outcome():
    """proposal should be set whether the question succeeds or is rejected."""
    valid_result = _ask("What is total spend?")
    rejected_result = _ask("What is the readmission rate?")
    assert valid_result["proposal"] is not None
    assert rejected_result["proposal"] is not None


def test_live_mode_fails_gracefully_not_a_crash(monkeypatch):
    """
    live mode is intentionally unimplemented. Setting LLM_MODE=live must
    surface as a normal rejection with a clear message — not an unhandled
    NotImplementedError crashing the whole graph invocation.
    """
    monkeypatch.setattr(config, "LLM_MODE", "live")
    result = _ask("What is total spend?")
    assert result["valid"] is False
    assert result["proposal"] is None
    assert "live mode" in result["rejection_reason"]
    assert "I can't answer that as asked" in result["summary"]
