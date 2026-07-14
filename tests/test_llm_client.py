"""
Tests for app.llm_client's mock mode.

local_sim and live mode aren't tested here — local_sim requires a running
Ollama instance (tested on the actual dev machine, not this suite), and
live mode is intentionally unimplemented per the project's build order.
"""

import pytest

from app.llm_client import propose_query
from app import config


@pytest.fixture(autouse=True)
def force_mock_mode(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "mock")


def test_mock_returns_expected_shape():
    result = propose_query("What is total spend?")
    assert set(result.keys()) == {
        "metric",
        "candidate_metrics",
        "dimensions",
        "filters",
        "time_grain",
        "start_date",
        "end_date",
        "source",
    }
    assert result["source"] == "mock"


def test_mock_is_deterministic():
    q = "What is PMPM for Region 3 this year?"
    assert propose_query(q) == propose_query(q)


@pytest.mark.parametrize(
    "question,expected_metric",
    [
        ("What is total spend by region?", "total_spend"),
        ("What is PMPM this quarter?", "pmpm"),
        ("per member per month cost", "pmpm"),
        ("What is the denial rate?", "denial_rate"),
        ("How many claims were denied?", "denial_rate"),
        ("Average days to adjudication?", "days_to_adjudication"),
        ("How long to adjudicate claims?", "days_to_adjudication"),
        ("What's the cost trend?", "total_spend"),
    ],
)
def test_mock_metric_keyword_mapping(question, expected_metric):
    assert propose_query(question)["metric"] == expected_metric


def test_mock_defaults_to_total_spend_when_no_keyword_matches():
    result = propose_query("Tell me something interesting about claims.")
    assert result["metric"] == "total_spend"


@pytest.mark.parametrize(
    "question,expected_grain",
    [
        ("Spend by quarter", "quarter"),
        ("Spend this year", "year"),
        ("Annual spend", "year"),
        ("Weekly spend", "week"),
        ("Monthly spend", "month"),
        ("Spend overall", "month"),  # default
    ],
)
def test_mock_time_grain_keyword_mapping(question, expected_grain):
    assert propose_query(question)["time_grain"] == expected_grain


def test_mock_extracts_specific_region_as_filter():
    result = propose_query("What is total spend in Region 3?")
    assert result["dimensions"] == ["region"]
    assert result["filters"] == [{"dimension": "region", "operator": "equals", "value": "Region 3"}]


def test_mock_generic_region_mention_adds_dimension_without_filter():
    result = propose_query("What is total spend by region?")
    assert result["dimensions"] == ["region"]
    assert result["filters"] == []


def test_mock_no_region_mention_has_no_region_dimension():
    result = propose_query("What is total spend?")
    assert "region" not in result["dimensions"]
    assert result["filters"] == []


def test_mock_no_match_returns_null_metric_with_verified_candidates():
    """
    This is intentional: 'adherence' has no corresponding real metric.
    Rather than substituting a real-but-wrong metric (the old behavior,
    which produced confident-but-unhelpful answers), the model should
    explicitly decline with metric=None and offer candidate metrics for
    the user to consider instead.
    """
    result = propose_query("What is the average therapy adherence rate?")
    assert result["metric"] is None
    assert len(result["candidate_metrics"]) > 0


def test_mock_deliberately_proposes_fabricated_metric_for_readmission_questions():
    """
    'readmission' maps to a plausible-sounding metric name that does NOT
    exist in the semantic layer — simulating a real LLM hallucinating a
    specific (as opposed to declining) wrong metric. This preserves
    coverage of the guardrail's metric_exists() rejection path even in
    mock mode, separate from the no-match/candidates path above.
    """
    result = propose_query("What is the readmission rate?")
    assert result["metric"] == "readmission_rate"
    assert result["candidate_metrics"] == []


def test_unknown_llm_mode_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "not_a_real_mode")
    with pytest.raises(ValueError, match="Unknown LLM_MODE"):
        propose_query("anything")
