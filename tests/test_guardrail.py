"""
Tests for app.guardrail.

Uses a small hand-crafted fixture manifest (not the real project's
semantic_manifest.json) so these tests are fast, isolated, and don't
require `dbt parse` to have been run first. A separate integration test
at the bottom checks the guardrail against the real generated manifest.
"""

import json
from pathlib import Path

import pytest

from app.guardrail import (
    SemanticManifest,
    ValidationResult,
    dimension_exists_for_metric,
    metric_exists,
    valid_time_grain,
    validate_query,
)

FIXTURE_MANIFEST = {
    "semantic_models": [
        {
            "name": "claims",
            "dimensions": [{"name": "submitted_date"}, {"name": "region"}],
            "measures": [
                {"name": "total_spend"},
                {"name": "claim_count"},
                {"name": "denied_claim_count"},
            ],
        },
        {
            "name": "member_months",
            "dimensions": [{"name": "month_start"}, {"name": "region"}],
            "measures": [{"name": "member_month_count"}],
        },
    ],
    "metrics": [
        {
            "name": "total_spend",
            "type": "simple",
            "type_params": {"measure": {"name": "total_spend"}},
        },
        {
            "name": "claim_count",
            "type": "simple",
            "type_params": {"measure": {"name": "claim_count"}},
        },
        {
            "name": "member_months",
            "type": "simple",
            "type_params": {"measure": {"name": "member_month_count"}},
        },
        {
            "name": "pmpm",
            "type": "ratio",
            "type_params": {
                "numerator": {"name": "total_spend"},
                "denominator": {"name": "member_months"},
            },
        },
    ],
}


@pytest.fixture
def manifest(tmp_path: Path) -> SemanticManifest:
    manifest_path = tmp_path / "semantic_manifest.json"
    manifest_path.write_text(json.dumps(FIXTURE_MANIFEST))
    return SemanticManifest(manifest_path)


# --- SemanticManifest loading ---


def test_missing_manifest_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SemanticManifest(tmp_path / "does_not_exist.json")


def test_metric_names(manifest: SemanticManifest):
    assert manifest.metric_names() == {"total_spend", "claim_count", "member_months", "pmpm"}


# --- underlying_measure_names / semantic_models_for_metric ---


def test_underlying_measures_simple_metric(manifest: SemanticManifest):
    assert manifest.underlying_measure_names("total_spend") == ["total_spend"]


def test_underlying_measures_ratio_metric_spans_two_measures(manifest: SemanticManifest):
    measures = manifest.underlying_measure_names("pmpm")
    assert set(measures) == {"total_spend", "member_month_count"}


def test_underlying_measures_unknown_metric_returns_empty(manifest: SemanticManifest):
    assert manifest.underlying_measure_names("does_not_exist") == []


def test_semantic_models_for_simple_metric(manifest: SemanticManifest):
    assert manifest.semantic_models_for_metric("total_spend") == ["claims"]


def test_semantic_models_for_ratio_metric_spans_two_models(manifest: SemanticManifest):
    # pmpm = total_spend (claims model) / member_months (member_months model)
    assert manifest.semantic_models_for_metric("pmpm") == ["claims", "member_months"]


# --- metric_exists ---


def test_metric_exists_true(manifest: SemanticManifest):
    assert metric_exists(manifest, "total_spend") is True


def test_metric_exists_false(manifest: SemanticManifest):
    assert metric_exists(manifest, "average_wait_time") is False


# --- dimension_exists_for_metric ---


def test_dimension_exists_for_simple_metric(manifest: SemanticManifest):
    assert dimension_exists_for_metric(manifest, "total_spend", "region") is True


def test_dimension_not_exists_for_metric(manifest: SemanticManifest):
    assert dimension_exists_for_metric(manifest, "total_spend", "drug_id") is False


def test_dimension_exists_for_ratio_metric_from_either_side(manifest: SemanticManifest):
    # pmpm spans claims + member_months; region exists on both,
    # submitted_date only exists on claims, month_start only on member_months.
    assert dimension_exists_for_metric(manifest, "pmpm", "region") is True
    assert dimension_exists_for_metric(manifest, "pmpm", "submitted_date") is True
    assert dimension_exists_for_metric(manifest, "pmpm", "month_start") is True


def test_dimension_exists_for_unknown_metric_is_false(manifest: SemanticManifest):
    assert dimension_exists_for_metric(manifest, "does_not_exist", "region") is False


# --- dimension_type_for_model / is_time_dimension_for_metric ---


def test_dimension_type_for_model_time():
    fixture_with_types = {
        "semantic_models": [
            {
                "name": "claims",
                "dimensions": [
                    {"name": "submitted_date", "type": "time"},
                    {"name": "region", "type": "categorical"},
                ],
                "measures": [{"name": "total_spend"}],
            }
        ],
        "metrics": [
            {
                "name": "total_spend",
                "type": "simple",
                "type_params": {"measure": {"name": "total_spend"}},
            }
        ],
    }
    import json
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as d:
        p = P(d) / "manifest.json"
        p.write_text(json.dumps(fixture_with_types))
        m = SemanticManifest(p)

        assert m.dimension_type_for_model("claims", "submitted_date") == "time"
        assert m.dimension_type_for_model("claims", "region") == "categorical"
        assert m.dimension_type_for_model("claims", "does_not_exist") is None
        assert m.dimension_type_for_model("does_not_exist_model", "region") is None

        assert m.is_time_dimension_for_metric("total_spend", "submitted_date") is True
        assert m.is_time_dimension_for_metric("total_spend", "region") is False
        assert m.is_time_dimension_for_metric("total_spend", "does_not_exist") is False


# --- valid_time_grain ---


@pytest.mark.parametrize("grain", ["day", "week", "month", "quarter", "year"])
def test_valid_time_grains(grain: str):
    assert valid_time_grain(grain) is True


@pytest.mark.parametrize("grain", ["fortnight", "hour", "", "MONTH"])
def test_invalid_time_grains(grain: str):
    assert valid_time_grain(grain) is False


# --- validate_query (the guardrail's actual entry point) ---


def test_validate_query_accepts_valid_query(manifest: SemanticManifest):
    result = validate_query(manifest, metric="total_spend", dimensions=["region"], time_grain="month")
    assert result.valid is True
    assert result.reason is None
    assert result.resolved_dimensions == ["region"]


def test_validate_query_rejects_unknown_metric(manifest: SemanticManifest):
    result = validate_query(manifest, metric="average_wait_time")
    assert result.valid is False
    assert "average_wait_time" in result.reason
    assert "does not exist" in result.reason


def test_validate_query_rejects_invalid_dimension(manifest: SemanticManifest):
    result = validate_query(manifest, metric="total_spend", dimensions=["drug_id"])
    assert result.valid is False
    assert "drug_id" in result.reason
    assert "not valid for metric" in result.reason


def test_validate_query_gives_specific_hint_for_time_grain_word_used_as_dimension(
    manifest: SemanticManifest,
):
    """
    Found via real local_sim testing: a model proposed dimensions=['month']
    — a time-grain WORD used as a fake dimension name, distinct from
    misusing a real time dimension like submitted_date. The rejection
    message should point at time_grain specifically, not just list valid
    dimensions generically.
    """
    result = validate_query(manifest, metric="total_spend", dimensions=["month"])
    assert result.valid is False
    assert "time granularity" in result.reason
    assert "time_grain='month'" in result.reason


def test_validate_query_rejects_invalid_time_grain(manifest: SemanticManifest):
    result = validate_query(manifest, metric="total_spend", time_grain="fortnight")
    assert result.valid is False
    assert "fortnight" in result.reason


def test_validate_query_accepts_ratio_metric_with_dimension_from_either_side(
    manifest: SemanticManifest,
):
    result = validate_query(manifest, metric="pmpm", dimensions=["month_start"])
    assert result.valid is True


def test_validate_query_with_no_dimensions_or_grain_is_valid(manifest: SemanticManifest):
    result = validate_query(manifest, metric="claim_count")
    assert result.valid is True
    assert result.resolved_dimensions == []


def test_validate_query_rejects_mix_of_valid_and_invalid_dimensions(manifest: SemanticManifest):
    result = validate_query(manifest, metric="total_spend", dimensions=["region", "drug_id"])
    assert result.valid is False
    assert "drug_id" in result.reason
    assert "region" not in result.reason.split("not valid for metric")[0]


# --- integration test against the REAL generated manifest ---


def test_against_real_project_manifest():
    """
    Sanity check against the actual semantic_manifest.json this project
    generates, not the fixture. Skips if dbt parse hasn't been run yet —
    this is a supplementary check, not a substitute for the isolated
    fixture-based tests above.
    """
    real_manifest_path = (
        Path(__file__).resolve().parent.parent
        / "claims_metrics"
        / "target"
        / "semantic_manifest.json"
    )
    if not real_manifest_path.exists():
        pytest.skip("Run `dbt parse` in claims_metrics/ first to generate semantic_manifest.json")

    manifest = SemanticManifest(real_manifest_path)

    for expected_metric in ["total_spend", "pmpm", "denial_rate", "days_to_adjudication"]:
        assert metric_exists(manifest, expected_metric), f"expected metric {expected_metric} missing"

    assert validate_query(manifest, metric="total_spend", dimensions=["region"]).valid
    assert validate_query(manifest, metric="pmpm", dimensions=["region"]).valid
    assert not validate_query(manifest, metric="total_spend", dimensions=["not_a_real_dim"]).valid
    assert not metric_exists(manifest, "average_wait_time_between_refills")
