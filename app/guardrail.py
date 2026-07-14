"""
Guardrail validation logic for the claims-cost intelligence platform.

Deliberately plain Python against MetricFlow's own semantic_manifest.json —
not another LLM call. The guardrail's job is to answer one question with
total determinism: does this proposed query (metric + dimensions + filters +
time grain) actually correspond to something the semantic layer can execute?

If MetricFlow's manifest changes shape in a future version, these functions
are the single place that would need updating — the agent and API layers
never touch the manifest directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_TIME_GRAINS = frozenset({"day", "week", "month", "quarter", "year"})


@dataclass
class ValidationResult:
    """Outcome of validating a proposed semantic-layer query."""

    valid: bool
    reason: str | None = None
    resolved_dimensions: list[str] = field(default_factory=list)


class SemanticManifest:
    """
    Thin wrapper around MetricFlow's semantic_manifest.json.

    Loads once, exposes lookups the guardrail functions need. Does not
    execute queries or talk to DuckDB — read-only metadata access only.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"semantic_manifest.json not found at {self.manifest_path}. "
                "Run `dbt parse` (or `dbt build`) in the dbt project first."
            )
        with open(self.manifest_path) as f:
            self._manifest = json.load(f)

        self._metrics_by_name = {m["name"]: m for m in self._manifest.get("metrics", [])}
        self._semantic_models_by_name = {
            sm["name"]: sm for sm in self._manifest.get("semantic_models", [])
        }
        # measure name -> semantic model name, for tracing metric -> model
        self._measure_to_model: dict[str, str] = {}
        for sm in self._manifest.get("semantic_models", []):
            for measure in sm.get("measures", []):
                self._measure_to_model[measure["name"]] = sm["name"]

    def metric_names(self) -> set[str]:
        return set(self._metrics_by_name.keys())

    def get_metric(self, metric_name: str) -> dict | None:
        return self._metrics_by_name.get(metric_name)

    def underlying_measure_names(self, metric_name: str) -> list[str]:
        """
        Return the measure name(s) a metric ultimately depends on.
        Simple metrics depend on one measure. Ratio metrics depend on
        their numerator and denominator metrics' measures (recursively).
        """
        metric = self.get_metric(metric_name)
        if metric is None:
            return []

        metric_type = metric["type"]
        type_params = metric["type_params"]

        if metric_type == "simple":
            measure = type_params.get("measure")
            return [measure["name"]] if measure else []

        if metric_type == "ratio":
            measures: list[str] = []
            numerator = type_params.get("numerator")
            denominator = type_params.get("denominator")
            if numerator:
                measures.extend(self.underlying_measure_names(numerator["name"]))
            if denominator:
                measures.extend(self.underlying_measure_names(denominator["name"]))
            return measures

        # derived/cumulative/other metric types: not needed for this project's
        # four metrics, but fail closed rather than silently returning [].
        return []

    def semantic_models_for_metric(self, metric_name: str) -> list[str]:
        """Which semantic model(s) a metric's measures ultimately live in."""
        model_names = set()
        for measure_name in self.underlying_measure_names(metric_name):
            model_name = self._measure_to_model.get(measure_name)
            if model_name:
                model_names.add(model_name)
        return sorted(model_names)

    def dimension_names_for_model(self, model_name: str) -> set[str]:
        sm = self._semantic_models_by_name.get(model_name)
        if sm is None:
            return set()
        return {d["name"] for d in sm.get("dimensions", [])}

    def dimension_type_for_model(self, model_name: str, dimension_name: str) -> str | None:
        """Returns 'time', 'categorical', or None if not found on this model."""
        sm = self._semantic_models_by_name.get(model_name)
        if sm is None:
            return None
        for d in sm.get("dimensions", []):
            if d["name"] == dimension_name:
                return d.get("type")
        return None

    def is_time_dimension_for_metric(self, metric_name: str, dimension_name: str) -> bool:
        """
        True if dimension_name resolves to a time-typed dimension on ANY
        semantic model underlying metric_name. Used to reject filters on
        time dimensions early — this schema only supports 'equals' filters
        on categorical dimensions; time windows belong in time_grain.
        """
        for model_name in self.semantic_models_for_metric(metric_name):
            if self.dimension_type_for_model(model_name, dimension_name) == "time":
                return True
        return False


def metric_exists(manifest: SemanticManifest, metric_name: str) -> bool:
    return metric_name in manifest.metric_names()


def dimension_exists_for_metric(
    manifest: SemanticManifest, metric_name: str, dimension_name: str
) -> bool:
    """
    True if dimension_name is a valid dimension for metric_name — i.e. it
    exists on at least one semantic model the metric's measures live in.
    Handles ratio metrics spanning two semantic models (e.g. pmpm).
    """
    model_names = manifest.semantic_models_for_metric(metric_name)
    if not model_names:
        return False
    return any(
        dimension_name in manifest.dimension_names_for_model(model_name)
        for model_name in model_names
    )


def valid_time_grain(time_grain: str) -> bool:
    return time_grain in VALID_TIME_GRAINS


def validate_query(
    manifest: SemanticManifest,
    metric: str,
    dimensions: list[str] | None = None,
    time_grain: str | None = None,
) -> ValidationResult:
    """
    The guardrail's single entry point. Validates a proposed query in the
    same shape the agent will produce: metric + dimensions + time grain.
    (Filters are validated the same way as group-by dimensions — a filter
    on a dimension the metric doesn't have is rejected identically.)

    Fails on the first problem found, with a specific, demoable reason —
    this is what step 8's "caught rejection" README section shows.
    """
    dimensions = dimensions or []

    if not metric_exists(manifest, metric):
        return ValidationResult(
            valid=False,
            reason=f"Metric '{metric}' does not exist in the semantic layer. "
            f"Available metrics: {sorted(manifest.metric_names())}",
        )

    if time_grain is not None and not valid_time_grain(time_grain):
        return ValidationResult(
            valid=False,
            reason=f"'{time_grain}' is not a valid time grain. "
            f"Valid grains: {sorted(VALID_TIME_GRAINS)}",
        )

    invalid_dims = [
        d for d in dimensions if not dimension_exists_for_metric(manifest, metric, d)
    ]
    if invalid_dims:
        model_names = manifest.semantic_models_for_metric(metric)
        available = set()
        for model_name in model_names:
            available |= manifest.dimension_names_for_model(model_name)

        # A recurring, specific failure mode: a model tries to express time
        # bucketing via the dimensions list using a time-grain WORD as a
        # fake dimension name (e.g. "month") rather than the actual
        # time_grain field. This is different from the earlier time-typed-
        # dimension check (that catches reusing a REAL dimension like
        # submitted_date) — here the invented name doesn't exist at all,
        # so give a pointed hint instead of just listing valid dimensions.
        disguised_time_grains = [d for d in invalid_dims if d in VALID_TIME_GRAINS]
        if disguised_time_grains:
            return ValidationResult(
                valid=False,
                reason=f"{disguised_time_grains} looks like a time granularity, not a "
                f"dimension. Use time_grain={disguised_time_grains[0]!r} instead of "
                f"adding it to dimensions. Available dimensions for '{metric}': "
                f"{sorted(available)}",
            )

        return ValidationResult(
            valid=False,
            reason=f"Dimension(s) {invalid_dims} not valid for metric '{metric}'. "
            f"Available dimensions for this metric: {sorted(available)}",
        )

    return ValidationResult(valid=True, resolved_dimensions=dimensions)
