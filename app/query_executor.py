"""
Executes an already-validated query proposal against MetricFlow.

Deliberately only called AFTER app.guardrail.validate_query has passed —
this module trusts the guardrail's yes/no completely and does no name
validation of its own. But it handles one important gap the guardrail
intentionally doesn't cover: the guardrail checks whether a dimension
exists *anywhere* on a metric's underlying semantic model(s), which is
right for catching fabricated names, but MetricFlow's actual join
resolution can still refuse to execute some guardrail-approved
combinations — e.g. a ratio metric spanning two differently-keyed
semantic models (pmpm) often can't be grouped by a dimension that exists
on both models independently, because there's no single join path MetricFlow
will use automatically. This module asks MetricFlow itself (via
`mf list dimensions --metrics <metric>`) for the authoritative,
entity-qualified dimension names before executing, and raises a clear,
specific QueryExecutionError if a guardrail-approved dimension isn't
actually resolvable — rather than letting MetricFlow's raw engine
traceback bubble up to the agent.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.guardrail import SemanticManifest


class QueryExecutionError(Exception):
    pass


def _run_mf(cmd: list[str], dbt_project_dir: str, dbt_profiles_dir: str, timeout: int = 60):
    env = os.environ.copy()
    # dbt-core natively recognizes DBT_PROJECT_DIR. We control the project
    # location via subprocess cwd instead — an inherited DBT_PROJECT_DIR
    # from the caller's shell (e.g. left over from unrelated dbt work) can
    # silently conflict with that. Scrub it so cwd is the single source of
    # truth, rather than relying on nobody ever having it set.
    env.pop("DBT_PROJECT_DIR", None)
    env["DBT_PROFILES_DIR"] = dbt_profiles_dir
    return subprocess.run(
        cmd, cwd=dbt_project_dir, env=env, capture_output=True, text=True, timeout=timeout
    )


def _get_qualified_dimensions(
    metric: str, dbt_project_dir: str, dbt_profiles_dir: str, mf_binary: str
) -> dict[str, str]:
    """
    Returns {bare_dimension_name: entity_qualified_name}, e.g.
    {'region': 'claim__region', 'submitted_date': 'claim__submitted_date'}.
    metric_time is intentionally excluded — it's handled separately via
    time_grain, not as a named dimension.
    """
    result = _run_mf(
        [mf_binary, "list", "dimensions", "--metrics", metric],
        dbt_project_dir,
        dbt_profiles_dir,
    )
    if result.returncode != 0:
        raise QueryExecutionError(
            f"Could not list dimensions for metric '{metric}': {result.stderr.strip()}"
        )

    qualified_names = re.findall(r"^•\s*(\S+)\s*$", result.stdout, re.MULTILINE)
    mapping: dict[str, str] = {}
    for qname in qualified_names:
        if qname == "metric_time":
            continue
        if "__" in qname:
            bare_name = qname.split("__", 1)[1]
            mapping[bare_name] = qname
    return mapping


SUPPORTED_FILTER_OPERATORS = frozenset({"equals"})
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date_range(start_date: str | None, end_date: str | None) -> None:
    for label, value in [("start_date", start_date), ("end_date", end_date)]:
        if value is not None and not _DATE_PATTERN.match(value):
            raise QueryExecutionError(
                f"{label}={value!r} is not a valid YYYY-MM-DD date. "
                f"The model may have failed to resolve a relative time expression "
                f"into an explicit date."
            )
    if start_date and end_date and start_date > end_date:
        raise QueryExecutionError(
            f"start_date ({start_date}) is after end_date ({end_date}) — "
            f"the model likely resolved a relative time expression incorrectly."
        )


def _build_where_clauses(filters: list[dict[str, Any]], qualified_dims: dict[str, str]) -> list[str]:
    clauses = []
    for f in filters:
        bare_dim = f["dimension"]
        operator = f.get("operator", "equals")
        value = f["value"]

        if operator not in SUPPORTED_FILTER_OPERATORS:
            raise QueryExecutionError(
                f"Filter operator '{operator}' on dimension '{bare_dim}' is not supported. "
                f"Only {sorted(SUPPORTED_FILTER_OPERATORS)} filters are currently handled — "
                f"relative time expressions like 'last quarter' aren't parsed into date ranges "
                f"by this schema. Time windows should be expressed via time_grain, not filters."
            )

        qualified = qualified_dims.get(bare_dim)
        if qualified is None:
            raise QueryExecutionError(
                f"Dimension '{bare_dim}' passed the guardrail but MetricFlow's join "
                f"resolution can't actually filter on it for this metric. "
                f"Resolvable dimensions: {sorted(qualified_dims.keys())}"
            )
        clauses.append(f"{{{{ Dimension('{qualified}') }}}} = '{value}'")
    return clauses


def execute_query(
    proposal: dict[str, Any],
    dbt_project_dir: str,
    dbt_profiles_dir: str,
    mf_binary: str = "mf",
    manifest: "SemanticManifest | None" = None,
) -> list[dict[str, Any]]:
    metric = proposal["metric"]
    dimensions = list(proposal.get("dimensions", []))
    time_grain = proposal.get("time_grain")
    filters = proposal.get("filters", [])
    start_date = proposal.get("start_date")
    end_date = proposal.get("end_date")

    _validate_date_range(start_date, end_date)

    # Fail fast, before any subprocess calls: this schema never supports
    # time dimensions in the group-by list OR in filters (only time_grain +
    # start_date/end_date express time). Checking dimension TYPE, not just
    # operator/list-membership, catches cases like a model adding a time
    # dimension to `dimensions` (silently returns an unintended, overly
    # granular breakdown) or proposing operator="equals" with a relative-time
    # VALUE in filters — both pass a naive check but are wrong for a date column.
    if manifest is not None:
        for dim in dimensions:
            if manifest.is_time_dimension_for_metric(metric, dim):
                raise QueryExecutionError(
                    f"'{dim}' is a time dimension and cannot be used in the "
                    f"group-by dimensions list — use time_grain to control time "
                    f"granularity instead."
                )
        for f in filters:
            if manifest.is_time_dimension_for_metric(metric, f["dimension"]):
                raise QueryExecutionError(
                    f"Filter on '{f['dimension']}' is not supported — it's a time "
                    f"dimension. Time windows must be expressed via start_date/"
                    f"end_date, not as a filter value (got value={f['value']!r})."
                )

    qualified_dims = _get_qualified_dimensions(metric, dbt_project_dir, dbt_profiles_dir, mf_binary)

    group_by: list[str] = []
    for dim in dimensions:
        qualified = qualified_dims.get(dim)
        if qualified is None:
            raise QueryExecutionError(
                f"Dimension '{dim}' passed the guardrail but MetricFlow's join "
                f"resolution can't actually group '{metric}' by it — likely because "
                f"'{metric}' spans multiple semantic models with no single join path "
                f"for this dimension. Resolvable dimensions for '{metric}': "
                f"{sorted(qualified_dims.keys())}"
            )
        group_by.append(qualified)
    if time_grain:
        group_by.append(f"metric_time__{time_grain}")

    where_clauses = _build_where_clauses(filters, qualified_dims)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = tmp.name

    try:
        cmd = [mf_binary, "query", "--metrics", metric, "--csv", csv_path]
        if group_by:
            cmd += ["--group-by", ",".join(group_by)]
        for clause in where_clauses:
            cmd += ["--where", clause]
        if start_date:
            cmd += ["--start-time", start_date]
        if end_date:
            cmd += ["--end-time", end_date]

        result = _run_mf(cmd, dbt_project_dir, dbt_profiles_dir)
        if result.returncode != 0:
            # MetricFlow writes its actual error text to stdout, not stderr —
            # include both so the real reason isn't silently swallowed.
            detail = (result.stdout.strip() + "\n" + result.stderr.strip()).strip()
            raise QueryExecutionError(f"mf query failed (exit {result.returncode}): {detail}")

        rows: list[dict[str, Any]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    finally:
        Path(csv_path).unlink(missing_ok=True)
