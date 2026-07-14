"""
LLM client for the claims-cost intelligence platform's NL-to-query step.

Three modes, same response shape from every mode:
    {
        "metric": str | None,   # None when nothing clearly matches (see candidate_metrics)
        "candidate_metrics": list[str],   # only populated when metric is None
        "dimensions": list[str],
        "filters": list[dict],   # [{"dimension": ..., "operator": "equals", "value": ...}]
        "time_grain": str | None,
        "start_date": str | None,   # YYYY-MM-DD, resolved from relative time expressions
        "end_date": str | None,     # YYYY-MM-DD
        "source": "mock" | "local_sim" | "live",
    }

The proposal is NOT validated here — that's the guardrail's job, deliberately
kept separate. This client's only responsibility is turning a natural
language question into a *structured* query proposal, which may well be
wrong (a nonexistent metric, an invalid dimension) — the guardrail is what
catches that, not this layer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app import config

# ---------------------------------------------------------------------------
# mock mode: deterministic, rule-based, $0, no model call at all
# ---------------------------------------------------------------------------

_METRIC_KEYWORDS: list[tuple[str, str]] = [
    ("pmpm", "pmpm"),
    ("per member per month", "pmpm"),
    ("per-member-per-month", "pmpm"),
    ("denial", "denial_rate"),
    ("denied", "denial_rate"),
    ("adjudication", "days_to_adjudication"),
    ("adjudicate", "days_to_adjudication"),
    # deliberately NOT a real metric — mirrors the kind of plausible-sounding
    # but fabricated metric name a real LLM might hallucinate, so the mock
    # path can also exercise the guardrail's metric_exists() rejection
    # deterministically, without needing Ollama.
    ("readmission", "readmission_rate"),
    ("spend", "total_spend"),
    ("cost", "total_spend"),
]

# Questions matching these keywords have NO real corresponding metric at all
# — mock mode's version of "the model correctly declines to guess." Maps to
# candidate_metrics rather than a fabricated or wrong-but-real metric name.
_NO_MATCH_KEYWORDS: dict[str, list[str]] = {
    "adherence": ["denial_rate", "days_to_adjudication"],
}

_TIME_GRAIN_KEYWORDS: list[tuple[str, str]] = [
    ("quarter", "quarter"),
    ("annual", "year"),
    ("yearly", "year"),
    ("year", "year"),
    ("week", "week"),
    ("month", "month"),
]

_REGION_PATTERN = re.compile(r"region\s*(\d+)", re.IGNORECASE)


def _mock_propose_query(question: str) -> dict[str, Any]:
    q = question.lower()

    for keyword, candidates in _NO_MATCH_KEYWORDS.items():
        if keyword in q:
            return {
                "metric": None,
                "candidate_metrics": candidates,
                "dimensions": [],
                "filters": [],
                "time_grain": None,
                "start_date": None,
                "end_date": None,
                "source": "mock",
            }

    metric = "total_spend"
    for keyword, mapped_metric in _METRIC_KEYWORDS:
        if keyword in q:
            metric = mapped_metric
            break

    time_grain = "month"
    for keyword, mapped_grain in _TIME_GRAIN_KEYWORDS:
        if keyword in q:
            time_grain = mapped_grain
            break

    dimensions: list[str] = []
    filters: list[dict[str, Any]] = []

    region_match = _REGION_PATTERN.search(question)
    if region_match:
        region_value = f"Region {region_match.group(1)}"
        dimensions.append("region")
        filters.append({"dimension": "region", "operator": "equals", "value": region_value})
    elif "region" in q:
        dimensions.append("region")

    return {
        "metric": metric,
        "candidate_metrics": [],
        "dimensions": dimensions,
        "filters": filters,
        "time_grain": time_grain,
        "start_date": None,
        "end_date": None,
        "source": "mock",
    }


# ---------------------------------------------------------------------------
# local_sim mode: real Ollama call for varied output + free count_tokens
# for accurate simulated cost under Claude's real tokenizer
# ---------------------------------------------------------------------------


def _build_system_prompt(manifest_path: str) -> str:
    """
    Build the system prompt from the semantic layer's OWN metadata, not a
    hardcoded list — so the agent is grounded in whatever metrics actually
    exist right now, and the prompt doesn't silently drift out of sync with
    the semantic layer as metrics are added or renamed.
    """
    import datetime

    from app.guardrail import SemanticManifest

    manifest = SemanticManifest(manifest_path)
    lines = ["Available metrics and their valid GROUP-BY dimensions (categorical only):"]
    for metric_name in sorted(manifest.metric_names()):
        model_names = manifest.semantic_models_for_metric(metric_name)
        categorical_dims: set[str] = set()
        for model_name in model_names:
            for dim_name in manifest.dimension_names_for_model(model_name):
                if manifest.dimension_type_for_model(model_name, dim_name) != "time":
                    categorical_dims.add(dim_name)
        lines.append(f"- {metric_name}: dimensions = {sorted(categorical_dims)}")

    reference_date = config.AGENT_REFERENCE_DATE or datetime.date.today().isoformat()

    return (
        "You are a query planner for a claims-cost analytics semantic layer.\n"
        f"Today's date is {reference_date}. Resolve any relative time expression "
        "(e.g. 'last quarter', 'this year', 'last month') against this date "
        "into explicit start_date and end_date values (YYYY-MM-DD).\n\n"
        "Given a natural language question, respond with ONLY a JSON object "
        "(no markdown fences, no commentary) matching this exact schema:\n"
        '{"metric": "<metric_name or null>", "candidate_metrics": ["<metric_name>", ...], '
        '"dimensions": ["<dim>", ...], '
        '"filters": [{"dimension": "<dim>", "operator": "equals", "value": "<value>"}], '
        '"time_grain": "<day|week|month|quarter|year>", '
        '"start_date": "<YYYY-MM-DD or null>", "end_date": "<YYYY-MM-DD or null>"}\n\n'
        + "\n".join(lines)
        + "\n\nIMPORTANT — do not guess an existing metric that doesn't fit:\n"
        "- If the question CLEARLY matches one of the metrics above, set \"metric\" to "
        "that name and leave \"candidate_metrics\" as an empty list.\n"
        "- If the question does NOT clearly match any metric above, set \"metric\" to "
        "null. Do NOT substitute a real-but-unrelated metric just to give an answer — "
        "an incorrect confident answer is worse than admitting no match. Instead, "
        "populate \"candidate_metrics\" with 1-3 metric names from the list above that "
        "seem loosely related to what the user might actually be asking, so they can "
        "rewrite their question. If truly nothing is related, use an empty list.\n\n"
        "IMPORTANT constraints on filters:\n"
        '- The ONLY supported filter operator is "equals" — no ranges, no relative '
        "time expressions, no comparisons.\n"
        "- NEVER put a time/date dimension in filters — not even with 'equals'. "
        "Time windows belong ONLY in start_date/end_date, never in filters.\n"
        "- Only add a filter when the question names a SPECIFIC categorical value "
        "(e.g. 'Region 3'), never for a time period.\n"
        "- If the question mentions a relative time period, set start_date and "
        "end_date to the resolved explicit dates. If no time period is mentioned, "
        "leave both as null.\n\n"
        "IMPORTANT constraints on dimensions:\n"
        "- The 'dimensions' list above is ALL you may choose from — it is already "
        "categorical-only. Do not add any other dimension, and do not add a date/time "
        "dimension to 'dimensions' even if it seems relevant to the question.\n"
        "- Time bucketing is controlled ENTIRELY by time_grain plus start_date/end_date. "
        "Never try to express time granularity or a time window via the dimensions list.\n\n"
        "Do not invent dimensions not listed above."
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Defensive parse: strip markdown fences, grab the first {...} block."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))


def _call_ollama(prompt: str, system_prompt: str, force_json: bool = True) -> str:
    import requests

    if not config.OLLAMA_MODEL:
        raise RuntimeError(
            "OLLAMA_MODEL is not set. Run `ollama list` to see pulled models, "
            "then `export OLLAMA_MODEL=<name>` (e.g. llama3.1:8b)."
        )

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
    }
    if force_json:
        payload["format"] = "json"

    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=config.OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    output_text = body.get("response", "")
    if not output_text:
        raise RuntimeError("Ollama returned an empty response.")
    return output_text


def _count_tokens_or_estimate(text: str, model: str) -> tuple[int, bool]:
    """
    Returns (token_count, was_stub_estimate). Tries Anthropic's free
    count_tokens endpoint first; falls back to a ~4 chars/token stub
    estimate if no API key is configured or the call fails, so cost-sim
    plumbing is still testable without credentials.
    """
    try:
        import os

        import anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("no API key configured")

        client = anthropic.Anthropic()
        result = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return result.input_tokens, False
    except Exception:
        return max(1, len(text) // 4), True


def _record_cumulative_cost(input_tokens: int, output_tokens: int, model: str, was_stub: bool) -> None:
    pricing = config.MODEL_PRICING.get(model)
    if pricing is None:
        return

    cost = (input_tokens / 1_000_000) * pricing["input"] + (
        output_tokens / 1_000_000
    ) * pricing["output"]

    log_path = Path(config.LOCAL_SIM_COST_LOG_PATH)
    if log_path.exists():
        log = json.loads(log_path.read_text())
    else:
        log = {
            "call_count": 0,
            "cumulative_cost_usd": 0.0,
            "stub_call_count": 0,
            "stub_cumulative_cost_usd": 0.0,
        }

    if was_stub:
        log["stub_call_count"] += 1
        log["stub_cumulative_cost_usd"] += cost
    else:
        log["call_count"] += 1
        log["cumulative_cost_usd"] += cost

    log_path.write_text(json.dumps(log, indent=2))


def _local_sim_propose_query(question: str, manifest_path: str) -> dict[str, Any]:
    system_prompt = _build_system_prompt(manifest_path)
    raw_output = _call_ollama(prompt=question, system_prompt=system_prompt)

    try:
        parsed = _extract_json_object(raw_output)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"local_sim: model output was not valid JSON after defensive parsing: {e}"
        ) from e

    input_tokens, input_was_stub = _count_tokens_or_estimate(
        system_prompt + question, config.COST_SIM_MODEL
    )
    output_tokens, output_was_stub = _count_tokens_or_estimate(raw_output, config.COST_SIM_MODEL)
    _record_cumulative_cost(
        input_tokens, output_tokens, config.COST_SIM_MODEL, was_stub=input_was_stub or output_was_stub
    )

    return {
        "metric": parsed.get("metric"),
        "candidate_metrics": parsed.get("candidate_metrics", []),
        "dimensions": parsed.get("dimensions", []),
        "filters": parsed.get("filters", []),
        "time_grain": parsed.get("time_grain"),
        "start_date": parsed.get("start_date"),
        "end_date": parsed.get("end_date"),
        "source": "local_sim",
    }


# ---------------------------------------------------------------------------
# live mode: real Anthropic API call — deferred, per build order
# ---------------------------------------------------------------------------


def _live_propose_query(question: str, manifest_path: str) -> dict[str, Any]:
    raise NotImplementedError(
        "live mode is intentionally not built yet — local_sim is proven first, "
        "per the project's build order."
    )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def propose_query(question: str, manifest_path: str | None = None) -> dict[str, Any]:
    manifest_path = manifest_path or config.SEMANTIC_MANIFEST_PATH

    if config.LLM_MODE == "mock":
        return _mock_propose_query(question)
    if config.LLM_MODE == "local_sim":
        return _local_sim_propose_query(question, manifest_path)
    if config.LLM_MODE == "live":
        return _live_propose_query(question, manifest_path)

    raise ValueError(f"Unknown LLM_MODE: {config.LLM_MODE!r}")


# ---------------------------------------------------------------------------
# summarize_result: turns query rows into a plain-English answer.
# Same three-mode pattern; free text here (not structured JSON) since this
# is presentation, not something a downstream system parses.
# ---------------------------------------------------------------------------


def _mock_summarize_result(question: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No data was returned for this query."
    if len(rows) == 1:
        return f"Result: {rows[0]}"
    lines = [f"- {row}" for row in rows]
    return "Here's what the data shows:\n" + "\n".join(lines)


def _local_sim_summarize_result(question: str, rows: list[dict[str, Any]]) -> str:
    system_prompt = (
        "You are a claims-cost analytics assistant. Given a user's question and "
        "the query results (already correct — do not second-guess the numbers), "
        "write a brief, plain-English answer. 2-4 sentences. No markdown, no JSON."
    )
    prompt = f"Question: {question}\n\nQuery results (JSON): {json.dumps(rows)}"
    raw_output = _call_ollama(prompt=prompt, system_prompt=system_prompt, force_json=False)

    input_tokens, input_was_stub = _count_tokens_or_estimate(
        system_prompt + prompt, config.COST_SIM_MODEL
    )
    output_tokens, output_was_stub = _count_tokens_or_estimate(raw_output, config.COST_SIM_MODEL)
    _record_cumulative_cost(
        input_tokens, output_tokens, config.COST_SIM_MODEL, was_stub=input_was_stub or output_was_stub
    )

    return raw_output.strip()


def _live_summarize_result(question: str, rows: list[dict[str, Any]]) -> str:
    raise NotImplementedError(
        "live mode is intentionally not built yet — local_sim is proven first, "
        "per the project's build order."
    )


def summarize_result(question: str, rows: list[dict[str, Any]]) -> str:
    if config.LLM_MODE == "mock":
        return _mock_summarize_result(question, rows)
    if config.LLM_MODE == "local_sim":
        return _local_sim_summarize_result(question, rows)
    if config.LLM_MODE == "live":
        return _live_summarize_result(question, rows)

    raise ValueError(f"Unknown LLM_MODE: {config.LLM_MODE!r}")
