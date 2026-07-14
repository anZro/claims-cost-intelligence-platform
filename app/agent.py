"""
LangGraph agent for the claims-cost intelligence platform.

Flow: NL question -> propose query -> guardrail validate -> cache check ->
execute on miss (or reject) -> summarize. Deliberately a small, purposeful
graph — parse, validate, cache, execute, summarize — not a framework
showcase. Same pattern used in the cost-anomaly-monitor sibling project.

The guardrail is the load-bearing safety mechanism: it's a fully separate,
deterministic function (app.guardrail.validate_query) that this graph calls
and obeys. The agent cannot bypass it — a rejected proposal never reaches
the cache or execute_query.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app import cache
from app.guardrail import SemanticManifest, validate_query
from app.llm_client import propose_query, summarize_result
from app.query_executor import QueryExecutionError, execute_query


class AgentState(TypedDict, total=False):
    question: str
    proposal: dict[str, Any] | None
    valid: bool | None
    rejection_reason: str | None
    rows: list[dict[str, Any]] | None
    cache_hit: bool | None
    summary: str | None


def build_agent_graph(
    manifest_path: str,
    dbt_project_dir: str,
    dbt_profiles_dir: str,
    mf_binary: str = "mf",
):
    """
    Returns a compiled LangGraph app. Config (manifest path, dbt project
    location, mf binary) is bound at build time via closures, kept out of
    per-request state — state only carries what changes per question.
    """
    manifest = SemanticManifest(manifest_path)

    def propose_node(state: AgentState) -> AgentState:
        try:
            proposal = propose_query(state["question"], manifest_path=manifest_path)
            return {"proposal": proposal}
        except NotImplementedError as e:
            # live mode is intentionally unimplemented — fail gracefully with
            # a clear in-chat message rather than crashing the request.
            return {"proposal": None, "valid": False, "rejection_reason": str(e)}

    def route_after_propose(state: AgentState) -> str:
        return "validate" if state.get("proposal") is not None else "reject"

    def validate_node(state: AgentState) -> AgentState:
        proposal = state["proposal"]

        if proposal.get("metric") is None:
            # The model explicitly declined to guess rather than substitute
            # a real-but-unrelated metric (see llm_client's system prompt).
            # Filter its candidate_metrics against the REAL manifest before
            # ever showing them — never surface a suggestion we haven't
            # verified exists, even one the model claims is real.
            raw_candidates = proposal.get("candidate_metrics") or []
            verified_candidates = [c for c in raw_candidates if c in manifest.metric_names()]
            if verified_candidates:
                reason = (
                    "Your question doesn't clearly match any available metric. "
                    f"Did you mean one of: {verified_candidates}?"
                )
            else:
                reason = (
                    "Your question doesn't clearly match any available metric. "
                    f"Available metrics: {sorted(manifest.metric_names())}"
                )
            return {"valid": False, "rejection_reason": reason}

        result = validate_query(
            manifest,
            metric=proposal.get("metric"),
            dimensions=proposal.get("dimensions"),
            time_grain=proposal.get("time_grain"),
        )
        return {"valid": result.valid, "rejection_reason": result.reason}

    def route_after_validate(state: AgentState) -> str:
        return "check_cache" if state["valid"] else "reject"

    def check_cache_node(state: AgentState) -> AgentState:
        cached_rows = cache.get(state["proposal"])
        if cached_rows is not None:
            return {"rows": cached_rows, "cache_hit": True}
        return {"cache_hit": False}

    def route_after_cache(state: AgentState) -> str:
        return "summarize" if state.get("rows") is not None else "execute"

    def execute_node(state: AgentState) -> AgentState:
        try:
            rows = execute_query(
                state["proposal"],
                dbt_project_dir=dbt_project_dir,
                dbt_profiles_dir=dbt_profiles_dir,
                mf_binary=mf_binary,
                manifest=manifest,
            )
            cache.set(state["proposal"], rows)
            return {"rows": rows}
        except QueryExecutionError as e:
            # A guardrail pass doesn't guarantee MetricFlow can actually
            # execute the query (see query_executor's docstring — e.g. a
            # ratio metric grouped by a dimension with no single join
            # path). Treat this the same as a guardrail rejection: fail
            # closed, with a specific reason, not a crash.
            return {"valid": False, "rejection_reason": str(e)}

    def route_after_execute(state: AgentState) -> str:
        return "summarize" if state.get("rows") is not None else "reject"

    def reject_node(state: AgentState) -> AgentState:
        reason = state.get("rejection_reason") or "Query could not be validated."
        return {"summary": f"I can't answer that as asked: {reason}"}

    def summarize_node(state: AgentState) -> AgentState:
        summary = summarize_result(state["question"], state["rows"])
        return {"summary": summary}

    graph = StateGraph(AgentState)
    graph.add_node("propose", propose_node)
    graph.add_node("validate", validate_node)
    graph.add_node("check_cache", check_cache_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reject", reject_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "propose")
    graph.add_conditional_edges(
        "propose", route_after_propose, {"validate": "validate", "reject": "reject"}
    )
    graph.add_conditional_edges(
        "validate", route_after_validate, {"check_cache": "check_cache", "reject": "reject"}
    )
    graph.add_conditional_edges(
        "check_cache", route_after_cache, {"summarize": "summarize", "execute": "execute"}
    )
    graph.add_conditional_edges(
        "execute", route_after_execute, {"summarize": "summarize", "reject": "reject"}
    )
    graph.add_edge("summarize", END)
    graph.add_edge("reject", END)

    return graph.compile()


def ask(
    question: str,
    manifest_path: str,
    dbt_project_dir: str,
    dbt_profiles_dir: str,
    mf_binary: str = "mf",
) -> AgentState:
    """Convenience entry point: build the graph and run one question through it."""
    app = build_agent_graph(manifest_path, dbt_project_dir, dbt_profiles_dir, mf_binary)
    return app.invoke({"question": question})
