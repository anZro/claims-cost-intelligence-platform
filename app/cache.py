"""
Simple in-memory query cache, keyed by (metric, dimensions, filters,
time_grain, start_date, end_date).

Deliberately a plain module-level dict for the portfolio version — this
keeps the build simple without pretending it's production-scale. Redis
(or another shared cache) is the natural production upgrade path: this
process-local dict means cache state doesn't survive a restart and isn't
shared across multiple API worker processes, which would matter at real
scale but doesn't for a single-process demo/dev deployment.

`source` is deliberately excluded from the cache key — it records which
LLM mode produced the proposal, not what the query actually asks for. Two
proposals that differ only in `source` (e.g. a mock-mode and a
local_sim-mode run of the same underlying question) should hit the same
cache entry, since they'd execute identically against the semantic layer.
"""

from __future__ import annotations

import json
from typing import Any

_cache: dict[str, list[dict[str, Any]]] = {}
_stats = {"hits": 0, "misses": 0}


def _cache_key(proposal: dict[str, Any]) -> str:
    filters = proposal.get("filters") or []
    normalized_filters = sorted(
        filters, key=lambda f: (f.get("dimension", ""), f.get("operator", ""), str(f.get("value", "")))
    )
    relevant = {
        "metric": proposal.get("metric"),
        "dimensions": sorted(proposal.get("dimensions") or []),
        "filters": normalized_filters,
        "time_grain": proposal.get("time_grain"),
        "start_date": proposal.get("start_date"),
        "end_date": proposal.get("end_date"),
    }
    return json.dumps(relevant, sort_keys=True)


def get(proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
    key = _cache_key(proposal)
    if key in _cache:
        _stats["hits"] += 1
        return _cache[key]
    _stats["misses"] += 1
    return None


def set(proposal: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _cache[_cache_key(proposal)] = rows


def stats() -> dict[str, int]:
    return dict(_stats)


def clear() -> None:
    """Clears both cached entries and hit/miss stats. Mainly for tests."""
    _cache.clear()
    _stats["hits"] = 0
    _stats["misses"] = 0
