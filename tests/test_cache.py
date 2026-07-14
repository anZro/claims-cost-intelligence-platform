"""Tests for app.cache — isolated, no dependency on mf/dbt/duckdb at all."""

import pytest

from app import cache


@pytest.fixture(autouse=True)
def clear_cache_between_tests():
    cache.clear()
    yield
    cache.clear()


def test_miss_on_empty_cache():
    proposal = {"metric": "total_spend", "dimensions": [], "filters": [], "time_grain": "month"}
    assert cache.get(proposal) is None


def test_set_then_get_hits():
    proposal = {"metric": "total_spend", "dimensions": [], "filters": [], "time_grain": "month"}
    rows = [{"total_spend": "100"}]
    cache.set(proposal, rows)
    assert cache.get(proposal) == rows


def test_different_metric_is_a_different_key():
    p1 = {"metric": "total_spend", "dimensions": [], "filters": [], "time_grain": "month"}
    p2 = {"metric": "pmpm", "dimensions": [], "filters": [], "time_grain": "month"}
    cache.set(p1, [{"a": 1}])
    assert cache.get(p2) is None


def test_dimension_order_does_not_affect_key():
    p1 = {"metric": "total_spend", "dimensions": ["region", "submitted_date"], "filters": [], "time_grain": "month"}
    p2 = {"metric": "total_spend", "dimensions": ["submitted_date", "region"], "filters": [], "time_grain": "month"}
    cache.set(p1, [{"a": 1}])
    assert cache.get(p2) == [{"a": 1}]


def test_filter_order_does_not_affect_key():
    p1 = {
        "metric": "total_spend",
        "dimensions": ["region"],
        "filters": [
            {"dimension": "region", "operator": "equals", "value": "Region 3"},
            {"dimension": "drug_id", "operator": "equals", "value": "D1"},
        ],
        "time_grain": "month",
    }
    p2 = {
        "metric": "total_spend",
        "dimensions": ["region"],
        "filters": [
            {"dimension": "drug_id", "operator": "equals", "value": "D1"},
            {"dimension": "region", "operator": "equals", "value": "Region 3"},
        ],
        "time_grain": "month",
    }
    cache.set(p1, [{"a": 1}])
    assert cache.get(p2) == [{"a": 1}]


def test_source_field_does_not_affect_key():
    """
    mock-mode and local_sim-mode runs of 'the same' underlying question
    should hit the same cache entry — source records provenance, not
    query identity.
    """
    p1 = {"metric": "total_spend", "dimensions": [], "filters": [], "time_grain": "month", "source": "mock"}
    p2 = {"metric": "total_spend", "dimensions": [], "filters": [], "time_grain": "month", "source": "local_sim"}
    cache.set(p1, [{"a": 1}])
    assert cache.get(p2) == [{"a": 1}]


def test_different_start_end_date_is_a_different_key():
    p1 = {
        "metric": "total_spend",
        "dimensions": [],
        "filters": [],
        "time_grain": "quarter",
        "start_date": "2025-01-01",
        "end_date": "2025-03-31",
    }
    p2 = dict(p1, start_date="2025-04-01", end_date="2025-06-30")
    cache.set(p1, [{"a": 1}])
    assert cache.get(p2) is None


def test_stats_track_hits_and_misses():
    proposal = {"metric": "total_spend", "dimensions": [], "filters": [], "time_grain": "month"}
    cache.get(proposal)  # miss
    cache.set(proposal, [{"a": 1}])
    cache.get(proposal)  # hit
    cache.get(proposal)  # hit
    stats = cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 2


def test_clear_resets_entries_and_stats():
    proposal = {"metric": "total_spend", "dimensions": [], "filters": [], "time_grain": "month"}
    cache.set(proposal, [{"a": 1}])
    cache.get(proposal)
    cache.clear()
    assert cache.get(proposal) is None
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1  # the get() call right after clear() counts as a fresh miss
