"""
Tests for app.api — the FastAPI wrapper around the agent.

Forces LLM_MODE=mock for determinism, same reasoning as test_agent.py.
Skips cleanly if mf/semantic_manifest.json aren't available, since the
agent graph built at app startup needs both.
"""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app import cache as cache_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
def configure_for_tests(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "mock")
    monkeypatch.setattr(config, "CLAIMS_DBT_PROJECT_DIR", str(PROJECT_ROOT / "claims_metrics"))
    monkeypatch.setattr(config, "DBT_PROFILES_DIR", str(Path.home() / ".dbt"))
    monkeypatch.setattr(config, "MF_BINARY", MF_BINARY)
    cache_module.clear()
    yield
    cache_module.clear()


@pytest.fixture
def client():
    from app.api import app

    with TestClient(app) as c:
        yield c


def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_serves_chat_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "GUARDRAIL: ARMED" in r.text


def test_mode_endpoint_reflects_mock_mode(client):
    r = client.get("/mode")
    assert r.status_code == 200
    assert r.json() == {"llm_mode": "mock", "label": "MOCK"}


def test_mode_endpoint_reflects_local_sim_with_model(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "local_sim")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "llama3.1:8b")
    r = client.get("/mode")
    assert r.json() == {"llm_mode": "local_sim", "label": "LOCAL_SIM · llama3.1:8b"}


def test_ask_valid_question_returns_rows_and_summary(client):
    r = client.post("/ask", json={"question": "What is total spend by region?"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["rows"] is not None
    assert len(body["rows"]) > 0
    assert body["summary"] is not None
    assert body["rejection_reason"] is None


def test_ask_fabricated_metric_returns_200_with_valid_false():
    """
    A rejection is NOT an HTTP error — it's a successful response describing
    a rejected query. The guardrail's job is to produce a clear answer about
    why the question can't be answered, not to fail the HTTP request.
    """
    from app.api import app

    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "What is the readmission rate?"})
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is False
        assert body["rows"] is None
        assert "does not exist" in body["rejection_reason"]
        assert "I can't answer that as asked" in body["summary"]


def test_ask_empty_question_returns_422(client):
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 422


def test_ask_missing_question_field_returns_422(client):
    r = client.post("/ask", json={})
    assert r.status_code == 422


def test_ask_response_always_includes_question(client):
    question = "What is the denial rate?"
    r = client.post("/ask", json={"question": question})
    assert r.json()["question"] == question


def test_second_identical_question_is_a_cache_hit(client):
    question = "What is total spend by region?"
    r1 = client.post("/ask", json={"question": question})
    assert r1.json()["cache_hit"] is False

    r2 = client.post("/ask", json={"question": question})
    assert r2.json()["cache_hit"] is True
    assert r2.json()["rows"] == r1.json()["rows"]
