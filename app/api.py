"""
FastAPI wrapper around the LangGraph agent.

Thin by design — no new logic lives here. The agent graph is built ONCE at
startup (it loads and parses the semantic manifest, which doesn't change
between requests) and reused across requests, rather than rebuilt per call.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import config
from app.agent import build_agent_graph

_agent_app = None  # set at startup
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent_app
    _agent_app = build_agent_graph(
        manifest_path=config.SEMANTIC_MANIFEST_PATH,
        dbt_project_dir=config.CLAIMS_DBT_PROJECT_DIR,
        dbt_profiles_dir=config.DBT_PROFILES_DIR,
        mf_binary=config.MF_BINARY,
    )
    yield
    _agent_app = None


app = FastAPI(title="Claims-Cost Intelligence Platform", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    valid: bool
    proposal: dict[str, Any] | None = None
    rejection_reason: str | None = None
    rows: list[dict[str, Any]] | None = None
    cache_hit: bool | None = None
    summary: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/mode")
def mode_info() -> dict[str, str]:
    """
    Exposes the currently active LLM mode + model, so the UI can show
    which one is live rather than leaving it invisible. Useful both for
    demoing the mock/local_sim/live distinction and as a sanity check that
    the deployment is running the mode you think it is.
    """
    label = config.LLM_MODE.upper()
    if config.LLM_MODE == "local_sim":
        label += f" · {config.OLLAMA_MODEL or 'no model set'}"
    elif config.LLM_MODE == "live":
        label += f" · {config.ANTHROPIC_MODEL}"
    return {"llm_mode": config.LLM_MODE, "label": label}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    if _agent_app is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet.")

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty.")

    result = _agent_app.invoke({"question": request.question})

    return AskResponse(
        question=request.question,
        valid=bool(result.get("valid")),
        proposal=result.get("proposal"),
        rejection_reason=result.get("rejection_reason"),
        rows=result.get("rows"),
        cache_hit=result.get("cache_hit"),
        summary=result.get("summary"),
    )
