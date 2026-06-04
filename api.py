"""
api.py — FastAPI server for the Research Assistant RAG Pipeline.

Endpoints:
  GET  /              → serves static/index.html
  GET  /health        → health check + model status
  POST /query         → run full RAG pipeline (JSON response)
  POST /query/stream  → run pipeline with SSE progress updates
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — warm ALL singletons before first request (Task 10)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup")
    try:
        from graph import get_graph, warm_models
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, warm_models)
        await loop.run_in_executor(None, get_graph)
        log.info("api.graph_warmed")
    except Exception as exc:
        log.warning("api.warmup_failed", error=str(exc))
    yield
    log.info("api.shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Assistant API",
    description=(
        "Production-grade RAG research assistant: Groq decomposer + evidence extraction + "
        "strict-grounding synthesiser, Qdrant dense retrieval, BM25 lexical retrieval, "
        "MiniLM reranker, Neo4j graph escalation, RAGAS-style evaluation, "
        "confidence scoring, sentence-level attribution checking."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "example": {"query": "What methods do LLM agents use for tool selection?"}
        }
    }


class QueryResponse(BaseModel):
    # Core
    query: str
    answer: str
    citations: list[str]           # valid verified citations only
    escalated: bool
    chunks_used: int
    sub_queries: list[dict]
    graph_context_items: int
    elapsed_seconds: float

    # Research quality metrics (Task 4 — ResearchAnswer fields)
    confidence: float
    confidence_band: str           # HIGH / MEDIUM / LOW
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    citation_coverage: float

    # Citation audit (Task 5)
    valid_citations: list[str]
    invalid_citations: list[str]

    # Attribution (Task 6)
    attribution: list[dict]

    # Performance (Task 11)
    timing: dict

    # Mode
    research_mode: bool
    synthesis_skipped: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static/")
    return FileResponse(str(index))


@app.get("/health")
async def health():
    """Health check — returns model availability and research mode status."""
    from utils.groq_key_manager import get_key_manager
    mgr = get_key_manager()
    return {
        "status": "ok",
        "timestamp": time.time(),
        "research_mode": os.getenv("RESEARCH_MODE", "true").lower() == "true",
        "groq_keys_available": mgr.available_count,
        "primary_synthesis_model": os.getenv("PRIMARY_SYNTHESIS_MODEL", "llama-3.3-70b-versatile"),
        "fallback_synthesis_model": os.getenv("FALLBACK_SYNTHESIS_MODEL", "deepseek-r1-distill-llama-70b"),
        "embed_model": os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "reranker_model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "eval_model": os.getenv("GROQ_EVAL_MODEL", "llama-3.1-8b-instant"),
        "retrieval": {
            "dense_top_k": 15,
            "lexical_top_k": 15,
            "rerank_top_k": 10,
            "mmr_final_k": 8,
        },
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Run the full research assistant pipeline and return a structured JSON response."""
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    log.info("api.query.start", query=req.query[:100])
    t0 = time.perf_counter()

    try:
        loop = asyncio.get_event_loop()
        from graph import run_query
        state = await loop.run_in_executor(None, run_query, req.query)
    except Exception as exc:
        log.error("api.query.failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    elapsed = time.perf_counter() - t0
    log.info("api.query.done", elapsed_s=round(elapsed, 2))

    evaluation = state.get("evaluation", {})

    return QueryResponse(
        # Core
        query=req.query,
        answer=state.get("answer", ""),
        citations=state.get("valid_citations", state.get("citations", [])),
        escalated=bool(state.get("escalated", False)),
        chunks_used=len(state.get("reranked_chunks", [])),
        sub_queries=state.get("sub_queries", []),
        graph_context_items=len(state.get("graph_context", [])),
        elapsed_seconds=round(elapsed, 2),
        # RAGAS metrics
        confidence=state.get("confidence", 0.0),
        confidence_band=state.get("confidence_band", "LOW"),
        faithfulness=evaluation.get("faithfulness", 0.0),
        answer_relevance=evaluation.get("answer_relevance", 0.0),
        context_precision=evaluation.get("context_precision", 0.0),
        context_recall=evaluation.get("context_recall", 0.0),
        citation_coverage=evaluation.get("citation_coverage", 0.0),
        # Citation audit
        valid_citations=state.get("valid_citations", []),
        invalid_citations=state.get("invalid_citations", []),
        # Attribution
        attribution=state.get("attribution", []),
        # Timing
        timing=state.get("timing", {}),
        # Mode
        research_mode=state.get("research_mode", True),
        synthesis_skipped=state.get("synthesis_skipped", False),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    SSE streaming endpoint — emits stage progress events while the
    pipeline runs in the background. Extended for research mode stages.
    """
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    async def event_stream() -> AsyncGenerator[str, None]:
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("status", {"stage": "starting", "message": "Initialising research pipeline…"})
        await asyncio.sleep(0)

        stages = [
            ("decomposer",          "Decomposing query into sub-queries…"),
            ("retrieval",           "Running dense (top-15) + lexical (top-15) retrieval…"),
            ("reranker",            "Re-ranking with MiniLM cross-encoder (top-30 → top-8)…"),
            ("evidence_extraction", "Extracting explicit claims from evidence…"),
            ("grader",              "Grading chunk quality…"),
            ("synthesis",           "Synthesising grounded answer with Groq…"),
            ("attribution_checker", "Checking sentence-level attribution…"),
            ("verifier",            "Verifying citations against retrieved evidence…"),
            ("ragas_evaluator",     "Computing RAGAS-style evaluation metrics…"),
            ("confidence_scorer",   "Scoring answer confidence…"),
        ]

        pipeline_task: asyncio.Task = asyncio.create_task(
            asyncio.get_event_loop().run_in_executor(None, _run_pipeline, req.query)
        )

        stage_delay = 4.0
        for stage, message in stages:
            if pipeline_task.done():
                break
            yield sse("status", {"stage": stage, "message": message})
            try:
                await asyncio.wait_for(asyncio.shield(pipeline_task), timeout=stage_delay)
                break
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

        try:
            state, elapsed = await pipeline_task
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
            return

        evaluation = state.get("evaluation", {})
        yield sse("status", {"stage": "done", "message": "Research complete!"})
        yield sse("result", {
            "query": req.query,
            "answer": state.get("answer", ""),
            "citations": state.get("valid_citations", state.get("citations", [])),
            "escalated": bool(state.get("escalated", False)),
            "chunks_used": len(state.get("reranked_chunks", [])),
            "sub_queries": state.get("sub_queries", []),
            "graph_context_items": len(state.get("graph_context", [])),
            "elapsed_seconds": round(elapsed, 2),
            # RAGAS metrics
            "confidence": state.get("confidence", 0.0),
            "confidence_band": state.get("confidence_band", "LOW"),
            "faithfulness": evaluation.get("faithfulness", 0.0),
            "answer_relevance": evaluation.get("answer_relevance", 0.0),
            "context_precision": evaluation.get("context_precision", 0.0),
            "context_recall": evaluation.get("context_recall", 0.0),
            "citation_coverage": evaluation.get("citation_coverage", 0.0),
            # Citation audit
            "valid_citations": state.get("valid_citations", []),
            "invalid_citations": state.get("invalid_citations", []),
            # Attribution
            "attribution": state.get("attribution", []),
            # Timing
            "timing": state.get("timing", {}),
            # Mode
            "research_mode": state.get("research_mode", True),
            "synthesis_skipped": state.get("synthesis_skipped", False),
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _run_pipeline(query: str):
    """Synchronous pipeline wrapper used by the executor."""
    from graph import run_query
    t0 = time.perf_counter()
    state = run_query(query)
    return state, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
