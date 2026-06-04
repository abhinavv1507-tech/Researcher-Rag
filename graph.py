"""
graph.py — Production-Grade Research Assistant Pipeline (LangGraph Orchestration)

Architecture:
  decomposer → [dense_worker ‖ lexical_worker] → reranker
             → evidence_extraction → grader → (conditional) neo4j_escalation
             → synthesis → attribution_checker → verifier
             → ragas_evaluator → confidence_scorer → END

Models (all Groq — no Gemini dependency):
  Decomposer  : GROQ_DECOMPOSE_MODEL  (default: llama-3.3-70b-versatile)
  Grader      : GROQ_GRADE_MODEL      (default: llama-3.3-70b-versatile)
  Synthesis   : PRIMARY_SYNTHESIS_MODEL → FALLBACK_SYNTHESIS_MODEL
  Evaluation  : GROQ_EVAL_MODEL       (default: llama-3.1-8b-instant, fast)

Research Mode (RESEARCH_MODE=true):
  - Contextual Retrieval enabled (at index time)
  - Evidence Extraction enabled
  - Attribution Checker enabled
  - RAGAS Evaluation enabled
  - Confidence Scoring enabled
  - Strict Citation Verification enabled

Key manager : GroqKeyManager — 12-key round-robin with 429 cooldown
Singleton   : Embedding + reranker loaded once via utils.model_cache
Performance : Dense top-15, Lexical top-15, Rerank top-30, MMR top-8
Fail-safe   : Every node catches exceptions; pipeline never crashes
              If faithfulness < 0.50 → return retrieved evidence only
              If evidence is insufficient → explicit message, no fabrication
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from langgraph.graph import END, StateGraph  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
import pickle
from typing_extensions import TypedDict  # type: ignore

from utils.groq_key_manager import get_key_manager
from utils.logging_config import configure_logging, get_logger
from utils.model_cache import get_embedding_model, get_reranker_model
from utils.neo4j_client import Neo4jClient
from utils.reranker import rerank

load_dotenv()
configure_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from .env
# ---------------------------------------------------------------------------

GROQ_DECOMPOSE_MODEL: str = os.getenv("GROQ_DECOMPOSE_MODEL", "llama-3.3-70b-versatile")
GROQ_GRADE_MODEL: str = os.getenv("GROQ_GRADE_MODEL", "llama-3.3-70b-versatile")
GROQ_EVIDENCE_MODEL: str = os.getenv("GROQ_EVIDENCE_MODEL", "llama-3.1-8b-instant")
PRIMARY_SYNTHESIS_MODEL: str = os.getenv("PRIMARY_SYNTHESIS_MODEL", "llama-3.3-70b-versatile")
FALLBACK_SYNTHESIS_MODEL: str = os.getenv("FALLBACK_SYNTHESIS_MODEL", "deepseek-r1-distill-llama-70b")

QDRANT_COLLECTION: str = "research_chunks"
BM25_PKL: Path = Path("bm25_index.pkl")

# ── Retrieval Parameters (Task 9) ──────────────────────────────────────────
DENSE_TOP_K: int = 15       # top-15 dense results per sub-query
LEXICAL_TOP_K: int = 15     # top-15 BM25 results per sub-query
RERANK_TOP_K: int = 10      # keep top-10 after cross-encoder reranking
MMR_FINAL_K: int = 8        # final chunks after MMR diversity pass

# ── Research Mode ──────────────────────────────────────────────────────────
RESEARCH_MODE: bool = os.getenv("RESEARCH_MODE", "true").lower() == "true"

# ── Faithfulness Gates ─────────────────────────────────────────────────────
FAITHFULNESS_BLOCK_THRESHOLD: float = float(
    os.getenv("FAITHFULNESS_GATE_THRESHOLD", "0.50")
)
FAITHFULNESS_LOW_THRESHOLD: float = 0.70

# ── Insufficient Evidence Marker ───────────────────────────────────────────
INSUFFICIENT_EVIDENCE_MARKER = "__INSUFFICIENT_EVIDENCE__"
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "The retrieved evidence does not explicitly answer this question. "
    "Below are the most relevant passages found:\n\n"
)


# ---------------------------------------------------------------------------
# Agent State (extended for research mode)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    sub_queries: list[dict]            # [{query, tool}]
    dense_results: list[dict]
    lexical_results: list[dict]
    reranked_chunks: list[dict]
    graph_context: list[dict]
    evidence: list[dict]               # NEW: extracted evidence per chunk
    attribution: list[dict]            # NEW: sentence-level attribution
    answer: str
    citations: list[str]
    valid_citations: list[str]         # NEW: verified citations only
    invalid_citations: list[str]       # NEW: citations not in retrieved set
    evaluation: dict                   # NEW: RAGAS metric scores
    confidence: float                  # NEW: weighted confidence score
    confidence_band: str               # NEW: HIGH / MEDIUM / LOW
    # Separate timing fields to avoid concurrent writes in parallel nodes
    timing: dict                       # Final merged timing dict
    dense_timing: dict                 # Dense retrieval timing (parallel)
    lexical_timing: dict               # Lexical retrieval timing (parallel)
    reranker_timing: dict              # Reranker timing
    evidence_timing: dict              # Evidence extraction timing
    synthesis_timing: dict             # Synthesis timing
    attribution_timing: dict           # Attribution checking timing
    verification_timing: dict          # Verification timing
    evaluation_timing: dict            # Evaluation timing
    escalated: bool
    research_mode: bool                # NEW: whether research mode is active
    synthesis_skipped: bool            # NEW: True if faithfulness gate blocked synthesis


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured LLM Output
# ---------------------------------------------------------------------------

class SubQuery(BaseModel):
    query: str = Field(description="A focused sub-question derived from the main query")
    tool: Literal["dense_search", "lexical_search"] = Field(
        description="Retrieval tool: dense_search for semantic, lexical_search for keyword"
    )


class SubQueryList(BaseModel):
    sub_queries: list[SubQuery] = Field(
        description="2-4 sub-queries covering different aspects of the main question"
    )


class GraderOutput(BaseModel):
    sufficient: bool = Field(
        description="True if retrieved chunks sufficiently answer the query"
    )
    reasoning: str = Field(description="Brief one-sentence justification")


class EvidenceItem(BaseModel):
    """Explicit claims extracted from a single chunk — no inference allowed."""
    paper_id: str = Field(description="arXiv ID of the source paper")
    source: str = Field(description="Paper title or short identifier")
    explicit_claims: list[str] = Field(
        description=(
            "List of explicit factual claims from this chunk. "
            "No inference, no summarization, no speculation — only explicit statements."
        )
    )


class EvidenceBatch(BaseModel):
    items: list[EvidenceItem] = Field(
        description="Evidence extracted from each retrieved chunk"
    )


class SynthesisOutput(BaseModel):
    """
    Strict output schema for research-grade synthesis.
    No thought_process / reasoning fields.
    """
    answer: str = Field(
        description=(
            "Research answer synthesized ONLY from extracted evidence. "
            "Every factual sentence must have an inline citation [arXiv:XXXX.XXXXX]. "
            "If evidence is insufficient, return exactly: '__INSUFFICIENT_EVIDENCE__'"
        )
    )
    citations: list[str] = Field(
        description="List of arXiv IDs cited (e.g. ['2401.12345', '2312.00001'])"
    )


class ResearchAnswer(BaseModel):
    """
    Final structured answer with all RAGAS metrics and confidence scoring.
    """
    answer: str
    citations: list[str]
    confidence: float
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    citation_coverage: float


# ---------------------------------------------------------------------------
# Infrastructure Singletons (clients, not models)
# ---------------------------------------------------------------------------

_qdrant: Optional[QdrantClient] = None
_bm25_data: Optional[dict] = None
_neo4j: Optional[Neo4jClient] = None
_chunks_lookup: Optional[dict] = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
        log.debug("qdrant.client.created")
    return _qdrant


def _get_bm25() -> dict:
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_PKL, "rb") as f:
            _bm25_data = pickle.load(f)
        log.debug("bm25.index.loaded", chunk_count=len(_bm25_data.get("chunk_ids", [])))
    return _bm25_data


def _get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
        log.debug("neo4j.client.created")
    return _neo4j


def _get_chunks_lookup() -> dict:
    """Cache the chunks.json lookup dict for BM25 payload enrichment."""
    global _chunks_lookup
    if _chunks_lookup is None:
        p = Path("chunks.json")
        if p.exists():
            with open(p, encoding="utf-8") as f:
                chunks = json.load(f)
            _chunks_lookup = {c["chunk_id"]: c for c in chunks}
            log.debug("chunks_lookup.loaded", count=len(_chunks_lookup))
        else:
            _chunks_lookup = {}
            log.warning("chunks_lookup.missing", path=str(p))
    return _chunks_lookup


# ---------------------------------------------------------------------------
# Groq LLM factory — integrates with GroqKeyManager
# ---------------------------------------------------------------------------

def _groq_llm(model: str, temperature: float = 0.0) -> ChatGroq:
    key = get_key_manager().next_key()
    return ChatGroq(model=model, temperature=temperature, api_key=key)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def _handle_groq_error(exc: Exception, key: str) -> None:
    mgr = get_key_manager()
    if _is_rate_limit_error(exc):
        mgr.mark_rate_limited(key)
        log.warning("groq.rate_limited", key_suffix=key[-6:], error=str(exc)[:120])
    elif "401" in str(exc) or "invalid_api_key" in str(exc).lower():
        mgr.mark_failed(key)
        log.error("groq.invalid_key", key_suffix=key[-6:])
    else:
        log.warning("groq.error", key_suffix=key[-6:], error=str(exc)[:200])


# ---------------------------------------------------------------------------
# Startup warm-up (Task 10)
# ---------------------------------------------------------------------------

def warm_models() -> None:
    """
    Pre-load ALL singletons on startup — eliminates first-query cold start.
    Warms: embedding model, reranker model, Qdrant client, BM25 index, Neo4j.
    """
    log.info("warmup.started")
    t0 = time.perf_counter()

    get_embedding_model()
    log.info("warmup.embedding.loaded")

    backend, _ = get_reranker_model()
    log.info("warmup.reranker.loaded", backend=backend)

    try:
        _get_qdrant()
        log.info("warmup.qdrant.connected")
    except Exception as exc:
        log.warning("warmup.qdrant.failed", error=str(exc)[:100])

    try:
        _get_bm25()
        log.info("warmup.bm25.loaded")
    except Exception as exc:
        log.warning("warmup.bm25.failed", error=str(exc)[:100])

    try:
        _get_chunks_lookup()
        log.info("warmup.chunks_lookup.loaded")
    except Exception as exc:
        log.warning("warmup.chunks_lookup.failed", error=str(exc)[:100])

    try:
        _get_neo4j()
        log.info("warmup.neo4j.connected")
    except Exception as exc:
        log.warning("warmup.neo4j.failed", error=str(exc)[:100])

    mgr = get_key_manager()
    log.info(
        "warmup.groq.configured",
        primary=PRIMARY_SYNTHESIS_MODEL,
        fallback=FALLBACK_SYNTHESIS_MODEL,
        groq_keys_available=mgr.available_count,
        research_mode=RESEARCH_MODE,
    )

    elapsed = time.perf_counter() - t0
    log.info("warmup.completed", warmup_time_s=round(elapsed, 2))


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _update_timing(state: AgentState, key: str, elapsed: float) -> dict:
    """Return a timing dict update merged with existing timing."""
    timing = dict(state.get("timing", {}))
    timing[key] = round(elapsed, 3)
    return timing


# ---------------------------------------------------------------------------
# Node: decomposer
# ---------------------------------------------------------------------------

def _decompose_with_retry(query: str, max_attempts: int = 4) -> list[dict]:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        key = get_key_manager().next_key()
        try:
            llm = ChatGroq(
                model=GROQ_DECOMPOSE_MODEL,
                temperature=0.0,
                api_key=key,
            ).with_structured_output(SubQueryList)
            messages = [
                SystemMessage(
                    content=(
                        "You are a research query analyst. Break the user's research question "
                        "into 2–4 focused sub-queries. Assign 'dense_search' to conceptual or "
                        "semantic sub-queries, and 'lexical_search' to sub-queries best served "
                        "by exact keyword matching (method names, metric names, author names). "
                        "Return ONLY a JSON object with a 'sub_queries' array."
                    )
                ),
                HumanMessage(content=f"Research question: {query}"),
            ]
            result: SubQueryList = llm.invoke(messages)
            return [sq.dict() for sq in result.sub_queries]
        except Exception as exc:
            _handle_groq_error(exc, key)
            last_exc = exc
            log.warning("decomposer.retry", attempt=attempt + 1, max=max_attempts, error=str(exc)[:120])

    log.error("decomposer.all_attempts_failed", error=str(last_exc))
    return [{"query": query, "tool": "dense_search"}]


def decomposer_node(state: AgentState) -> dict:
    log.info("node.decomposer", query=state["query"][:100])
    t0 = time.perf_counter()
    sub_queries = _decompose_with_retry(state["query"])
    elapsed = time.perf_counter() - t0
    log.info("node.decomposer.done", n_sub_queries=len(sub_queries), elapsed_s=round(elapsed, 3))
    timing = _update_timing(state, "decomposer_time", elapsed)
    return {
        "sub_queries": sub_queries,
        "research_mode": RESEARCH_MODE,
        "timing": timing,
    }


# ---------------------------------------------------------------------------
# Node: dense_worker
# ---------------------------------------------------------------------------

def _qdrant_search(query_text: str, top_k: int = DENSE_TOP_K) -> list[dict]:
    embedder = get_embedding_model()
    vec = embedder.encode(query_text, normalize_embeddings=True).tolist()
    t0 = time.perf_counter()
    hits = _get_qdrant().query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        limit=top_k,
        with_payload=True,
    ).points
    elapsed = time.perf_counter() - t0
    results: list[dict] = []
    for hit in hits:
        r = dict(hit.payload)
        r["score"] = hit.score
        r["retrieval_source"] = "dense"
        results.append(r)
    log.debug("qdrant_search.done", results=len(results), elapsed_s=round(elapsed, 3))
    return results


def dense_worker_node(state: AgentState) -> dict:
    log.info("retrieval.started", source="dense", top_k=DENSE_TOP_K)
    t0 = time.perf_counter()

    queries = [sq["query"] for sq in state["sub_queries"] if sq.get("tool") == "dense_search"]
    if not queries:
        queries = [state["query"]]

    all_results: list[dict] = []
    for q in queries:
        try:
            all_results.extend(_qdrant_search(q, top_k=DENSE_TOP_K))
        except Exception as exc:
            log.error("dense_worker.search_failed", query=q[:80], error=str(exc))

    seen: dict[Any, dict] = {}
    for r in all_results:
        cid = r.get("chunk_id")
        if cid not in seen or r["score"] > seen[cid]["score"]:
            seen[cid] = r
    deduped = list(seen.values())

    elapsed = time.perf_counter() - t0
    log.info("retrieval.completed", source="dense", results=len(deduped), elapsed_s=round(elapsed, 3))
    dense_timing = {"embedding_time": round(elapsed, 3), "qdrant_time": round(elapsed, 3)}
    return {"dense_results": deduped, "dense_timing": dense_timing}


# ---------------------------------------------------------------------------
# Node: lexical_worker
# ---------------------------------------------------------------------------

def _bm25_search(query_text: str, top_k: int = LEXICAL_TOP_K) -> list[dict]:
    data = _get_bm25()
    index = data["index"]
    chunk_ids = data["chunk_ids"]
    lookup = _get_chunks_lookup()

    tokens = query_text.lower().split()
    scores = index.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results: list[dict] = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        r: dict[str, Any] = {
            "chunk_id": chunk_ids[idx],
            "score": float(scores[idx]),
            "retrieval_source": "lexical",
        }
        if chunk_ids[idx] in lookup:
            r.update(lookup[chunk_ids[idx]])
        results.append(r)
    return results


def lexical_worker_node(state: AgentState) -> dict:
    log.info("retrieval.started", source="lexical", top_k=LEXICAL_TOP_K)
    t0 = time.perf_counter()

    queries = [sq["query"] for sq in state["sub_queries"] if sq.get("tool") == "lexical_search"]
    if not queries:
        queries = [state["query"]]

    all_results: list[dict] = []
    for q in queries:
        try:
            all_results.extend(_bm25_search(q, top_k=LEXICAL_TOP_K))
        except Exception as exc:
            log.error("lexical_worker.search_failed", query=q[:80], error=str(exc))

    seen: dict[Any, dict] = {}
    for r in all_results:
        cid = r.get("chunk_id")
        if cid not in seen or r["score"] > seen[cid]["score"]:
            seen[cid] = r
    deduped = list(seen.values())

    elapsed = time.perf_counter() - t0
    log.info("retrieval.completed", source="lexical", results=len(deduped), elapsed_s=round(elapsed, 3))
    lexical_timing = {"bm25_time": round(elapsed, 3)}
    return {"lexical_results": deduped, "lexical_timing": lexical_timing}


# ---------------------------------------------------------------------------
# Node: reranker
# ---------------------------------------------------------------------------

def reranker_node(state: AgentState) -> dict:
    log.info("node.reranker", input_chunks=len(state["dense_results"]) + len(state["lexical_results"]))
    pooled = state["dense_results"] + state["lexical_results"]
    t0 = time.perf_counter()
    reranked = rerank(state["query"], pooled, final_k=MMR_FINAL_K)
    elapsed = time.perf_counter() - t0
    log.info("reranker.completed", final=len(reranked), elapsed_s=round(elapsed, 3))
    reranker_timing = {"reranker_time": round(elapsed, 3)}
    return {"reranked_chunks": reranked, "reranker_timing": reranker_timing}


# ---------------------------------------------------------------------------
# Node: evidence_extraction (Task 2)
# ---------------------------------------------------------------------------

_EVIDENCE_SYSTEM_PROMPT = (
    "You are an evidence extractor for academic research papers.\n"
    "For each provided chunk, extract ONLY explicit factual claims.\n\n"
    "Rules:\n"
    "1. Extract only EXPLICIT statements — no inference.\n"
    "2. No summarization — use the paper's own language where possible.\n"
    "3. No speculation or interpretation.\n"
    "4. No reasoning about what the paper 'likely' means.\n"
    "5. Each claim must be a complete, standalone sentence.\n"
    "6. If a chunk contains no extractable claims, return an empty list.\n\n"
    "Return JSON with 'items': list of {paper_id, source, explicit_claims}."
)


def _extract_evidence_with_retry(
    query: str,
    chunks: list[dict],
    max_attempts: int = 3,
) -> list[dict]:
    """Extract explicit claims from chunks using Groq with key rotation."""
    if not chunks:
        return []

    # Build prompt content
    chunk_blocks = []
    for c in chunks[:8]:  # cap to avoid huge prompts
        aid = c.get("arxiv_id", c.get("paper_id", "unknown"))
        title = c.get("title", "")
        text = c.get("full_contextualized_text", "")[:500]
        chunk_blocks.append(f"[{aid}] Title: {title}\nContent: {text}")

    chunks_text = "\n\n---\n\n".join(chunk_blocks)

    messages = [
        SystemMessage(content=_EVIDENCE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research query: {query}\n\n"
                f"Retrieved chunks:\n\n{chunks_text}"
            )
        ),
    ]

    for attempt in range(max_attempts):
        key = get_key_manager().next_key()
        try:
            llm = ChatGroq(
                model=GROQ_EVIDENCE_MODEL,
                temperature=0.0,
                api_key=key,
            ).with_structured_output(EvidenceBatch)
            result: EvidenceBatch = llm.invoke(messages)
            return [item.dict() for item in result.items]
        except Exception as exc:
            _handle_groq_error(exc, key)
            log.warning("evidence_extraction.retry", attempt=attempt + 1, error=str(exc)[:100])

    # Fallback: build minimal evidence from chunk metadata
    log.warning("evidence_extraction.fallback — using chunk metadata")
    fallback = []
    for c in chunks:
        aid = c.get("arxiv_id", c.get("paper_id", "unknown"))
        fallback.append({
            "paper_id": aid,
            "source": c.get("title", ""),
            "explicit_claims": [],
        })
    return fallback


def evidence_extraction_node(state: AgentState) -> dict:
    if not state.get("research_mode", RESEARCH_MODE):
        return {"evidence": [], "evidence_timing": {}}

    log.info("node.evidence_extraction")
    t0 = time.perf_counter()
    evidence = _extract_evidence_with_retry(state["query"], state["reranked_chunks"])
    elapsed = time.perf_counter() - t0

    total_claims = sum(len(e.get("explicit_claims", [])) for e in evidence)
    log.info(
        "node.evidence_extraction.done",
        n_chunks=len(state["reranked_chunks"]),
        n_evidence_items=len(evidence),
        total_claims=total_claims,
        elapsed_s=round(elapsed, 3),
    )
    evidence_timing = {"evidence_extraction_time": round(elapsed, 3)}
    return {"evidence": evidence, "evidence_timing": evidence_timing}


# ---------------------------------------------------------------------------
# Node: grader
# ---------------------------------------------------------------------------

def _grade_with_retry(query: str, chunks: list[dict], max_attempts: int = 4) -> GraderOutput:
    chunk_summary = "\n\n".join(
        f"[{c.get('arxiv_id', '?')}] {c.get('full_contextualized_text', '')[:400]}"
        for c in chunks[:10]
    )
    messages = [
        SystemMessage(
            content=(
                "You are a research quality assessor. Given a query and retrieved evidence, "
                "determine if the chunks sufficiently answer the query. "
                "Return JSON with 'sufficient' (bool) and 'reasoning' (string)."
            )
        ),
        HumanMessage(
            content=f"Query: {query}\n\nEvidence:\n{chunk_summary}\n\nAre these chunks sufficient?"
        ),
    ]

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        key = get_key_manager().next_key()
        try:
            llm = ChatGroq(
                model=GROQ_GRADE_MODEL,
                temperature=0.0,
                api_key=key,
            ).with_structured_output(GraderOutput)
            return llm.invoke(messages)
        except Exception as exc:
            _handle_groq_error(exc, key)
            last_exc = exc
            log.warning("grader.retry", attempt=attempt + 1, error=str(exc)[:120])

    log.error("grader.all_attempts_failed", error=str(last_exc))
    return GraderOutput(sufficient=True, reasoning="Grader unavailable — defaulting to sufficient")


def grader_node(state: AgentState) -> dict:
    log.info("node.grader")
    if not state["reranked_chunks"]:
        log.warning("grader.no_chunks", escalating=True)
        return {"escalated": True, "evidence_timing": {}}

    t0 = time.perf_counter()
    assessment = _grade_with_retry(state["query"], state["reranked_chunks"])
    elapsed = time.perf_counter() - t0
    escalated = not assessment.sufficient

    log.info(
        "grader.completed",
        sufficient=assessment.sufficient,
        escalated=escalated,
        elapsed_s=round(elapsed, 3),
    )
    evidence_timing = {"grader_time": round(elapsed, 3)}
    return {"escalated": escalated, "evidence_timing": evidence_timing}


def should_escalate(state: AgentState) -> str:
    return "neo4j_escalation" if state["escalated"] else "synthesis"


# ---------------------------------------------------------------------------
# Node: neo4j_escalation
# ---------------------------------------------------------------------------

def neo4j_escalation_node(state: AgentState) -> dict:
    log.info("node.neo4j_escalation")
    arxiv_ids = list({
        c.get("arxiv_id")
        for c in state["reranked_chunks"]
        if c.get("arxiv_id")
    })
    if not arxiv_ids:
        return {"graph_context": [], "evidence_timing": {}}

    t0 = time.perf_counter()
    try:
        graph_context = _get_neo4j().multi_hop_query(arxiv_ids)
    except Exception as exc:
        log.error("neo4j.failed", error=str(exc))
        graph_context = []

    elapsed = time.perf_counter() - t0
    log.info("neo4j.completed", graph_items=len(graph_context), elapsed_s=round(elapsed, 3))
    evidence_timing = {"neo4j_time": round(elapsed, 3)}
    return {"graph_context": graph_context, "evidence_timing": evidence_timing}


# ---------------------------------------------------------------------------
# Node: synthesis — research-grade strict grounding prompt (Task 3)
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a research-grade synthesis engine. Your role is to produce factual, \
grounded answers based ONLY on the provided retrieved evidence.

STRICT RULES — violating any rule is a critical failure:
1. Use ONLY facts explicitly stated in the retrieved evidence.
2. Every factual sentence MUST end with an inline citation: [arXiv:XXXX.XXXXX]
3. Do NOT infer methods, techniques, or conclusions not mentioned in the evidence.
4. Do NOT generalize across papers unless the evidence explicitly makes that connection.
5. Do NOT merge information from different papers unless the evidence explicitly links them.
6. Do NOT fabricate citations. Only cite paper IDs from the provided evidence.
7. Do NOT invent evaluation metrics, benchmark scores, or dataset names.
8. If the evidence is insufficient to answer the question, return exactly the string: \
__INSUFFICIENT_EVIDENCE__
9. Return ONLY valid JSON. No markdown, no code blocks, no preamble.
10. Schema: {"answer": "<your grounded answer>", "citations": ["XXXX.XXXXX", ...]}
"""


def _build_synthesis_prompt(
    query: str,
    chunks: list[dict],
    graph_context: list[dict],
    evidence: list[dict],
) -> list:
    # Use extracted evidence claims if available
    evidence_section = ""
    if evidence:
        ev_lines = []
        for ev in evidence:
            pid = ev.get("paper_id", "?")
            claims = ev.get("explicit_claims", [])
            if claims:
                ev_lines.append(f"[{pid}] Extracted claims:")
                ev_lines.extend(f"  • {claim}" for claim in claims)
        if ev_lines:
            evidence_section = "\n\nExtracted explicit claims:\n" + "\n".join(ev_lines)

    chunk_text = "\n\n".join(
        f"[arXiv:{c.get('arxiv_id', '?')}] Title: {c.get('title', '')}\n"
        f"{c.get('full_contextualized_text', '')[:700]}"
        for c in chunks
    )
    graph_text = ""
    if graph_context:
        graph_text = "\n\nGraph context (multi-hop related papers):\n" + "\n".join(
            f"- {g.get('relation', 'related')}: [{g.get('arxiv_id', '?')}] {g.get('title', '')}"
            for g in graph_context
        )
    return [
        SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research question: {query}\n\n"
                f"Retrieved evidence:\n{chunk_text}"
                f"{evidence_section}"
                f"{graph_text}"
            )
        ),
    ]


def _context_summary_fallback(chunks: list[dict], query: str) -> SynthesisOutput:
    """Fail-safe: build answer from raw chunks if all synthesis models fail."""
    log.warning("synthesis.using_context_fallback")
    citations: list[str] = []
    lines: list[str] = [
        f"⚠️ Synthesis models unavailable. Retrieved evidence for: {query}\n"
    ]
    for c in chunks[:5]:
        aid = c.get("arxiv_id", "")
        text = c.get("full_contextualized_text", "")[:300]
        if aid:
            lines.append(f"• [arXiv:{aid}] {text}")
            citations.append(aid)
        else:
            lines.append(f"• {text}")
    return SynthesisOutput(answer="\n".join(lines), citations=citations)


def _synthesize(
    query: str,
    chunks: list[dict],
    graph_context: list[dict],
    evidence: list[dict],
) -> SynthesisOutput:
    models = list(dict.fromkeys([PRIMARY_SYNTHESIS_MODEL, FALLBACK_SYNTHESIS_MODEL]))
    messages = _build_synthesis_prompt(query, chunks, graph_context, evidence)

    for model_name in models:
        for attempt in range(3):
            key = get_key_manager().next_key()
            try:
                t0 = time.perf_counter()
                llm = ChatGroq(
                    model=model_name,
                    temperature=0.1,
                    api_key=key,
                ).with_structured_output(SynthesisOutput)
                result: SynthesisOutput = llm.invoke(messages)
                elapsed = time.perf_counter() - t0
                log.info(
                    "synthesis.completed",
                    model=model_name,
                    citations=len(result.citations),
                    synthesis_time_s=round(elapsed, 3),
                )
                return result
            except Exception as exc:
                _handle_groq_error(exc, key)
                if model_name == models[0] and attempt == 2:
                    log.warning(
                        "fallback.used",
                        failed_model=model_name,
                        fallback_model=models[1] if len(models) > 1 else "context_summary",
                    )
                log.warning("synthesis.attempt_failed", model=model_name, attempt=attempt + 1, error=str(exc)[:200])

    log.error("synthesis.all_models_failed", tried=models)
    return _context_summary_fallback(chunks, query)


def synthesis_node(state: AgentState) -> dict:
    log.info("synthesis.started", primary=PRIMARY_SYNTHESIS_MODEL)
    t0 = time.perf_counter()

    output = _synthesize(
        state["query"],
        state["reranked_chunks"],
        state.get("graph_context", []),
        state.get("evidence", []),
    )
    elapsed = time.perf_counter() - t0

    # Handle insufficient evidence marker
    synthesis_skipped = False
    if output.answer.strip() == INSUFFICIENT_EVIDENCE_MARKER:
        log.warning("synthesis.insufficient_evidence — building fallback evidence answer")
        synthesis_skipped = True
        evidence_lines = []
        for c in state["reranked_chunks"][:5]:
            aid = c.get("arxiv_id", "")
            text = c.get("full_contextualized_text", "")[:250]
            if aid:
                evidence_lines.append(f"• [arXiv:{aid}] {text}")
        output = SynthesisOutput(
            answer=INSUFFICIENT_EVIDENCE_RESPONSE + "\n".join(evidence_lines),
            citations=[
                c.get("arxiv_id", "") for c in state["reranked_chunks"][:5]
                if c.get("arxiv_id")
            ],
        )

    log.info("node.synthesis.done", citations=len(output.citations), elapsed_s=round(elapsed, 3))
    synthesis_timing = {"synthesis_time": round(elapsed, 3)}
    return {
        "answer": output.answer,
        "citations": output.citations,
        "synthesis_skipped": synthesis_skipped,
        "synthesis_timing": synthesis_timing,
    }


# ---------------------------------------------------------------------------
# Node: attribution_checker (Task 6)
# ---------------------------------------------------------------------------

def _sentence_is_supported(
    sentence: str,
    chunks: list[dict],
    min_overlap_tokens: int = 3,
) -> tuple[bool, list[str]]:
    """
    Check if a sentence is supported by at least one retrieved chunk.
    Returns (supported: bool, supporting_arxiv_ids: list[str]).
    """
    STOPWORDS = {
        "the", "a", "an", "is", "in", "of", "to", "and", "for",
        "on", "at", "by", "with", "as", "it", "its", "are", "was",
        "be", "do", "have", "from", "this", "that",
    }
    # Remove citation markers from sentence for comparison
    clean_sent = re.sub(r"\[arXiv:[\d.]+\]", "", sentence).lower()
    sent_tokens = {
        t for t in re.findall(r"\b[a-z]{4,}\b", clean_sent)
        if t not in STOPWORDS
    }

    if len(sent_tokens) < 3:
        # Very short / trivial sentence — count as supported
        return True, []

    supporting: list[str] = []
    for chunk in chunks:
        chunk_text = chunk.get("full_contextualized_text", "").lower()
        chunk_tokens = set(re.findall(r"\b[a-z]{4,}\b", chunk_text))
        overlap = sent_tokens & chunk_tokens
        if len(overlap) >= min_overlap_tokens:
            aid = chunk.get("arxiv_id", chunk.get("paper_id", ""))
            if aid and aid not in supporting:
                supporting.append(aid)

    return len(supporting) > 0, supporting


def attribution_checker_node(state: AgentState) -> dict:
    if not state.get("research_mode", RESEARCH_MODE):
        return {"attribution": [], "attribution_timing": {}}

    log.info("node.attribution_checker")
    t0 = time.perf_counter()

    answer = state.get("answer", "")
    chunks = state.get("reranked_chunks", [])

    if not answer.strip() or INSUFFICIENT_EVIDENCE_RESPONSE in answer:
        attribution_timing = {"attribution_time": 0.0}
        return {"attribution": [], "attribution_timing": attribution_timing}

    # Split answer into sentences
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", answer)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    attribution: list[dict] = []
    supported_sentences: list[str] = []

    for sentence in sentences:
        supported, sources = _sentence_is_supported(sentence, chunks)
        attribution.append({
            "sentence": sentence,
            "supported": supported,
            "supporting_sources": sources,
        })
        if supported:
            supported_sentences.append(sentence)

    # Rebuild answer with only supported sentences
    supported_count = sum(1 for a in attribution if a["supported"])
    total_count = len(attribution)

    if supported_sentences and supported_count < total_count:
        # Reconstruct answer with only supported sentences
        cleaned_answer = " ".join(supported_sentences)
        log.info(
            "attribution_checker.pruned",
            removed=total_count - supported_count,
            kept=supported_count,
        )
    else:
        cleaned_answer = answer

    elapsed = time.perf_counter() - t0
    log.info(
        "node.attribution_checker.done",
        total_sentences=total_count,
        supported=supported_count,
        elapsed_s=round(elapsed, 3),
    )
    attribution_timing = {"attribution_time": round(elapsed, 3)}
    return {
        "answer": cleaned_answer,
        "attribution": attribution,
        "attribution_timing": attribution_timing,
    }


# ---------------------------------------------------------------------------
# Node: verifier (Task 5 — rewritten, fixes valid_citations=0 bug)
# ---------------------------------------------------------------------------

_ARXIV_RE = re.compile(r"\[arXiv:([\d]{4}\.[\d]{4,5})\]")


def verifier_node(state: AgentState) -> dict:
    """
    Strict citation verifier.

    Algorithm:
    1. Extract all [arXiv:XXXX.XXXXX] references from the answer text.
    2. Build allowed_ids = {chunk["arxiv_id"]} from retrieved chunks.
    3. A citation is VALID if its arXiv ID is in allowed_ids.
    4. Remove sentences containing ONLY invalid citations.
    5. Log all invalid citations.

    This fixes the previous SequenceMatcher(ratio=0.35) approach that
    incorrectly stripped all valid citations.
    """
    log.info("node.verifier")
    t0 = time.perf_counter()

    chunks = state["reranked_chunks"]
    answer = state.get("answer", "")

    # Build set of allowed paper IDs from retrieved chunks
    allowed_ids: set[str] = {
        c.get("arxiv_id", c.get("paper_id", ""))
        for c in chunks
        if c.get("arxiv_id") or c.get("paper_id")
    }
    allowed_ids.discard("")

    log.debug("verifier.allowed_ids", count=len(allowed_ids), ids=list(allowed_ids)[:5])

    # Also check citations list from synthesis output
    synthesis_citations = state.get("citations", [])

    valid_citations: list[str] = []
    invalid_citations: list[str] = []

    # Process line by line to keep/remove citations
    lines = answer.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        found_ids = _ARXIV_RE.findall(line)
        if not found_ids:
            cleaned_lines.append(line)
            continue

        line_valid = True
        for arxiv_id in found_ids:
            if arxiv_id in allowed_ids:
                if arxiv_id not in valid_citations:
                    valid_citations.append(arxiv_id)
            else:
                if arxiv_id not in invalid_citations:
                    invalid_citations.append(arxiv_id)
                    log.info(
                        "verifier.invalid_citation",
                        arxiv_id=arxiv_id,
                        snippet=line[:80],
                    )
                # Keep the line but note the invalid citation
                # (don't remove — it might have valid citations too)

        # If ALL citations on this line are invalid, strip the line
        if found_ids and all(cid not in allowed_ids for cid in found_ids):
            log.info("verifier.stripped_line", found=found_ids, snippet=line[:80])
            line_valid = False

        if line_valid:
            cleaned_lines.append(line)

    # Also validate synthesis citations list
    for cid in synthesis_citations:
        if cid in allowed_ids:
            if cid not in valid_citations:
                valid_citations.append(cid)
        else:
            if cid not in invalid_citations:
                invalid_citations.append(cid)

    cleaned_answer = "\n".join(cleaned_lines)

    elapsed = time.perf_counter() - t0
    log.info(
        "node.verifier.done",
        valid_citations=len(valid_citations),
        invalid_citations=len(invalid_citations),
        elapsed_s=round(elapsed, 3),
    )
    verification_timing = {"verification_time": round(elapsed, 3)}
    return {
        "answer": cleaned_answer,
        "citations": valid_citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "verification_timing": verification_timing,
    }


# ---------------------------------------------------------------------------
# Node: ragas_evaluator (Task 7)
# ---------------------------------------------------------------------------

def ragas_evaluator_node(state: AgentState) -> dict:
    if not state.get("research_mode", RESEARCH_MODE):
        default_eval = {
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "citation_coverage": 1.0,
        }
        return {"evaluation": default_eval, "evaluation_timing": {}}

    log.info("node.ragas_evaluator")
    t0 = time.perf_counter()

    from evaluation.ragas_evaluator import RAGASEvaluator
    evaluator = RAGASEvaluator()

    try:
        metrics = evaluator.evaluate(
            query=state["query"],
            answer=state.get("answer", ""),
            chunks=state.get("reranked_chunks", []),
            valid_citations=state.get("valid_citations", []),
            evidence=state.get("evidence", []),
        )
    except Exception as exc:
        log.error("ragas_evaluator.failed", error=str(exc))
        metrics = {
            "faithfulness": 0.5,
            "answer_relevance": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "citation_coverage": 0.0,
        }

    elapsed = time.perf_counter() - t0
    log.info("node.ragas_evaluator.done", elapsed_s=round(elapsed, 3), **{k: round(v, 3) for k, v in metrics.items()})
    evaluation_timing = {"evaluation_time": round(elapsed, 3)}
    return {"evaluation": metrics, "evaluation_timing": evaluation_timing}


# ---------------------------------------------------------------------------
# Node: confidence_scorer (Task 8)
# ---------------------------------------------------------------------------

def confidence_scorer_node(state: AgentState) -> dict:
    if not state.get("research_mode", RESEARCH_MODE):
        return {"confidence": 1.0, "confidence_band": "HIGH", "evaluation_timing": {}}

    log.info("node.confidence_scorer")

    from evaluation.scoring import ConfidenceScorer
    scorer = ConfidenceScorer()

    evaluation = state.get("evaluation", {})
    result = scorer.score(evaluation)

    confidence = result["confidence"]
    band = result["band"]
    should_synthesize = result["should_synthesize"]

    log.info(
        "node.confidence_scorer.done",
        confidence=confidence,
        band=band,
        should_synthesize=should_synthesize,
    )

    # Fail-safe: if faithfulness too low, replace answer with evidence-only response
    final_answer = state.get("answer", "")
    if not should_synthesize and not state.get("synthesis_skipped", False):
        log.warning(
            "confidence_scorer.faithfulness_gate_triggered",
            faithfulness=evaluation.get("faithfulness", 0),
            threshold=FAITHFULNESS_BLOCK_THRESHOLD,
        )
        evidence_lines = [
            INSUFFICIENT_EVIDENCE_RESPONSE,
            "⚠️ Answer confidence too low (faithfulness < 0.50). "
            "Returning retrieved evidence only:\n",
        ]
        for c in state.get("reranked_chunks", [])[:5]:
            aid = c.get("arxiv_id", "")
            text = c.get("full_contextualized_text", "")[:300]
            if aid:
                evidence_lines.append(f"• [arXiv:{aid}] {text}")
        final_answer = "\n".join(evidence_lines)

    evaluation_timing = {}
    return {
        "answer": final_answer,
        "confidence": confidence,
        "confidence_band": band,
        "evaluation_timing": evaluation_timing,
    }


# ---------------------------------------------------------------------------
# Total timing aggregator (runs just before END)
# ---------------------------------------------------------------------------

def timing_aggregator_node(state: AgentState) -> dict:
    """Merge all separate timing fields into final timing dict."""
    timing = dict(state.get("timing", {}))
    
    # Merge separate timing fields from parallel nodes
    timing.update(state.get("dense_timing", {}))
    timing.update(state.get("lexical_timing", {}))
    timing.update(state.get("reranker_timing", {}))
    timing.update(state.get("evidence_timing", {}))
    timing.update(state.get("synthesis_timing", {}))
    timing.update(state.get("attribution_timing", {}))
    timing.update(state.get("verification_timing", {}))
    timing.update(state.get("evaluation_timing", {}))
    
    # Compute total time
    total = sum(v for v in timing.values() if isinstance(v, (int, float)))
    timing["total_time"] = round(total, 3)
    log.info("pipeline.timing", **timing)
    return {"timing": timing}


# ---------------------------------------------------------------------------
# LangGraph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """Compile the full research agent LangGraph."""
    g = StateGraph(AgentState)

    # Core retrieval nodes
    g.add_node("decomposer", decomposer_node)
    g.add_node("dense_worker", dense_worker_node)
    g.add_node("lexical_worker", lexical_worker_node)
    g.add_node("reranker", reranker_node)

    # Research quality nodes
    g.add_node("evidence_extraction", evidence_extraction_node)
    g.add_node("grader", grader_node)
    g.add_node("neo4j_escalation", neo4j_escalation_node)
    g.add_node("synthesis", synthesis_node)
    g.add_node("attribution_checker", attribution_checker_node)
    g.add_node("verifier", verifier_node)
    g.add_node("ragas_evaluator", ragas_evaluator_node)
    g.add_node("confidence_scorer", confidence_scorer_node)
    g.add_node("timing_aggregator", timing_aggregator_node)

    # Edges
    g.set_entry_point("decomposer")
    g.add_edge("decomposer", "dense_worker")
    g.add_edge("decomposer", "lexical_worker")
    g.add_edge("dense_worker", "reranker")
    g.add_edge("lexical_worker", "reranker")
    g.add_edge("reranker", "evidence_extraction")
    g.add_edge("evidence_extraction", "grader")
    g.add_conditional_edges(
        "grader",
        should_escalate,
        {"neo4j_escalation": "neo4j_escalation", "synthesis": "synthesis"},
    )
    g.add_edge("neo4j_escalation", "synthesis")
    g.add_edge("synthesis", "attribution_checker")
    g.add_edge("attribution_checker", "verifier")
    g.add_edge("verifier", "ragas_evaluator")
    g.add_edge("ragas_evaluator", "confidence_scorer")
    g.add_edge("confidence_scorer", "timing_aggregator")
    g.add_edge("timing_aggregator", END)

    return g.compile()


# Compiled graph singleton
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> AgentState:
    """
    Run a query through the full research assistant pipeline.
    Returns the final AgentState — never raises.
    """
    t_start = time.perf_counter()
    try:
        graph = get_graph()
        initial_state: AgentState = {
            "query": query,
            "sub_queries": [],
            "dense_results": [],
            "lexical_results": [],
            "reranked_chunks": [],
            "graph_context": [],
            "evidence": [],
            "attribution": [],
            "answer": "",
            "citations": [],
            "valid_citations": [],
            "invalid_citations": [],
            "evaluation": {},
            "confidence": 0.0,
            "confidence_band": "LOW",
            "timing": {},
            "dense_timing": {},
            "lexical_timing": {},
            "reranker_timing": {},
            "evidence_timing": {},
            "synthesis_timing": {},
            "attribution_timing": {},
            "verification_timing": {},
            "evaluation_timing": {},
            "escalated": False,
            "research_mode": RESEARCH_MODE,
            "synthesis_skipped": False,
        }
        final_state = graph.invoke(initial_state)
        elapsed = time.perf_counter() - t_start
        log.info("pipeline.completed", query=query[:80], total_time_s=round(elapsed, 3))
        return final_state
    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        log.error("pipeline.failed", error=str(exc), elapsed_s=round(elapsed, 3))
        return {
            "query": query,
            "sub_queries": [],
            "dense_results": [],
            "lexical_results": [],
            "reranked_chunks": [],
            "graph_context": [],
            "evidence": [],
            "attribution": [],
            "answer": f"⚠️ Pipeline error: {exc}",
            "citations": [],
            "valid_citations": [],
            "invalid_citations": [],
            "evaluation": {},
            "confidence": 0.0,
            "confidence_band": "LOW",
            "timing": {"total_time": round(elapsed, 3)},
            "dense_timing": {},
            "lexical_timing": {},
            "reranker_timing": {},
            "evidence_timing": {},
            "synthesis_timing": {},
            "attribution_timing": {},
            "verification_timing": {},
            "evaluation_timing": {},
            "escalated": False,
            "research_mode": RESEARCH_MODE,
            "synthesis_skipped": False,
        }
