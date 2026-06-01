"""
graph.py — Stages 3 & 4: LangGraph Orchestration

Full pipeline graph:
  decomposer → [dense_worker ‖ lexical_worker] → reranker
             → grader → (conditional) neo4j_escalation
             → synthesis → verifier → END

Models:
  Groq/Llama 3.1  — decomposer, grader
  Gemini 2.5 Flash — synthesis (CoT + citations)
"""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from langgraph.graph import END, StateGraph  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
from typing_extensions import TypedDict  # type: ignore

from utils.logging_config import configure_logging, get_logger
from utils.neo4j_client import Neo4jClient
from utils.reranker import rerank

load_dotenv()
configure_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROQ_DECOMPOSE_MODEL = "llama-3.3-70b-versatile"
GROQ_GRADE_MODEL = "llama-3.3-70b-versatile"
GEMINI_SYNTHESIS_MODEL = "gemini-2.5-flash-preview-04-17"
QDRANT_COLLECTION = "research_chunks"
BM25_PKL = Path("bm25_index.pkl")
DENSE_TOP_K = 20
LEXICAL_TOP_K = 20
CITATION_THRESHOLD = 0.35  # SequenceMatcher ratio; below this, claim+citation is stripped


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    sub_queries: list[dict]       # [{query: str, tool: "dense_search"|"lexical_search"}]
    dense_results: list[dict]
    lexical_results: list[dict]
    reranked_chunks: list[dict]
    graph_context: list[dict]
    answer: str
    citations: list[str]
    escalated: bool


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------

class SubQuery(BaseModel):
    query: str = Field(description="A specific sub-question derived from the main query")
    tool: Literal["dense_search", "lexical_search"] = Field(
        description="Which retrieval tool to use for this sub-query"
    )


class SubQueryList(BaseModel):
    sub_queries: list[SubQuery] = Field(
        description="List of sub-queries with assigned tools"
    )


class GraderOutput(BaseModel):
    sufficient: bool = Field(
        description="True if retrieved chunks sufficiently answer the query"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


class SynthesisOutput(BaseModel):
    thought_process: str = Field(
        description="Step-by-step reasoning over the retrieved evidence"
    )
    answer: str = Field(
        description="Final answer with inline [arXiv:XXXX.XXXXX] citations"
    )
    citations: list[str] = Field(
        description="List of arXiv IDs cited in the answer (e.g. '2401.12345')"
    )


# ---------------------------------------------------------------------------
# Lazy-loaded clients
# ---------------------------------------------------------------------------

_qdrant: Optional[QdrantClient] = None
_bm25_data: Optional[dict] = None
_neo4j: Optional[Neo4jClient] = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _qdrant


def _get_bm25() -> dict:
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_PKL, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def _get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j


# ---------------------------------------------------------------------------
# LLM factory helpers
# ---------------------------------------------------------------------------

def _groq_llm(model: str = GROQ_DECOMPOSE_MODEL) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=0.0,
        api_key=os.environ["GROQ_API_KEY"],
    )


def _gemini_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_SYNTHESIS_MODEL,
        temperature=0.1,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Node: decomposer
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _decompose(query: str) -> list[dict]:
    llm = _groq_llm(GROQ_DECOMPOSE_MODEL).with_structured_output(SubQueryList)
    messages = [
        SystemMessage(
            content=(
                "You are a research query analyst. Break the user's research question "
                "into 2–4 focused sub-queries. Assign 'dense_search' to conceptual or "
                "semantic sub-queries, and 'lexical_search' to sub-queries best served "
                "by exact keyword matching (method names, metric names, author names)."
            )
        ),
        HumanMessage(content=f"Research question: {query}"),
    ]
    result: SubQueryList = llm.invoke(messages)
    return [sq.dict() for sq in result.sub_queries]


def decomposer_node(state: AgentState) -> dict:
    log.info("node.decomposer", query=state["query"])
    sub_queries = _decompose(state["query"])
    log.info("node.decomposer.done", n_sub_queries=len(sub_queries))
    return {"sub_queries": sub_queries}


# ---------------------------------------------------------------------------
# Node: dense_worker
# ---------------------------------------------------------------------------

def _qdrant_search(query_text: str, top_k: int = DENSE_TOP_K) -> list[dict]:
    from sentence_transformers import SentenceTransformer  # type: ignore
    embedder = SentenceTransformer("BAAI/bge-m3")
    vec = embedder.encode(query_text, normalize_embeddings=True).tolist()
    hits = _get_qdrant().search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
    )
    results = []
    for hit in hits:
        r = dict(hit.payload)
        r["score"] = hit.score
        r["retrieval_source"] = "dense"
        results.append(r)
    return results


def dense_worker_node(state: AgentState) -> dict:
    log.info("node.dense_worker")
    dense_queries = [sq["query"] for sq in state["sub_queries"] if sq.get("tool") == "dense_search"]
    if not dense_queries:
        dense_queries = [state["query"]]

    all_results: list[dict] = []
    for q in dense_queries:
        all_results.extend(_qdrant_search(q))

    # Deduplicate by chunk_id (keep highest score)
    seen: dict[Any, dict] = {}
    for r in all_results:
        cid = r.get("chunk_id")
        if cid not in seen or r["score"] > seen[cid]["score"]:
            seen[cid] = r
    deduped = list(seen.values())

    log.info("node.dense_worker.done", results=len(deduped))
    return {"dense_results": deduped}


# ---------------------------------------------------------------------------
# Node: lexical_worker
# ---------------------------------------------------------------------------

def _bm25_search(query_text: str, top_k: int = LEXICAL_TOP_K, chunks_lookup: dict | None = None) -> list[dict]:
    data = _get_bm25()
    index = data["index"]
    chunk_ids = data["chunk_ids"]

    tokens = query_text.lower().split()
    scores = index.get_scores(tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        r: dict[str, Any] = {"chunk_id": chunk_ids[idx], "score": float(scores[idx]), "retrieval_source": "lexical"}
        if chunks_lookup and chunk_ids[idx] in chunks_lookup:
            r.update(chunks_lookup[chunk_ids[idx]])
        results.append(r)
    return results


def lexical_worker_node(state: AgentState) -> dict:
    log.info("node.lexical_worker")
    lexical_queries = [sq["query"] for sq in state["sub_queries"] if sq.get("tool") == "lexical_search"]
    if not lexical_queries:
        lexical_queries = [state["query"]]

    # Load chunks for payload enrichment
    chunks_lookup: dict = {}
    if Path("chunks.json").exists():
        with open("chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        chunks_lookup = {c["chunk_id"]: c for c in chunks}

    all_results: list[dict] = []
    for q in lexical_queries:
        all_results.extend(_bm25_search(q, chunks_lookup=chunks_lookup))

    seen: dict[Any, dict] = {}
    for r in all_results:
        cid = r.get("chunk_id")
        if cid not in seen or r["score"] > seen[cid]["score"]:
            seen[cid] = r
    deduped = list(seen.values())

    log.info("node.lexical_worker.done", results=len(deduped))
    return {"lexical_results": deduped}


# ---------------------------------------------------------------------------
# Node: reranker
# ---------------------------------------------------------------------------

def reranker_node(state: AgentState) -> dict:
    log.info("node.reranker")
    pooled = state["dense_results"] + state["lexical_results"]
    reranked = rerank(state["query"], pooled)
    log.info("node.reranker.done", reranked=len(reranked))
    return {"reranked_chunks": reranked}


# ---------------------------------------------------------------------------
# Node: grader (Self-RAG reflection)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _grade_chunks(query: str, chunks: list[dict]) -> GraderOutput:
    llm = _groq_llm(GROQ_GRADE_MODEL).with_structured_output(GraderOutput)
    chunk_summary = "\n\n".join(
        f"[{c.get('arxiv_id', '?')}] {c.get('full_contextualized_text', '')[:400]}"
        for c in chunks[:10]
    )
    messages = [
        SystemMessage(
            content=(
                "You are a research quality assessor. Given a query and retrieved evidence chunks, "
                "determine whether the chunks provide sufficient information to fully answer the query. "
                "Consider: coverage of key concepts, presence of specific methods/metrics, and source diversity."
            )
        ),
        HumanMessage(
            content=f"Query: {query}\n\nRetrieved evidence:\n{chunk_summary}\n\nAre these chunks sufficient?"
        ),
    ]
    return llm.invoke(messages)


def grader_node(state: AgentState) -> dict:
    log.info("node.grader")
    if not state["reranked_chunks"]:
        log.warning("node.grader.no_chunks")
        return {"escalated": True}
    assessment = _grade_chunks(state["query"], state["reranked_chunks"])
    escalated = not assessment.sufficient
    log.info("node.grader.done", sufficient=assessment.sufficient, escalated=escalated)
    return {"escalated": escalated}


def should_escalate(state: AgentState) -> str:
    return "neo4j_escalation" if state["escalated"] else "synthesis"


# ---------------------------------------------------------------------------
# Node: neo4j_escalation
# ---------------------------------------------------------------------------

def neo4j_escalation_node(state: AgentState) -> dict:
    log.info("node.neo4j_escalation")
    arxiv_ids = list({c.get("arxiv_id") for c in state["reranked_chunks"] if c.get("arxiv_id")})
    if not arxiv_ids:
        return {"graph_context": []}
    try:
        graph_context = _get_neo4j().multi_hop_query(arxiv_ids)
    except Exception as e:
        log.error("node.neo4j_escalation.failed", error=str(e))
        graph_context = []
    log.info("node.neo4j_escalation.done", graph_items=len(graph_context))
    return {"graph_context": graph_context}


# ---------------------------------------------------------------------------
# Node: synthesis (Gemini 2.5 Flash)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
def _synthesize(query: str, chunks: list[dict], graph_context: list[dict]) -> SynthesisOutput:
    llm = _gemini_llm().with_structured_output(SynthesisOutput)

    chunk_text = "\n\n".join(
        f"[arXiv:{c.get('arxiv_id', '?')}] {c.get('full_contextualized_text', '')[:600]}"
        for c in chunks
    )
    graph_text = ""
    if graph_context:
        graph_text = "\n\nGraph context (multi-hop):\n" + "\n".join(
            f"- {g.get('relation', 'related')}: [{g.get('arxiv_id', '?')}] {g.get('title', '')}"
            for g in graph_context
        )

    messages = [
        SystemMessage(
            content=(
                "You are an expert AI research synthesist. "
                "You MUST structure your response with a <thought_process> block first, "
                "where you reason step-by-step over the retrieved evidence before writing your answer. "
                "Your final answer must:\n"
                "1. Synthesize ONLY from the provided chunks and graph context.\n"
                "2. Include an inline citation [arXiv:XXXX.XXXXX] after every factual claim.\n"
                "3. Never fabricate information not present in the evidence.\n"
                "4. Be comprehensive but concise."
            )
        ),
        HumanMessage(
            content=(
                f"Research question: {query}\n\n"
                f"Retrieved evidence:\n{chunk_text}"
                f"{graph_text}"
            )
        ),
    ]
    return llm.invoke(messages)


def synthesis_node(state: AgentState) -> dict:
    log.info("node.synthesis")
    output = _synthesize(state["query"], state["reranked_chunks"], state.get("graph_context", []))
    log.info("node.synthesis.done", citations=len(output.citations))
    return {"answer": output.answer, "citations": output.citations}


# ---------------------------------------------------------------------------
# Node: verifier
# ---------------------------------------------------------------------------

_ARXIV_PATTERN = re.compile(r"\[arXiv:(\d{4}\.\d{4,5})\]")
_SENTENCE_PATTERN = re.compile(r"[^.!?]*\[arXiv:[^\]]+\][^.!?]*[.!?]?")


def _find_supporting_chunk(arxiv_id: str, chunks: list[dict]) -> str | None:
    for chunk in chunks:
        if chunk.get("arxiv_id") == arxiv_id:
            return chunk.get("full_contextualized_text", "")
    return None


def _claim_supported(claim_sentence: str, chunk_text: str) -> bool:
    """
    Check support via difflib SequenceMatcher ratio >= CITATION_THRESHOLD (0.35).
    Production upgrade: replace with cosine similarity of bge-m3 embeddings (threshold >= 0.65).
    """
    ratio = difflib.SequenceMatcher(None, claim_sentence.lower(), chunk_text.lower()[:500]).ratio()
    return ratio >= CITATION_THRESHOLD


def verifier_node(state: AgentState) -> dict:
    log.info("node.verifier")
    answer = state["answer"]
    chunks = state["reranked_chunks"]
    valid_citations: list[str] = []

    # Find all claim sentences with citations
    def process_answer(text: str) -> str:
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            citations_in_line = _ARXIV_PATTERN.findall(line)
            if not citations_in_line:
                cleaned_lines.append(line)
                continue

            line_ok = True
            for arxiv_id in citations_in_line:
                chunk_text = _find_supporting_chunk(arxiv_id, chunks)
                if chunk_text is None:
                    log.warning("verifier.no_chunk", arxiv_id=arxiv_id)
                    line_ok = False
                    break
                if not _claim_supported(line, chunk_text):
                    log.warning("verifier.unsupported", arxiv_id=arxiv_id, ratio="<0.35")
                    line_ok = False
                    break

            if line_ok:
                cleaned_lines.append(line)
                for aid in citations_in_line:
                    if aid not in valid_citations:
                        valid_citations.append(aid)
            else:
                log.info("verifier.stripped_line", snippet=line[:80])

        return "\n".join(cleaned_lines)

    cleaned_answer = process_answer(answer)
    log.info("node.verifier.done", valid_citations=len(valid_citations))
    return {"answer": cleaned_answer, "citations": valid_citations}


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_graph():
    """Compile and return the full research agent graph."""
    g = StateGraph(AgentState)

    g.add_node("decomposer", decomposer_node)
    g.add_node("dense_worker", dense_worker_node)
    g.add_node("lexical_worker", lexical_worker_node)
    g.add_node("reranker", reranker_node)
    g.add_node("grader", grader_node)
    g.add_node("neo4j_escalation", neo4j_escalation_node)
    g.add_node("synthesis", synthesis_node)
    g.add_node("verifier", verifier_node)

    # Edges
    g.set_entry_point("decomposer")
    g.add_edge("decomposer", "dense_worker")
    g.add_edge("decomposer", "lexical_worker")
    g.add_edge("dense_worker", "reranker")
    g.add_edge("lexical_worker", "reranker")
    g.add_edge("reranker", "grader")
    g.add_conditional_edges(
        "grader",
        should_escalate,
        {"neo4j_escalation": "neo4j_escalation", "synthesis": "synthesis"},
    )
    g.add_edge("neo4j_escalation", "synthesis")
    g.add_edge("synthesis", "verifier")
    g.add_edge("verifier", END)

    return g.compile()


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> AgentState:
    """Run a single query through the full pipeline and return final state."""
    graph = get_graph()
    initial_state: AgentState = {
        "query": query,
        "sub_queries": [],
        "dense_results": [],
        "lexical_results": [],
        "reranked_chunks": [],
        "graph_context": [],
        "answer": "",
        "citations": [],
        "escalated": False,
    }
    final_state = graph.invoke(initial_state)
    return final_state
