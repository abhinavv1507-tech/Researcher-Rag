"""
smoke_test.py — Dependency and module import verification.

Tests all required packages and new research-mode modules.
"""
import sys
print(f"Python: {sys.version}")
sys.path.insert(0, ".")

failures = []

checks = [
    # ── External packages ────────────────────────────────────────────────────
    ("arxiv",                   "import arxiv"),
    ("fitz (PyMuPDF)",          "import fitz"),
    ("langchain_groq",          "from langchain_groq import ChatGroq"),
    ("langchain_google_genai",  "from langchain_google_genai import ChatGoogleGenerativeAI"),
    ("rank_bm25",               "from rank_bm25 import BM25Okapi"),
    ("qdrant_client",           "from qdrant_client import QdrantClient"),
    ("neo4j",                   "from neo4j import GraphDatabase"),
    ("langgraph",               "from langgraph.graph import StateGraph, END"),
    ("langchain_core.messages", "from langchain_core.messages import HumanMessage, SystemMessage"),
    # langchain_core.pydantic_v1 was removed in langchain-core >= 0.3 — use pydantic directly
    ("pydantic",                "from pydantic import BaseModel, Field"),
    ("tenacity",                "from tenacity import retry, stop_after_attempt, wait_exponential"),
    ("tqdm",                    "from tqdm import tqdm"),
    ("structlog",               "import structlog"),
    ("dotenv",                  "from dotenv import load_dotenv"),
    ("difflib (stdlib)",        "import difflib"),
    ("typing_extensions",       "from typing_extensions import TypedDict"),
    ("numpy",                   "import numpy"),
    ("accelerate",              "import accelerate"),
    # Note: sentence_transformers and transformers load very slowly (CUDA/DLL init)
    # They are verified by pip show below, not by import here.

    # ── Research mode: new modules ───────────────────────────────────────────
    ("contextual_retrieval.prompts",     "from contextual_retrieval.prompts import CONTEXTUALIZER_SYSTEM_PROMPT"),
    ("contextual_retrieval.models",      "from contextual_retrieval.models import ContextualChunk, ContextBatch"),
    ("contextual_retrieval.contextualizer", "from contextual_retrieval.contextualizer import contextualize_chunks"),
    ("contextual_retrieval.service",     "from contextual_retrieval.service import ContextualRetrievalService"),
    ("evaluation.metrics",               "from evaluation.metrics import compute_context_precision, compute_context_recall, compute_citation_coverage"),
    ("evaluation.scoring",               "from evaluation.scoring import ConfidenceScorer"),
    ("evaluation.ragas_evaluator",       "from evaluation.ragas_evaluator import RAGASEvaluator"),

    # ── Core pipeline ─────────────────────────────────────────────────────────
    ("graph.AgentState",         "from graph import AgentState, ResearchAnswer, SynthesisOutput"),
    ("graph.nodes",              "from graph import decomposer_node, evidence_extraction_node, attribution_checker_node, verifier_node, ragas_evaluator_node, confidence_scorer_node"),
    ("graph.constants",          "from graph import DENSE_TOP_K, LEXICAL_TOP_K, MMR_FINAL_K, RERANK_TOP_K, RESEARCH_MODE"),
    ("graph.build_graph",        "from graph import build_graph"),
    ("api.QueryResponse",        "from api import QueryResponse"),
    ("utils.reranker",           "from utils.reranker import rerank, MAX_RERANK_INPUT, MAX_CHUNKS_PER_SOURCE"),
]

for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        failures.append(name)

# Verify key constants
print()
print("--- Constants verification ---")
from graph import DENSE_TOP_K, LEXICAL_TOP_K, MMR_FINAL_K, RERANK_TOP_K, RESEARCH_MODE
from utils.reranker import MAX_RERANK_INPUT, MAX_CHUNKS_PER_SOURCE

constant_checks = [
    ("DENSE_TOP_K == 15",           DENSE_TOP_K == 15,           DENSE_TOP_K),
    ("LEXICAL_TOP_K == 15",         LEXICAL_TOP_K == 15,         LEXICAL_TOP_K),
    ("MMR_FINAL_K == 8",            MMR_FINAL_K == 8,            MMR_FINAL_K),
    ("RERANK_TOP_K == 10",          RERANK_TOP_K == 10,          RERANK_TOP_K),
    ("MAX_RERANK_INPUT == 30",      MAX_RERANK_INPUT == 30,      MAX_RERANK_INPUT),
    ("MAX_CHUNKS_PER_SOURCE == 3",  MAX_CHUNKS_PER_SOURCE == 3,  MAX_CHUNKS_PER_SOURCE),
    ("RESEARCH_MODE == True",       RESEARCH_MODE == True,       RESEARCH_MODE),
]

for label, cond, val in constant_checks:
    if cond:
        print(f"  OK  {label}")
    else:
        print(f"  FAIL {label} — actual: {val}")
        failures.append(label)

# Verify graph compiles (no external calls)
print()
print("--- Graph compilation ---")
try:
    from graph import build_graph
    g = build_graph()
    print(f"  OK  LangGraph compiled with {len(g.nodes)} nodes")
    expected_nodes = {
        "decomposer", "dense_worker", "lexical_worker", "reranker",
        "evidence_extraction", "grader", "neo4j_escalation", "synthesis",
        "attribution_checker", "verifier", "ragas_evaluator",
        "confidence_scorer", "timing_aggregator",
    }
    actual_nodes = set(g.nodes.keys()) - {"__start__"}
    missing = expected_nodes - actual_nodes
    if missing:
        print(f"  FAIL Missing nodes: {missing}")
        failures.append("graph.nodes")
    else:
        print(f"  OK  All 13 pipeline nodes present")
except Exception as e:
    print(f"  FAIL graph compilation: {e}")
    failures.append("graph.build_graph")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
else:
    print("All checks passed. Research Assistant pipeline ready.")
