"""
store.py — Stage 2: Storage & Indexing Engine

Populates three storage layers from chunks.json:
  1. Qdrant Cloud  — dense vectors via all-MiniLM-L6-v2
  2. BM25          — local lexical index (rank_bm25)
  3. Neo4j AuraDB  — strict-schema knowledge graph
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any
from itertools import cycle

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import (  # type: ignore
    Distance,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi  # type: ignore
from utils.model_cache import get_embedding_model, EMBED_MODEL_NAME
from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
from tqdm import tqdm  # type: ignore

from utils.logging_config import configure_logging, get_logger
from utils.neo4j_client import Neo4jClient

load_dotenv()
configure_logging()
log = get_logger(__name__)

CHUNKS_FILE = Path("chunks.json")
BM25_PKL = Path("bm25_index.pkl")
QDRANT_COLLECTION = "research_chunks"
EMBED_MODEL = EMBED_MODEL_NAME  # read from .env: EMBED_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)
EMBED_DIM = 384
QDRANT_BATCH = 2
GROQ_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM entity extraction
# ---------------------------------------------------------------------------

class MethodEntity(BaseModel):
    name: str = Field(description="Name of the method or technique")
    description: str = Field(description="One-sentence description of what it does")
    relation: str = Field(
        default="USES_METHOD",
        description="Relationship to paper: USES_METHOD or PROPOSES",
    )


class MetricEntity(BaseModel):
    name: str = Field(description="Name of the evaluation metric")
    unit: str = Field(default="", description="Unit of measurement if applicable")


class PaperEntities(BaseModel):
    methods: list[MethodEntity] = Field(
        default_factory=list,
        description="Methods or techniques used or proposed by this paper",
    )
    metrics: list[MetricEntity] = Field(
        default_factory=list,
        description="Evaluation metrics the paper uses",
    )
    cites: list[str] = Field(
        default_factory=list,
        description="List of arXiv IDs that this paper explicitly cites (e.g. '2401.12345')",
    )


# ---------------------------------------------------------------------------
# Stage 2a — Qdrant Dense Store
# ---------------------------------------------------------------------------

def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def _ensure_qdrant_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        log.info("qdrant.collection_created", name=QDRANT_COLLECTION)
    else:
        log.info("qdrant.collection_exists", name=QDRANT_COLLECTION)


def build_qdrant_store(chunks: list[dict[str, Any]]) -> None:
    """Embed all chunks with the singleton embedding model and upsert into Qdrant."""
    log.info("qdrant.start", n_chunks=len(chunks), embed_model=EMBED_MODEL)
    embedder = get_embedding_model()  # singleton — loaded once, reused here
    client = _get_qdrant_client()
    _ensure_qdrant_collection(client)

    texts = [c["full_contextualized_text"] for c in chunks]

    # Embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), QDRANT_BATCH), desc="Embedding"):
        batch = texts[i : i + QDRANT_BATCH]
        vecs = embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(vecs.tolist())

    # Upsert in batches with retry logic
    for i in tqdm(range(0, len(chunks), QDRANT_BATCH), desc="Upserting to Qdrant"):
        batch_chunks = chunks[i : i + QDRANT_BATCH]
        batch_vecs = all_embeddings[i : i + QDRANT_BATCH]
        points = [
            PointStruct(
                id=c["chunk_id"],
                vector=vec,
                payload={
                    k: v
                    for k, v in c.items()
                    if k != "chunk_id"  # id is stored separately
                },
            )
            for c, vec in zip(batch_chunks, batch_vecs)
        ]
        
        # Retry with exponential backoff for network issues
        retry_count = 0
        while retry_count < 5:
            try:
                client.upsert(collection_name=QDRANT_COLLECTION, points=points, timeout=60)
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= 5:
                    log.error("qdrant.upsert_failed", batch=i, error=str(e))
                    raise
                wait_time = 2 ** retry_count
                log.warning("qdrant.upsert_retry", batch=i, retry=retry_count, wait_seconds=wait_time)
                import time
                time.sleep(wait_time)

    log.info("qdrant.done", upserted=len(chunks))


# ---------------------------------------------------------------------------
# Stage 2b — BM25 Lexical Store
# ---------------------------------------------------------------------------

def build_bm25_store(chunks: list[dict[str, Any]]) -> None:
    """Build and persist a BM25 index over full_contextualized_text."""
    log.info("bm25.start", n_chunks=len(chunks))

    texts = [c["full_contextualized_text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Tokenise (simple whitespace split; adequate for BM25 retrieval)
    tokenized = [t.lower().split() for t in tqdm(texts, desc="Tokenising for BM25")]
    index = BM25Okapi(tokenized)

    data = {"index": index, "chunk_ids": chunk_ids}
    with open(BM25_PKL, "wb") as f:
        pickle.dump(data, f)

    log.info("bm25.done", path=str(BM25_PKL), n_docs=len(chunk_ids))


# ---------------------------------------------------------------------------
# Stage 2c — Neo4j Graph Store
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def _extract_entities_llm(
    llm_with_schema: Any, paper: dict[str, Any]
) -> PaperEntities:
    """Use Groq with structured output to extract entities from a paper chunk sample."""
    # Use title + abstract + first chunk text as context
    context = f"Title: {paper['title']}\n\nAbstract excerpt:\n{paper.get('abstract', '')}\n\nContent sample:\n{paper.get('sample_text', '')[:3000]}"
    messages = [
        SystemMessage(
            content=(
                "You are an expert at extracting structured entities from AI research papers. "
                "Extract only methods/techniques that are explicitly named, evaluation metrics, "
                "and arXiv IDs of papers cited. "
                "Only include USES_METHOD if the method is used but not introduced; "
                "use PROPOSES if the paper introduces it. "
                "Return arXiv IDs without the 'arXiv:' prefix."
            )
        ),
        HumanMessage(content=context),
    ]
    return llm_with_schema.invoke(messages)


def _create_groq_clients() -> list[Any]:
    """Create Groq clients from all available API keys."""
    api_keys_str = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
    
    if not api_keys_str:
        raise ValueError("No GROQ_API_KEYS or GROQ_API_KEY found in environment")
    
    # Support both single key and multiple keys separated by comma or semicolon
    api_keys = [key.strip() for key in api_keys_str.replace(";", ",").split(",") if key.strip()]
    
    if not api_keys:
        raise ValueError("No valid GROQ API keys found")
    
    clients = []
    for api_key in api_keys:
        client = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.0,
            api_key=api_key,
        )
        clients.append(client.with_structured_output(PaperEntities))
    
    log.info("groq.clients_created", count=len(clients))
    return clients


def build_neo4j_store(chunks: list[dict[str, Any]], paper_texts: dict[str, str]) -> None:
    """Extract entities from each paper and write to Neo4j with strict schema."""
    log.info("neo4j.start")

    # Create multiple clients and setup round-robin
    llm_clients = _create_groq_clients()
    llm_cycle = cycle(llm_clients)

    # Deduplicate: one extraction per paper
    seen_papers: set[str] = set()
    paper_chunk_map: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        aid = chunk["arxiv_id"]
        if aid not in seen_papers:
            seen_papers.add(aid)
            paper_chunk_map[aid] = chunk

    neo4j = Neo4jClient()
    neo4j.ensure_constraints()

    for arxiv_id, chunk in tqdm(paper_chunk_map.items(), desc="Neo4j entity extraction"):
        year = 0
        published = chunk.get("published", "")
        if published:
            try:
                year = int(published[:4])
            except ValueError:
                pass

        paper_info = {
            "title": chunk.get("title", ""),
            "abstract": "",
            "sample_text": paper_texts.get(arxiv_id, chunk.get("chunk_text", "")),
        }

        # Get next client from round-robin cycle
        current_llm = next(llm_cycle)

        try:
            entities: PaperEntities = _extract_entities_llm(current_llm, paper_info)
            neo4j.write_entities_from_extraction(
                arxiv_id=arxiv_id,
                title=chunk.get("title", ""),
                year=year,
                entities=entities.model_dump(),
            )
        except Exception as e:
            log.warning("neo4j.extraction_failed", arxiv_id=arxiv_id, error=str(e))
            # Still upsert the paper node
            try:
                neo4j.upsert_paper(arxiv_id, chunk.get("title", ""), year)
            except Exception as neo4j_err:
                log.warning("neo4j.upsert_paper_failed", arxiv_id=arxiv_id, error=str(neo4j_err))

    neo4j.close()
    log.info("neo4j.done", papers=len(paper_chunk_map), llm_clients=len(llm_clients))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_store(chunks_path: Path = CHUNKS_FILE) -> None:
    log.info("store.start", source=str(chunks_path))

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    log.info("store.loaded", n_chunks=len(chunks))

    # Collect first-chunk sample text per paper for Neo4j extraction
    paper_texts: dict[str, str] = {}
    for chunk in chunks:
        aid = chunk["arxiv_id"]
        if aid not in paper_texts:
            paper_texts[aid] = chunk.get("chunk_text", "")

    build_qdrant_store(chunks)
    build_bm25_store(chunks)
    build_neo4j_store(chunks, paper_texts)

    log.info("store.all_done")


if __name__ == "__main__":
    run_store()
