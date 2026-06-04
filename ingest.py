"""
ingest.py — Stage 1: Data Ingestion & Chunking

Fetches arXiv papers from cs.CL, cs.AI, cs.LG (Jan 2024 – Apr 2026),
filters for "LLM agents" / "tool use", downloads PDFs concurrently,
parses them in parallel with PyMuPDF, and writes all chunks to chunks.json.

Note: Set RESEARCH_MODE=true to enable Anthropic-style contextual retrieval.
When enabled, each chunk receives LLM-generated retrieval context
(title + abstract + section title + neighboring chunk awareness).
When disabled (default), abstract prefix is used as context_summary.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import arxiv  # type: ignore
import fitz  # PyMuPDF  # type: ignore
from dotenv import load_dotenv
from tqdm import tqdm  # type: ignore

from utils.chunking import chunk_document
from utils.logging_config import configure_logging, get_logger

RESEARCH_MODE: bool = os.getenv("RESEARCH_MODE", "false").lower() == "true"

load_dotenv()
configure_logging()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]
KEYWORDS = ["LLM agents", "tool use"]
START_DATE = "2024-01-01"
END_DATE = "2026-12-31"
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "401"))
MAX_DOWNLOAD_WORKERS = int(os.getenv("MAX_DOWNLOAD_WORKERS", "16"))
ARXIV_BATCH_SIZE = int(os.getenv("ARXIV_BATCH_SIZE", "100"))
PDF_DIR = Path("pdfs")
CHUNKS_FILE = Path("chunks.json")



# ---------------------------------------------------------------------------
# Step 1: Fetch paper metadata
# ---------------------------------------------------------------------------

def fetch_paper_metadata() -> list[dict[str, Any]]:
    """Query arXiv for papers matching keywords across target categories."""
    log.info("arxiv.fetch_start", categories=CATEGORIES, keywords=KEYWORDS)
    client = arxiv.Client(page_size=ARXIV_BATCH_SIZE, num_retries=5)

    # Build combined query
    cat_query = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    kw_query = " OR ".join(f'"{kw}"' for kw in KEYWORDS)
    search_query = f"({kw_query}) AND ({cat_query})"

    search = arxiv.Search(
        query=search_query,
        max_results=MAX_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[dict[str, Any]] = []
    for result in tqdm(client.results(search), desc="Fetching metadata", unit="paper"):
        # Filter by date range
        submitted = result.published.strftime("%Y-%m-%d")
        if submitted < START_DATE or submitted > END_DATE:
            continue
        papers.append(
            {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "published": submitted,
            }
        )
        if len(papers) >= MAX_PAPERS:
            break

    log.info("arxiv.fetch_done", count=len(papers))
    return papers


# ---------------------------------------------------------------------------
# Step 2: Parallel PDF downloads
# ---------------------------------------------------------------------------

def _download_pdf(paper: dict[str, Any]) -> tuple[str, Path | None]:
    """Download a single PDF. Returns (arxiv_id, path_or_None)."""
    arxiv_id = paper["arxiv_id"]
    dest = PDF_DIR / f"{arxiv_id}.pdf"
    if dest.exists():
        return arxiv_id, dest
    try:
        urllib.request.urlretrieve(paper["pdf_url"], dest)
        return arxiv_id, dest
    except Exception as e:
        log.warning("pdf.download_failed", arxiv_id=arxiv_id, error=str(e))
        return arxiv_id, None


def download_pdfs_parallel(papers: list[dict[str, Any]]) -> dict[str, Path]:
    """Download all PDFs concurrently. Returns {arxiv_id: pdf_path}."""
    PDF_DIR.mkdir(exist_ok=True)
    results: dict[str, Path] = {}
    log.info("pdf.download_start", count=len(papers))

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_download_pdf, p): p for p in papers}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDFs"):
            arxiv_id, path = future.result()
            if path:
                results[arxiv_id] = path

    log.info("pdf.download_done", downloaded=len(results), failed=len(papers) - len(results))
    return results


# ---------------------------------------------------------------------------
# Step 3: Parallel PDF parsing (PyMuPDF)
# ---------------------------------------------------------------------------

def _parse_pdf(args: tuple[str, Path]) -> tuple[str, str]:
    """Extract raw text from a PDF. Returns (arxiv_id, full_text)."""
    arxiv_id, pdf_path = args
    try:
        doc = fitz.open(str(pdf_path))
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return arxiv_id, "\n\n".join(pages)
    except Exception as e:
        log.warning("pdf.parse_failed", arxiv_id=arxiv_id, error=str(e))
        return arxiv_id, ""


async def parse_pdfs_async(pdf_paths: dict[str, Path]) -> dict[str, str]:
    """
    Parse all PDFs concurrently using asyncio + ThreadPoolExecutor.
    Returns {arxiv_id: full_text}.
    """
    loop = asyncio.get_event_loop()
    texts: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        parse_args = list(pdf_paths.items())
        tasks = [
            loop.run_in_executor(executor, _parse_pdf, arg)
            for arg in parse_args
        ]
        log.info("pdf.parse_start", count=len(tasks))
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Parsing PDFs",
        ):
            arxiv_id, text = await coro
            if text:
                texts[arxiv_id] = text

    log.info("pdf.parse_done", parsed=len(texts))
    return texts


# ---------------------------------------------------------------------------
# Step 4: Build chunks and save
# ---------------------------------------------------------------------------

def build_chunks(
    papers: list[dict[str, Any]],
    texts: dict[str, str],
) -> list[dict[str, Any]]:
    """
    For each paper, chunk its text. The context_summary field is populated
    from the paper's abstract (no LLM call needed).
    Returns the full list of chunk records.
    """
    all_chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for paper in tqdm(papers, desc="Building chunks"):
        arxiv_id = paper["arxiv_id"]
        text = texts.get(arxiv_id, "")
        if not text:
            log.debug("chunk.skip_no_text", arxiv_id=arxiv_id)
            continue

        # Use the abstract as the lightweight context summary (no LLM cost)
        summary = paper.get("abstract", "")[:300]
        raw_chunks = chunk_document(text)

        for i, raw_chunk in enumerate(raw_chunks):
            contextualized = f"[Context: {summary}] {raw_chunk}" if summary else raw_chunk
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": arxiv_id,
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "context_summary": summary,
                    "chunk_text": raw_chunk,
                    "full_contextualized_text": contextualized,
                    "chunk_index": i,
                    "published": paper.get("published", ""),
                }
            )
            chunk_id += 1

    log.info("chunks.built", total=chunk_id)
    return all_chunks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_ingest() -> None:
    log.info("ingest.start")

    # 1. Fetch metadata
    papers = fetch_paper_metadata()
    if not papers:
        log.error("ingest.no_papers")
        return

    # 2. Download PDFs concurrently (ThreadPoolExecutor)
    pdf_paths = download_pdfs_parallel(papers)

    # 3. Parse PDFs concurrently (asyncio + ThreadPoolExecutor)
    texts = await parse_pdfs_async(pdf_paths)

    # 4. Chunk
    chunks = build_chunks(papers, texts)

    # 4b. Contextual Retrieval enrichment (RESEARCH_MODE only)
    if RESEARCH_MODE:
        log.info("ingest.contextual_retrieval.enabled", n_chunks=len(chunks))
        try:
            from contextual_retrieval.service import ContextualRetrievalService
            # Build paper_metadata for the service
            paper_metadata = {
                p["arxiv_id"]: {
                    "arxiv_id": p["arxiv_id"],
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                }
                for p in papers
            }
            service = ContextualRetrievalService(use_llm=True)
            chunks = service.enrich(chunks, paper_metadata)
            log.info("ingest.contextual_retrieval.done", n_chunks=len(chunks))
        except Exception as exc:
            log.error(
                "ingest.contextual_retrieval.failed",
                error=str(exc),
                fallback="continuing with abstract-prefix context",
            )
    else:
        log.info("ingest.contextual_retrieval.disabled — set RESEARCH_MODE=true to enable")

    # 5. Save to disk
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("ingest.done", output=str(CHUNKS_FILE), total_chunks=len(chunks))


if __name__ == "__main__":
    asyncio.run(run_ingest())
