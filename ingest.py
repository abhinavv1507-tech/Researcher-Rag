"""
ingest.py — Stage 1: Data Ingestion & Contextual Chunking

Fetches arXiv papers from cs.CL, cs.AI, cs.LG (Jan 2024 – Apr 2026),
filters for "LLM agents" / "tool use", downloads PDFs concurrently,
parses them in parallel with PyMuPDF, prepends LLM contextual summaries,
and writes all chunks to chunks.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import arxiv  # type: ignore
import fitz  # PyMuPDF  # type: ignore
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
from tqdm import tqdm  # type: ignore

from utils.chunking import chunk_document
from utils.logging_config import configure_logging, get_logger

load_dotenv()
configure_logging()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]
KEYWORDS = ["LLM agents", "tool use"]
START_DATE = "2024-01-01"
END_DATE = "2026-04-30"
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "700"))
MAX_DOWNLOAD_WORKERS = int(os.getenv("MAX_DOWNLOAD_WORKERS", "16"))
ARXIV_BATCH_SIZE = int(os.getenv("ARXIV_BATCH_SIZE", "100"))
PDF_DIR = Path("pdfs")
CHUNKS_FILE = Path("chunks.json")
SUMMARY_MODEL = "gemini-1.5-flash"
CONTEXT_TOKEN_LIMIT = 6000  # characters, not tokens — conservative limit


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
# Step 4: LLM contextual summary generation
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def _generate_summary(llm: ChatGoogleGenerativeAI, text: str, title: str) -> str:
    """Generate a 2–3 sentence contextual summary for a paper."""
    truncated = text[:CONTEXT_TOKEN_LIMIT]
    messages = [
        SystemMessage(
            content=(
                "You are an expert AI research assistant. "
                "Given a research paper's content, produce exactly 2–3 sentences "
                "summarising its core contribution. Be specific and technical. "
                "Do not include phrases like 'This paper' or 'The authors'."
            )
        ),
        HumanMessage(
            content=f"Title: {title}\n\nContent:\n{truncated}\n\nCore contribution summary:"
        ),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def generate_summaries(
    papers: list[dict[str, Any]], texts: dict[str, str]
) -> dict[str, str]:
    """
    Generate contextual summaries for all papers using Google Gemini.
    Returns {arxiv_id: summary}.
    Uses the paper abstract as fallback if text is unavailable or API fails.
    """
    llm = ChatGoogleGenerativeAI(
        model=SUMMARY_MODEL,
        temperature=0.0,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    # --- Preflight: validate key with one cheap call before looping all papers ---
    _gemini_available = True
    try:
        llm.invoke([HumanMessage(content="ping")])
        log.info("gemini.key_valid")
    except Exception as e:
        log.warning(
            "gemini.key_invalid",
            msg=f"Gemini preflight failed — skipping LLM summaries, using abstracts instead. Error: {e}"
        )
        _gemini_available = False

    summaries: dict[str, str] = {}
    paper_map = {p["arxiv_id"]: p for p in papers}

    for arxiv_id, text in tqdm(texts.items(), desc="Generating summaries"):
        paper = paper_map.get(arxiv_id, {})
        title = paper.get("title", arxiv_id)

        if not _gemini_available:
            # Fast path: abstract fallback, no API calls
            summaries[arxiv_id] = paper.get("abstract", "No summary available.")[:300]
            continue

        # Use abstract if text is too short
        effective_text = text if len(text) > 200 else paper.get("abstract", text)
        try:
            summary = _generate_summary(llm, effective_text, title)
        except Exception as e:
            log.warning("summary.failed", arxiv_id=arxiv_id, error=str(e))
            summary = paper.get("abstract", "No summary available.")[:300]
        summaries[arxiv_id] = summary
        # Brief sleep to respect Gemini rate limits (15 RPM on free tier)
        time.sleep(4.0)

    log.info("summaries.done", count=len(summaries))
    return summaries


# ---------------------------------------------------------------------------
# Step 5: Build chunks and save
# ---------------------------------------------------------------------------

def build_chunks(
    papers: list[dict[str, Any]],
    texts: dict[str, str],
    summaries: dict[str, str],
) -> list[dict[str, Any]]:
    """
    For each paper, chunk its text and prepend the contextual summary.
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

        summary = summaries.get(arxiv_id, "")
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

    # 4. Generate LLM summaries
    summaries = generate_summaries(papers, texts)

    # 5. Chunk and contextualize
    chunks = build_chunks(papers, texts, summaries)

    # 6. Save to disk
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("ingest.done", output=str(CHUNKS_FILE), total_chunks=len(chunks))


if __name__ == "__main__":
    asyncio.run(run_ingest())
