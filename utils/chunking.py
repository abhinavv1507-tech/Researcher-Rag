"""
utils/chunking.py

Paragraph Group Chunking with sliding window overlap.
Groups 3–4 consecutive paragraphs into a single chunk with 1-paragraph overlap.
"""
from __future__ import annotations

import re
from typing import Generator


def extract_paragraphs(raw_text: str) -> list[str]:
    """
    Split raw PyMuPDF block text into cleaned paragraphs.
    Paragraphs are separated by two or more newlines; single-line noise is filtered.
    """
    # Normalise line endings and split on blank lines
    blocks = re.split(r"\n{2,}", raw_text.strip())
    paragraphs: list[str] = []
    for block in blocks:
        cleaned = " ".join(block.split())  # collapse internal whitespace
        # Discard very short fragments (page numbers, headers, etc.)
        if len(cleaned) >= 60:
            paragraphs.append(cleaned)
    return paragraphs


def sliding_window_chunks(
    paragraphs: list[str],
    window: int = 4,
    stride: int = 3,
) -> Generator[str, None, None]:
    """
    Yield chunks of `window` consecutive paragraphs, advancing by `stride` each step.
    With window=4 and stride=3, every chunk overlaps the next by exactly 1 paragraph.

    Args:
        paragraphs: Ordered list of paragraph strings.
        window:     Number of paragraphs per chunk (default 4).
        stride:     Number of paragraphs to advance per step (default 3 → 1-para overlap).

    Yields:
        Joined chunk text (paragraphs separated by double newlines).
    """
    if not paragraphs:
        return

    n = len(paragraphs)
    start = 0
    while start < n:
        end = min(start + window, n)
        chunk_paras = paragraphs[start:end]
        yield "\n\n".join(chunk_paras)
        if end == n:
            break
        start += stride


def chunk_document(raw_text: str, window: int = 4, stride: int = 3) -> list[str]:
    """
    Full pipeline: raw text → paragraphs → sliding-window chunks.

    Returns:
        List of chunk strings (un-contextualized).
    """
    paragraphs = extract_paragraphs(raw_text)
    return list(sliding_window_chunks(paragraphs, window=window, stride=stride))
