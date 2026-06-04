"""
contextual_retrieval/prompts.py

Anthropic-style contextual retrieval prompt constants.
Used by the Contextualizer to generate retrieval-optimized context
for every chunk before embedding.
"""

CONTEXTUALIZER_SYSTEM_PROMPT = """\
You are generating retrieval context for academic research papers.

Given:
1. Document title
2. Document summary (abstract excerpt)
3. Section title (if available)
4. The current chunk text

Write 2-4 sentences describing:
- What document this chunk belongs to
- What topic is being discussed in this section
- How this chunk fits into the broader document (e.g. introduces method, presents results, discusses limitations)

Rules:
- Do NOT summarize the chunk itself.
- Do NOT copy sentences from the chunk.
- Return ONLY the contextual description — no preamble, no labels, no JSON.
- Be specific: mention the document title and key concept names.
- Write in present tense.
"""

CONTEXTUALIZER_USER_TEMPLATE = """\
Document title: {title}

Document summary: {summary}

Section title: {section_title}

Current chunk:
{chunk_text}
"""

# Fallback template used when LLM call fails
FALLBACK_CONTEXT_TEMPLATE = (
    "This chunk is from the paper '{title}'. "
    "It discusses: {summary_short}"
)
