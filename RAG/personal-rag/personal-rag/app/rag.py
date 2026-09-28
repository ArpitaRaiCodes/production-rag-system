"""The retrieval-augmented generation pipeline.

retrieve -> build a grounded prompt -> generate an answer that cites sources.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app import llm, vectorstore
from app.config import settings

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided by "
    "the user. Rules:\n"
    "1. Base every statement on the passages. Do not add outside knowledge.\n"
    "2. Cite the passage you used inline, like [1] or [2].\n"
    "3. If the passages do not contain the answer, reply exactly: "
    "\"I could not find that in your documents.\"\n"
    "4. Be direct and concise. No preamble."
)

NO_CONTEXT_ANSWER = (
    "I could not find that in your documents. Upload a file first, or try "
    "rephrasing the question using words that appear in your documents."
)


def format_context(hits: list[dict[str, Any]]) -> str:
    blocks = []
    for hit in hits:
        location = f", page {hit['page']}" if hit.get("page") else ""
        blocks.append(
            f"[{hit['rank']}] Source: {hit['filename']}{location}\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    user_content = (
        f"Context passages:\n\n{format_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above, with inline citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def retrieve(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    hits = vectorstore.search(question, top_k=top_k or settings.top_k)
    kept = [h for h in hits if h["score"] >= settings.min_score]
    # If the filter removes everything but we did get results, keep the best one
    # rather than pretending the knowledge base is empty.
    return kept or hits[:1]


def answer(
    question: str,
    top_k: int | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    hits = retrieve(question, top_k)
    if not hits:
        return {"answer": NO_CONTEXT_ANSWER, "sources": [], "grounded": False}

    text = llm.generate(
        build_messages(question, hits),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return {"answer": text, "sources": hits, "grounded": True}


def stream_answer(
    question: str,
    top_k: int | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[list[dict[str, Any]], Iterator[str]]:
    """Return the sources immediately, plus a token iterator for the answer."""
    hits = retrieve(question, top_k)
    if not hits:
        return [], iter([NO_CONTEXT_ANSWER])
    tokens = llm.stream(
        build_messages(question, hits),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return hits, tokens
