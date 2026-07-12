"""Gemini embeddings — the one place that turns text into vectors.

Task-type hints matter: Gemini embeds a *document* and a *query* into slightly
different spaces optimised for retrieval, so we tell it which is which.

The caller supplies the client: the same store can be indexed with one key
(the host's) and queried with another (a visitor's).
"""

from __future__ import annotations

from google import genai
from google.genai import types

from .config import EMBED_MODEL


def _embed(texts: list[str], task_type: str, client: genai.Client) -> list[list[float]]:
    if not texts:
        return []
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,  # type: ignore[arg-type]  # genai stubs want an invariant union list; list[str] is fine at runtime
        config=types.EmbedContentConfig(task_type=task_type),
    )
    if not result.embeddings:
        raise RuntimeError("Embedding API returned no vectors.")
    return [list(e.values or []) for e in result.embeddings]


def embed_documents(texts: list[str], client: genai.Client) -> list[list[float]]:
    """Embed chunks for storage."""
    return _embed(texts, "RETRIEVAL_DOCUMENT", client)


def embed_query(text: str, client: genai.Client) -> list[float]:
    """Embed a user question for searching."""
    return _embed([text], "RETRIEVAL_QUERY", client)[0]
