"""Gemini embeddings — the one place that turns text into vectors.

Task-type hints matter: Gemini embeds a *document* and a *query* into slightly
different spaces optimised for retrieval, so we tell it which is which.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from .config import EMBED_MODEL, require_api_key

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=require_api_key())
    return _client


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    if not texts:
        return []
    result = _get_client().models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks for storage."""
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a user question for searching."""
    return _embed([text], task_type="RETRIEVAL_QUERY")[0]
