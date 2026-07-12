"""RagIndex — the public API that ties ingest, embed, store, and generation together.

idx = RagIndex()
idx.ingest(["examples/"])
print(idx.query("What is the refund window?").format())
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from google import genai
from google.genai import types

from . import _gemini
from ._gemini import OnRetry
from .config import GEN_MODEL, TOP_K, require_api_key
from .embed import embed_documents, embed_query
from .ingest import chunks_for_file, discover_files
from .models import Answer, Citation, Chunk
from .store import VectorStore

SYSTEM = (
    "You answer questions using ONLY the provided context passages. "
    "Each passage is numbered like [1], [2]. Cite the passages you use inline, "
    "e.g. 'The refund window is 30 days [2].' "
    "Answer yes/no questions directly from the context. A statement that "
    "something is NOT offered, allowed, or available is itself a valid answer "
    "(answer 'no' and cite it) - not a reason to decline. "
    "Only if the context genuinely says nothing about the question, reply exactly: "
    '"I don\'t know based on the provided documents." Do not use outside knowledge.'
)


class RagIndex:
    def __init__(
        self, store: VectorStore | None = None, client: genai.Client | None = None
    ) -> None:
        self._store = store or VectorStore()
        self._client = client  # lazily created so tests can inject one

    def _get_client(self) -> genai.Client:
        """The one client used for both embedding and generation."""
        if self._client is None:
            self._client = genai.Client(api_key=require_api_key())
        return self._client

    # ---- ingest -------------------------------------------------------------
    def ingest(self, paths: list[str]) -> int:
        """Index every supported file under `paths`. Returns chunks added."""
        files = discover_files(paths)
        total = 0
        for path in files:
            chunks = chunks_for_file(path)
            if not chunks:
                continue
            embeddings = embed_documents([c.text for c in chunks], self._get_client())
            self._store.add(chunks, embeddings)
            total += len(chunks)
        return total

    def count(self) -> int:
        """How many chunks are currently indexed."""
        return self._store.count()

    # ---- query --------------------------------------------------------------
    def retrieve(self, question: str, top_k: int = TOP_K) -> list[Chunk]:
        return self._store.query(embed_query(question, self._get_client()), top_k=top_k)

    def query(
        self, question: str, top_k: int = TOP_K, on_retry: OnRetry | None = None
    ) -> Answer:
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return Answer(
                text="I don't know based on the provided documents.", citations=[]
            )

        response = _gemini.generate(
            self._get_client(),
            model=GEN_MODEL,
            contents=_prompt(question, retrieved),
            config=_gen_config(),
            on_retry=on_retry,
        )
        text = (response.text or "").strip()
        return Answer(text=text, citations=cite_answer(text, retrieved))

    def query_stream(
        self, question: str, top_k: int = TOP_K, on_retry: OnRetry | None = None
    ) -> tuple[Iterator[str], list[Chunk]]:
        """Like query(), but yields the answer text incrementally.

        Returns (text_chunks, retrieved): iterate text_chunks for the answer
        as it streams in, then pass the joined text + retrieved to
        cite_answer() to build the Sources list once streaming finishes.
        """
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:

            def _decline() -> Iterator[str]:
                yield "I don't know based on the provided documents."

            return _decline(), []

        response_stream = _gemini.generate_stream(
            self._get_client(),
            model=GEN_MODEL,
            contents=_prompt(question, retrieved),
            config=_gen_config(),
            on_retry=on_retry,
        )

        def _text_chunks() -> Iterator[str]:
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        return _text_chunks(), retrieved


def _prompt(question: str, retrieved: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {c.source})\n{c.text}" for i, c in enumerate(retrieved, 1)
    )
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


def _gen_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def cite_answer(answer_text: str, retrieved: list[Chunk]) -> list[Citation]:
    """Keep only the passages the answer actually referenced via [n] markers."""
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    citations = []
    for i, chunk in enumerate(retrieved, 1):
        if i in used:
            snippet = chunk.text[:160].replace("\n", " ").strip()
            citations.append(
                Citation(
                    source=chunk.source, chunk_index=chunk.chunk_index, snippet=snippet
                )
            )
    return citations
