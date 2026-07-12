"""Precompute embeddings for the bundled demo corpus.

Run after editing examples/:

    uv run python scripts/build_corpus_snapshot.py

Writes data/corpus_embeddings.json, which app.py loads at startup — so the
deployed demo needs no API key of its own (visitors bring their own key for
questions; the corpus vectors are baked in).
"""

from __future__ import annotations

import json
from pathlib import Path

from google import genai

from ragchat.config import EMBED_MODEL, require_api_key
from ragchat.embed import embed_documents
from ragchat.ingest import chunks_for_file, discover_files

CORPUS_DIR = "examples"
OUT = Path("data/corpus_embeddings.json")


def main() -> None:
    client = genai.Client(api_key=require_api_key())
    records = []
    for path in discover_files([CORPUS_DIR]):
        chunks = chunks_for_file(path)
        if not chunks:
            continue
        embeddings = embed_documents([c.text for c in chunks], client)
        for chunk, emb in zip(chunks, embeddings):
            records.append(
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "embedding": emb,
                }
            )
    if not records:
        raise SystemExit(f"No chunks found under {CORPUS_DIR}/ — nothing written.")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"model": EMBED_MODEL, "records": records}))
    print(f"Wrote {len(records)} chunk embeddings ({EMBED_MODEL}) to {OUT}")


if __name__ == "__main__":
    main()
