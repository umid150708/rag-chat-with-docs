"""The committed corpus snapshot must stay loadable and shaped right.

The deployed demo boots from data/corpus_embeddings.json with no API key —
if this file rots (bad JSON, missing fields, inconsistent dimensions), the
public app breaks at startup. Regenerate with scripts/build_corpus_snapshot.py.
"""

import json
from pathlib import Path

SNAPSHOT = Path(__file__).parent.parent / "data" / "corpus_embeddings.json"


def test_snapshot_exists_and_parses():
    data = json.loads(SNAPSHOT.read_text())
    assert data["model"] == "gemini-embedding-001"
    assert len(data["records"]) >= 2  # handbook.md + roasting.md at minimum


def test_snapshot_records_are_complete_and_consistent():
    records = json.loads(SNAPSHOT.read_text())["records"]
    dims = set()
    for r in records:
        assert r["text"].strip()
        assert r["source"].startswith("examples/")
        assert isinstance(r["chunk_index"], int)
        assert all(isinstance(v, float) for v in r["embedding"])
        dims.add(len(r["embedding"]))
    assert len(dims) == 1  # every vector in the same embedding space


def test_snapshot_covers_every_corpus_file():
    sources = {r["source"] for r in json.loads(SNAPSHOT.read_text())["records"]}
    corpus = Path(__file__).parent.parent / "examples"
    for doc in corpus.iterdir():
        assert f"examples/{doc.name}" in sources, f"{doc.name} missing from snapshot"
