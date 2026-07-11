"""Chunking + model tests — no API key needed."""

from ragchat.ingest import chunk_text
from ragchat.models import Answer, Citation


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_empty_text_is_no_chunks():
    assert chunk_text("   ") == []


def test_long_text_splits_with_overlap():
    # No spaces/newlines => no natural boundary => deterministic hard cuts,
    # so we can assert the overlap exactly.
    text = "a" * 3300
    chunks = chunk_text(text, size=1000, overlap=150)
    assert len(chunks) >= 3
    # the last 150 chars of one chunk are the first 150 of the next
    assert chunks[0][-150:] == chunks[1][:150]


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", size=100, overlap=100)


def test_answer_formats_with_sources():
    ans = Answer(
        text="The refund window is 30 days [1].",
        citations=[Citation(source="handbook.md", chunk_index=0, snippet="30 days")],
    )
    out = ans.format()
    assert "30 days" in out
    assert "Sources:" in out
    assert "handbook.md" in out


def test_answer_without_citations_has_no_sources_block():
    ans = Answer(text="I don't know.", citations=[])
    assert "Sources:" not in ans.format()
