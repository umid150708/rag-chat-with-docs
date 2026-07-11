# Changelog

## 0.1.0 — 2026-07-11

First working version.

- Ingest PDF/txt/md, chunk with overlap, embed with Gemini (`gemini-embedding-001`,
  document/query task-type hints), store in a persistent Chroma vector DB.
- `RagIndex` library API + `rag ingest` / `rag ask` CLI.
- Grounded answers with inline citations; declines ("I don't know") instead of
  hallucinating when the corpus lacks the answer.
- Evaluation harness (`python -m ragchat.eval`): golden set scored on faithfulness
  and relevance (LLM-as-judge) plus deterministic retrieval hit.
- 429 retry/backoff for Gemini free-tier rate limits.
- Key-free unit tests for chunking and answer formatting.
