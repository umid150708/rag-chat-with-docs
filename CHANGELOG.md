# Changelog

## 0.2.0 — 2026-07-12

Streamlit chat UI + live deployment.

- `app.py`: public Streamlit chat over the bundled sample corpus, with chat
  history, citations, and friendly rate-limit/invalid-key messages.
- Bring-your-own-key design: the host key (Streamlit secrets) indexes the
  corpus once per container; each visitor's own free Gemini key pays for their
  questions, so the public link never exhausts our quota.
- `embed.py` refactor: `embed_documents` / `embed_query` now take an explicit
  `genai.Client` (module-global client removed) — embedding is injectable, the
  enabler for the two-key design. CLI behaviour unchanged.
- `requirements.txt` exported from `uv.lock` for Streamlit Community Cloud.
- New key-free unit tests for the injectable embedding client.

## 0.1.1 — 2026-07-11

Quality tooling pass.

- Add mypy (type checking) and vulture (dead-code) to the dev toolchain and
  `CLAUDE.md` health stack.
- Fix two None-safety gaps mypy surfaced: embeddings API returning no vectors,
  and empty Chroma query results being indexed unguarded.
- Add a Mermaid architecture diagram to the README (`docs/architecture.mmd`).

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
