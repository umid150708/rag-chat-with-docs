# Spec — RAG "Chat With Your Docs"

## Context

An LLM alone can't answer questions about *your* documents — it never saw them,
and if you paste them in they blow past the context window and cost a fortune.
**Retrieval-Augmented Generation (RAG)** fixes this: index the documents once,
then at question time retrieve only the few most relevant chunks and feed those
to the model. This is the most-deployed pattern in production AI (support
copilots, internal search, "ask your policy docs"). This project builds a small
but production-literate RAG system with **citations** and an **evaluation
harness** — the two things that separate a real system from a demo.

Portfolio project 2 of an AI-engineering track. Built on Google Gemini (free
tier) for both embeddings and generation.

## Current State

Greenfield. Empty git repo at `02-rag-chat-with-docs/`. Reuses the toolchain
proven in project 1: Python 3.11+, `uv`, `google-genai`, `pydantic`,
`python-dotenv`, `ruff`, `pytest`.

## Proposed Design

```
  ┌─────────── INGEST (once) ───────────┐        ┌────────── ASK (per question) ──────────┐
  PDF / .txt / .md                                question
      │ load (pypdf / plain read)                     │ embed (Gemini)
      │ chunk (~1000 chars, 150 overlap)              │ similarity search (Chroma, top-k)
      │ embed each chunk (Gemini)                     │ build grounded prompt w/ retrieved chunks
      ▼                                               ▼ generate (Gemini) → answer + citations
  Chroma (persistent, on disk)  ◀──── same store ────┘
```

### Components (`src/ragchat/`)

| File | Role |
|------|------|
| `config.py` | Model IDs, chunk sizes, paths; loads `GEMINI_API_KEY` from `.env`. |
| `ingest.py` | Load PDF/txt/md → normalise text → chunk with overlap. Pure, testable. |
| `embed.py` | Gemini embeddings wrapper (batched); one place that calls the embed API. |
| `store.py` | Chroma wrapper: add chunks (+ metadata), query top-k. |
| `rag.py` | `RagIndex` orchestration: `ingest(paths)`, `query(question) -> Answer`. |
| `cli.py` | `rag ingest <paths...>`, `rag ask "<question>"`. |
| `eval.py` | Golden-set runner + LLM-as-judge scoring; prints a report. |

### Key data shapes (Pydantic)

- `Chunk`: `text`, `source` (file path), `chunk_index`.
- `Citation`: `source`, `chunk_index`, `snippet`.
- `Answer`: `text`, `citations: list[Citation]`.

### Retrieval + grounding

- Chunk: ~1000 chars, 150 overlap (sentence-aware split where cheap).
- Retrieve top-k = 4 by cosine similarity.
- Prompt instructs the model to answer **only** from the provided context, cite
  sources inline like `[1]`, and say "I don't know" if the context is
  insufficient (prevents hallucination — a graded eval criterion).

### Evaluation (`eval.py` + `eval/golden.jsonl`)

- Golden set: `{question, must_contain[], notes}` over the sample docs.
- Per question: run the real pipeline, then score two axes with an
  **LLM-as-judge** (Gemini, separate call): **faithfulness** (answer grounded in
  retrieved context, no invention) and **relevance** (answers the question).
  Plus a deterministic **retrieval hit** check (did a must-contain string appear
  in retrieved chunks?).
- Output: per-question table + aggregate scores. This is the "I can measure my
  system" signal hiring managers screen for.

## Acceptance Criteria

1. `rag ingest examples/` indexes the sample docs into a persistent Chroma store; re-running does not duplicate chunks.
2. `rag ask "<q>"` returns an answer that cites at least one source for an in-corpus question.
3. For a question with no answer in the corpus, the system replies "I don't know" (or equivalent) rather than inventing one.
4. `RagIndex` is importable and usable in ~5 lines without the CLI.
5. `python -m ragchat.eval` runs the golden set and prints per-question + aggregate faithfulness/relevance/retrieval-hit scores.
6. Unit tests (no API key needed) cover chunking (overlap, boundaries) and citation formatting; `pytest` green.
7. `ruff check` clean.

## Testing Plan

| Layer | What | Count |
|-------|------|-------|
| Unit | chunking (overlap, short/long text, empty), citation formatting | +4 |
| Unit | Pydantic model round-trips | +1 |
| Integration (key-gated, manual) | ingest → ask returns cited answer; unknown-question → "I don't know" | 2 |
| Eval | golden set faithfulness/relevance/retrieval-hit | golden run |

## Out of Scope (this version)

- Web UI / deployment → **next milestone** (Streamlit + deploy for the live demo).
- Multi-user, auth, doc-level access control.
- Reranking models, hybrid (keyword+vector) search — note as "future work".
- OCR for scanned PDFs (text-layer PDFs only).

## Rollback

Pure additive greenfield. The Chroma store lives in a gitignored dir; delete it
to reset. No shared state.

## Effort (CC-assisted)

~ ingest+chunk (15m) · embed+store (15m) · rag+cli (20m) · eval (20m) · tests (15m) · README (10m).
