# ragchat — project notes for AI assistants

Small, production-literate RAG system (chat with your docs) on Google Gemini.
See [SPEC.md](SPEC.md) for the design and [README.md](README.md) for usage.

## Health Stack

- lint: ruff check .
- format: ruff format --check .
- typecheck: mypy src/ragchat
- test: pytest
- deadcode: vulture

## Conventions

- Python 3.11+, managed with `uv`. Run tools via `uv run <tool>`.
- One module talks to each external system: `embed.py` (Gemini embeddings),
  `store.py` (Chroma), `rag.py` (Gemini generation via `_gemini.generate`).
- Never commit `.env` or `.ragstore/` (both gitignored).
- API generation calls go through `_gemini.generate` for 429 backoff.
