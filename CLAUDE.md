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
- `app.py` (repo root) is the Streamlit live-demo entry point; deployed on
  Streamlit Community Cloud from `requirements.txt` (regenerate it with
  `uv export --no-dev --no-hashes --no-annotate -o requirements.txt`
  whenever deps change).
- Never commit `.env`, `.ragstore/`, or `.streamlit/secrets.toml` (all gitignored).
- API generation calls go through `_gemini.generate` for 429 backoff.
