# Spec — Streamlit chat UI + live deployment

Status: **planned** (built next session). Planned with `/spec`. Decisions locked
via AskUserQuestion on 2026-07-11.

## Context

The RAG library + CLI work, but a recruiter can't *click* a CLI. The single
highest-value addition to this portfolio piece is a **public, clickable demo
link** — recruiters engage far more with a live demo than with code. This
milestone wraps the existing `RagIndex` in a Streamlit chat UI and deploys it
free.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Deploy target | **Streamlit Community Cloud** | Free, native for Streamlit, deploys from GitHub, built-in secrets. |
| Public quota | **Bring-your-own-key (BYO)** | Gemini free tier is 5/min, 20/day per key. A public link on our key dies after ~20 questions/day. Visitor pastes their own free key for queries → scales to any traffic, protects our quota. |
| Vector store persistence | **Ingest at startup, cached** | Cloud filesystem is ephemeral. On boot, embed the bundled `examples/` corpus once into Chroma, cached with `st.cache_resource` so it survives across user sessions in a container. No binary blob in git. |

## Current state (verified 2026-07-11)

- `RagIndex` ([src/ragchat/rag.py](src/ragchat/rag.py)): `ingest()`, `retrieve()`, `query()`, `count()`. Takes an optional `store` and `client`.
- **Blocker for BYO-key:** [src/ragchat/embed.py](src/ragchat/embed.py) builds a **module-global** client from `GEMINI_API_KEY` (env) via `_get_client()`. Embedding is therefore NOT injectable — a visitor's runtime key can't reach it. This must change (see Proposed Change).
- `VectorStore` ([src/ragchat/store.py](src/ragchat/store.py)) is **client-agnostic** (pure Chroma) — good, the store can be shared across clients.
- `_gemini.generate` already has 429 backoff.

## Proposed change

### Two keys, two roles (the core design)

- **Host key** (in Streamlit secrets): used **only once per container** to embed the fixed `examples/` corpus at startup. Bounded (~a handful of embed calls), cached — negligible quota use.
- **Visitor key** (pasted in the UI sidebar): used for **their** query embedding + generation. Unbounded work is paid by the visitor.

This works because `VectorStore` is client-agnostic: index the corpus once with the host client, then answer queries with a `RagIndex` bound to the visitor client but pointing at the **same cached store**.

### Refactor: make the Gemini client injectable

1. **`embed.py`** — drop the module-global client. Change signatures to take an explicit client:
   - `embed_documents(texts: list[str], client: genai.Client) -> list[list[float]]`
   - `embed_query(text: str, client: genai.Client) -> list[float]`
2. **`rag.py`** — `RagIndex` always holds a `genai.Client` (built from a key if not passed). `ingest()` calls `embed_documents(..., self._client)`; `retrieve()` calls `embed_query(..., self._client)`; generation already uses `self._client`. Embedding and generation share the one client.
3. Existing CLI (`cli.py`) still works: `RagIndex()` with no args builds the client from env, exactly as today.

### New: Streamlit app (`app.py` at repo root)

Streamlit Cloud runs a single entry file. Structure:

```python
import streamlit as st
from google import genai
from ragchat.rag import RagIndex
from ragchat.store import VectorStore

@st.cache_resource                     # once per container
def build_store():
    host_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    store = VectorStore()              # tmp/ephemeral is fine — cached in-process
    RagIndex(store=store, client=host_client).ingest(["examples/"])
    return store

store = build_store()

st.title("Chat with the Nimbus Coffee docs")
visitor_key = st.text_input("Your Gemini API key (free: aistudio.google.com/apikey)", type="password")
question = st.chat_input("Ask about refunds, shipping, roasts…")

if question:
    if not visitor_key:
        st.warning("Paste a free Gemini key above to ask.")
    else:
        idx = RagIndex(store=store, client=genai.Client(api_key=visitor_key))
        answer = idx.query(question)
        st.chat_message("assistant").write(answer.format())
```

(Final version adds: chat history via `st.session_state`, showing which sample docs are indexed, a friendly 429 message, and a short "what is this" blurb.)

### Deploy (Streamlit Community Cloud)

1. Add `streamlit` to `pyproject.toml` deps; ensure the repo has a way for Streamlit Cloud to install deps (a `requirements.txt` generated from the lock, since Streamlit Cloud doesn't use `uv`).
2. Push to GitHub (already there).
3. share.streamlit.io → New app → point at `umid150708/rag-chat-with-docs`, `app.py`, branch `main`.
4. Add `GEMINI_API_KEY` in the app's **Secrets** (host key, corpus-indexing only).
5. Get the public URL → add it to the README top ("🔗 Live demo").

## Acceptance criteria

1. `embed_documents` / `embed_query` take an explicit client; no module-global remains; `mypy` + `vulture` + `pytest` still clean.
2. Existing CLI (`rag ingest` / `rag ask`) works unchanged (env-key path).
3. `streamlit run app.py` locally: page loads, corpus auto-indexes, a question with a valid key returns a cited answer, an out-of-corpus question declines.
4. No key pasted → the app prompts for one, never crashes.
5. Deployed app reachable at a public URL; the host key is only in Streamlit secrets (never in git — `.streamlit/secrets.toml` gitignored).
6. README top has a working "Live demo" link.

## Testing plan

| Layer | What | Count |
|---|---|---|
| Unit | `embed_*` accept an injected client (mock/stub, no network) | +2 |
| Unit | existing chunking/model tests still green | (7 current) |
| Manual (key-gated) | local `streamlit run`: indexed → cited answer; unknown → "I don't know"; no-key → prompt | 3 |
| Post-deploy | `/qa` the live URL (loads, ask flow, error states); `/canary` after deploy | — |

## Skills to use next session (in order)

1. `git checkout -b feat/streamlit-ui` (branch first — so /review + /ship work).
2. Build per this spec.
3. `/verify` — drive the local app end-to-end.
4. `/qa` — exercise the deployed UI in a browser.
5. `/design-review` — visual polish pass.
6. `/review` then `/ship` — land it.
7. `/land-and-deploy` / `/canary` around the deploy if useful.

## Out of scope

- User document upload (fixed sample corpus only for the demo).
- Auth, multi-tenant, analytics.
- Streaming token-by-token output (nice-to-have, later).

## Rollback

Additive. `app.py` is new; the `embed.py` refactor is behind unchanged public
call sites (CLI unaffected). If deploy misbehaves, unpublish the Streamlit app —
the library/CLI are untouched. Revert the PR to undo the refactor.

## Effort (CC-assisted)

~ embed/rag refactor + tests (20m) · `app.py` (20m) · local verify (10m) ·
requirements.txt + deploy config (10m) · deploy + secrets + README link (15m) ·
/qa + /design-review polish (20m).
