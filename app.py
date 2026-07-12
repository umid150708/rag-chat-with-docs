"""Streamlit chat UI for ragchat — the live-demo entry point.

The server needs NO API key of its own: the bundled corpus ships with
precomputed embeddings (data/corpus_embeddings.json, regenerate with
scripts/build_corpus_snapshot.py), loaded straight into the vector store at
boot. The VISITOR's key (pasted in the sidebar) pays for their own question
embedding + generation, so the public link scales without any host quota.

This works because VectorStore is client-agnostic: vectors go in from a file,
queries come from a RagIndex bound to the visitor's client. If the snapshot is
missing (e.g. local dev with a changed corpus), we fall back to live-embedding
with a host key from Streamlit secrets or the environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from google import genai
from google.genai.errors import APIError

from ragchat.models import Chunk
from ragchat.rag import RagIndex
from ragchat.store import VectorStore

CORPUS_DIR = "examples"
SNAPSHOT_PATH = Path("data/corpus_embeddings.json")

st.set_page_config(page_title="Chat with the Nimbus Coffee docs", page_icon="📚")


def _host_api_key() -> str | None:
    """Host key: Streamlit secrets on Cloud, .env/env when running locally."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except FileNotFoundError:  # no secrets.toml — fine, fall back to env
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


@st.cache_resource(show_spinner="Loading the sample docs (one-time per server)…")
def build_store() -> VectorStore:
    """Load the precomputed corpus snapshot; fall back to live embedding."""
    store = VectorStore()  # ephemeral path is fine — cached in-process
    if SNAPSHOT_PATH.exists():
        records = json.loads(SNAPSHOT_PATH.read_text())["records"]
        store.add(
            [
                Chunk(text=r["text"], source=r["source"], chunk_index=r["chunk_index"])
                for r in records
            ],
            [r["embedding"] for r in records],
        )
    else:
        # Snapshot missing (local dev with a changed corpus): embed live.
        host_key = _host_api_key()
        if not host_key:
            st.error(
                "No corpus snapshot (data/corpus_embeddings.json) and no "
                "GEMINI_API_KEY in Streamlit secrets or environment — run "
                "scripts/build_corpus_snapshot.py or set a key."
            )
            st.stop()
        try:
            RagIndex(store=store, client=genai.Client(api_key=host_key)).ingest(
                [CORPUS_DIR]
            )
        except APIError as err:
            st.error(
                f"Couldn't index the sample docs (Gemini error {err.code}). Reload to retry."
            )
            st.stop()
    if store.count() == 0:
        st.error(f"No documents found under `{CORPUS_DIR}/` — nothing to chat with.")
        st.stop()
    return store


store = build_store()

# ---- sidebar: what this is + the visitor's key ------------------------------
with st.sidebar:
    st.header("What is this?")
    st.markdown(
        "A **RAG (retrieval-augmented generation)** demo: ask questions about "
        "the docs of *Nimbus Coffee*, a fictional company, and get answers "
        '**with citations** — or an honest *"I don\'t know"* when the docs '
        "don't cover it.\n\n"
        "[Source code & eval harness →](https://github.com/umid150708/rag-chat-with-docs)"
    )

    st.divider()
    visitor_key = st.text_input(
        "Your Gemini API key",
        type="password",
        help="Your key is used only for your own questions and never stored.",
    ).strip()
    st.caption(
        "Questions run on **your own free key** so the demo never rate-limits. "
        "Grab one in ~30 seconds at [aistudio.google.com/apikey]"
        "(https://aistudio.google.com/apikey) — no card required."
    )

    st.divider()
    st.subheader("Indexed sample docs")
    for doc in sorted(Path(CORPUS_DIR).glob("*")):
        st.markdown(f"- `{doc.name}`")

# ---- chat --------------------------------------------------------------------
st.title("📚 Chat with the Nimbus Coffee docs")
st.caption(
    "Try: *How many days do I have to return an unopened bag?* · "
    "*Do you ship internationally?* · *Which roast has blueberry notes?*"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about refunds, shipping, roasts…")

if question:
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    if not visitor_key:
        answer_md = (
            "Almost there — paste a **free Gemini API key** in the sidebar to ask. "
            "Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
        )
    else:
        idx = RagIndex(store=store, client=genai.Client(api_key=visitor_key))
        try:
            with st.spinner("Retrieving and answering…"):
                answer = idx.query(question)
            answer_md = answer.text
            if answer.citations:
                answer_md += "\n\n**Sources**\n" + "\n".join(
                    f"- `{c.source}` (chunk {c.chunk_index}): *{c.snippet}…*"
                    for c in answer.citations
                )
        except APIError as err:
            if err.code == 429:
                answer_md = (
                    "Your key hit a Gemini free-tier limit — either a few "
                    "requests/minute (wait a minute and retry) or the daily cap "
                    "(resets at midnight Pacific)."
                )
            elif err.code in (400, 401, 403):
                answer_md = (
                    "That key was rejected by Gemini — double-check it, or mint "
                    "a fresh free one at "
                    "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
                )
            else:
                answer_md = f"Gemini returned an unexpected error ({err.code}). Please try again."

    with st.chat_message("assistant"):
        st.markdown(answer_md)
    st.session_state.messages.append({"role": "assistant", "content": answer_md})
