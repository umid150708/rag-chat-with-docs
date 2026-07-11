"""Central config: models, chunking, paths. One place to tune the system."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Models (Gemini free tier)
GEN_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# Chunking
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 150  # characters shared between neighbouring chunks

# Retrieval
TOP_K = 4  # how many chunks to feed the model per question

# Where the persistent Chroma store lives (gitignored)
STORE_DIR = Path(os.environ.get("RAGSTORE_DIR", ".ragstore"))
COLLECTION = "docs"


def require_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No API key. Set GEMINI_API_KEY (see .env.example).")
    return key
