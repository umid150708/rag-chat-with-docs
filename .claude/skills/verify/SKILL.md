---
name: verify
description: Build/launch/drive recipe for verifying ragchat changes end-to-end (CLI + Streamlit UI).
---

# Verifying ragchat

`uv` is at `~/.local/bin/uv` (not on PATH in agent shells): `export PATH="$HOME/.local/bin:$PATH"`.

## CLI surface (library + env-key path)

```bash
uv run rag ingest examples/     # expect: "Indexed 2 chunks."
uv run rag ask "How many days do I have to return an unopened bag?"   # expect cited answer [1] handbook.md
uv run rag ask "What is the capital of France?"                       # expect "I don't know based on the provided documents."
```

Needs `GEMINI_API_KEY` (repo `.env` has one; config.py loads dotenv).

## Streamlit surface

```bash
uv run streamlit run app.py --server.port 8501 --server.headless true
```

(Or via the Browser pane: launch config `ragchat-streamlit` — may live in another
project's `.claude/launch.json` since sessions run elsewhere.)

Drive in a browser:
- Load: sidebar lists `handbook.md`, `roasting.md`; startup indexing spinner completes (host key from env/secrets).
- No key + question → friendly "paste a free Gemini key" message, no crash.
- Fake key (`not-a-real-key`) + question → "That key was rejected" message; history preserved.
- Real-key cited-answer path in the UI requires a human to paste a key (agents must not enter API keys into fields); the same pipeline is covered by `rag ask`.

## Gotchas

- Gemini free tier is ~5 req/min AND 20 generations/day per key. Per-minute 429s sit in `_gemini.py` backoff for 30–60s (not a hang); once the DAILY cap is hit, every retry 429s and the CLI exits 1 with a raw traceback — no amount of waiting helps until midnight Pacific. Budget test questions accordingly.
- Streamlit's chat input sometimes ignores Enter under automation — click the "Send message" button instead.
- Local Streamlit run shares `.ragstore/` with the CLI (stable chunk ids upsert, so no dupes).
