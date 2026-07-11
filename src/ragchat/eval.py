"""Evaluation harness — measure whether the RAG system actually works.

Three signals per question:
  - retrieval_hit  : did the retrieved chunks contain the expected fact? (deterministic)
  - faithfulness   : is the answer grounded in the retrieved context, no invention? (LLM judge, 1-5)
  - relevance      : does the answer actually address the question? (LLM judge, 1-5)

"Unknown" questions (fact not in the corpus) pass when the system correctly
declines instead of hallucinating.

Run:  python -m ragchat.eval
"""

from __future__ import annotations

import json
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

from . import _gemini
from .config import GEN_MODEL, require_api_key
from .rag import RagIndex

GOLDEN = Path(__file__).resolve().parents[2] / "eval" / "golden.jsonl"

JUDGE_SYSTEM = (
    "You are a strict evaluator of a retrieval-augmented answer. "
    "Given a QUESTION, the CONTEXT the system retrieved, and its ANSWER, score:\n"
    "- faithfulness (1-5): 5 = every claim is supported by the context, "
    "1 = the answer invents facts not in the context.\n"
    "- relevance (1-5): 5 = directly answers the question, 1 = off-topic.\n"
    "A correct 'I don't know' when the context lacks the answer is faithful (5)."
)


class JudgeScore(BaseModel):
    faithfulness: int
    relevance: int
    reason: str


def _judge(
    client: genai.Client, question: str, context: str, answer: str
) -> JudgeScore:
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    resp = _gemini.generate(
        client,
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM,
            response_mime_type="application/json",
            response_schema=JudgeScore,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return JudgeScore.model_validate_json(resp.text)


def run() -> None:
    cases = [
        json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()
    ]
    idx = RagIndex()
    if idx.count() == 0:
        raise SystemExit("Store is empty. Run `rag ingest examples/` first.")

    client = genai.Client(api_key=require_api_key())

    rows = []
    for case in cases:
        q = case["question"]
        retrieved = idx.retrieve(q)
        context = "\n\n".join(c.text for c in retrieved)
        answer = idx.query(q)
        answer_text = answer.text

        must = case.get("must_contain", [])
        hit = all(m.lower() in context.lower() for m in must) if must else True
        is_unknown = "i don't know" in answer_text.lower()

        if case.get("expect_unknown"):
            # Correct behaviour is to decline; score locally, skip the judge.
            score = JudgeScore(
                faithfulness=5 if is_unknown else 1,
                relevance=5 if is_unknown else 1,
                reason="declined as expected" if is_unknown else "should have declined",
            )
        else:
            score = _judge(client, q, context, answer_text)

        rows.append((q, hit, score))

    _report(rows)


def _report(rows) -> None:
    print(f"\nEVALUATION — {len(rows)} questions\n" + "=" * 60)
    print(f"{'faith':>5} {'rel':>4} {'hit':>4}  question")
    print("-" * 60)
    for q, hit, s in rows:
        print(
            f"{s.faithfulness:>5} {s.relevance:>4} {'yes' if hit else 'NO':>4}  {q[:44]}"
        )
    n = len(rows)
    avg_f = sum(s.faithfulness for _, _, s in rows) / n
    avg_r = sum(s.relevance for _, _, s in rows) / n
    hit_rate = sum(1 for _, hit, _ in rows if hit) / n
    print("-" * 60)
    print(
        f"avg faithfulness {avg_f:.2f}/5   avg relevance {avg_r:.2f}/5   retrieval hit {hit_rate:.0%}"
    )


if __name__ == "__main__":
    run()
