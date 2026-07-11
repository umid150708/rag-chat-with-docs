"""The small data shapes that flow through the system."""

from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    source: str  # file path the chunk came from
    chunk_index: int  # position within that file


class Citation(BaseModel):
    source: str
    chunk_index: int
    snippet: str  # short quote from the cited chunk


class Answer(BaseModel):
    text: str
    citations: list[Citation]

    def format(self) -> str:
        """Human-readable answer with a Sources list."""
        lines = [self.text.strip()]
        if self.citations:
            lines.append("\nSources:")
            for i, c in enumerate(self.citations, 1):
                lines.append(f"  [{i}] {c.source} (chunk {c.chunk_index})")
        return "\n".join(lines)
