"""Load documents and split them into overlapping chunks.

Pure functions, no API calls — so this is the easy part to unit-test.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .models import Chunk

SUPPORTED = {".txt", ".md", ".pdf"}


def load_text(path: Path) -> str:
    """Extract plain text from a supported file (text-layer PDFs only)."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into ~`size`-char windows that overlap by `overlap` chars.

    Overlap keeps ideas that straddle a boundary retrievable from either side.
    We prefer to break on a paragraph/sentence boundary near the window end so
    chunks read cleanly, falling back to a hard cut if none is close.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        window = text[start:end]
        if end < len(text):
            # try to end on a natural boundary within the last 20% of the window
            pivot = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if pivot > size * 0.8:
                end = start + pivot + 1
                window = text[start:end]
        chunk = window.strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def discover_files(paths: list[str]) -> list[Path]:
    """Expand a mix of files and directories into a flat list of supported files."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(
                f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED
            )
        elif p.suffix.lower() in SUPPORTED:
            found.append(p)
    return found


def chunks_for_file(path: Path) -> list[Chunk]:
    """Load one file and return its chunks with source metadata."""
    text = load_text(path)
    return [
        Chunk(text=c, source=str(path), chunk_index=i)
        for i, c in enumerate(chunk_text(text))
    ]
