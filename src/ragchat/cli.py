"""CLI:  rag ingest <paths...>   |   rag ask "<question>" """

from __future__ import annotations

import sys

from .rag import RagIndex


def main() -> None:
    args = sys.argv[1:]
    if not args:
        _usage()
        sys.exit(1)

    command, rest = args[0], args[1:]
    idx = RagIndex()

    if command == "ingest":
        if not rest:
            print("Usage: rag ingest <path> [<path> ...]", file=sys.stderr)
            sys.exit(1)
        n = idx.ingest(rest)
        print(f"Indexed {n} chunks. Store now holds {idx.count()} chunks total.")

    elif command == "ask":
        if not rest:
            print('Usage: rag ask "<question>"', file=sys.stderr)
            sys.exit(1)
        question = " ".join(rest)
        print(idx.query(question).format())

    else:
        _usage()
        sys.exit(1)


def _usage() -> None:
    print('Usage:\n  rag ingest <paths...>\n  rag ask "<question>"', file=sys.stderr)


if __name__ == "__main__":
    main()
