"""Index files or a whole folder without using the web interface.

    python scripts/ingest.py report.pdf notes.md
    python scripts/ingest.py ./my-documents
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import vectorstore  # noqa: E402
from app.ingestion import SUPPORTED_EXTENSIONS, build_chunks  # noqa: E402


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files += [
                p for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        elif path.is_file():
            files.append(path)
        else:
            print(f"skipped, not found: {raw}")
    return files


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    files = collect(sys.argv[1:])
    if not files:
        print("No supported files found.")
        raise SystemExit(1)

    total = 0
    for path in files:
        try:
            chunks = build_chunks(path)
            doc_id = uuid.uuid4().hex[:12]
            count = vectorstore.add_document(
                doc_id=doc_id,
                filename=path.name,
                chunks=[
                    {"text": c.text, "index": c.index, "page": c.page} for c in chunks
                ],
            )
            total += count
            print(f"  {path.name}: {count} chunks  (doc_id {doc_id})")
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.name}: failed, {exc}")

    print(f"\nIndexed {len(files)} files, {total} chunks in total.")


if __name__ == "__main__":
    main()
