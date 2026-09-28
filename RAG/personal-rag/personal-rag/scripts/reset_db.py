"""Wipe the vector database from the command line.

    python scripts/reset_db.py            # asks for confirmation
    python scripts/reset_db.py --yes      # no prompt
    python scripts/reset_db.py --hard     # also delete the Chroma folder itself
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the RAG vector database")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    parser.add_argument(
        "--hard",
        action="store_true",
        help="delete the Chroma directory and uploads from disk",
    )
    args = parser.parse_args()

    if not args.yes:
        reply = input(
            f"Delete every embedding in {settings.chroma_dir}? Type yes to continue: "
        )
        if reply.strip().lower() != "yes":
            print("Nothing was deleted.")
            return

    if args.hard:
        shutil.rmtree(settings.chroma_dir, ignore_errors=True)
        shutil.rmtree(settings.upload_dir, ignore_errors=True)
        settings.ensure_dirs()
        print("Chroma directory and uploads removed from disk.")
        return

    from app import vectorstore

    removed = vectorstore.reset_collection()
    shutil.rmtree(settings.upload_dir, ignore_errors=True)
    settings.ensure_dirs()
    print(f"Removed {removed} chunks. The collection is empty and ready to use.")


if __name__ == "__main__":
    main()
