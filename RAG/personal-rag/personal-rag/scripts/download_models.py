"""Pre-download the Qwen model and the embedding model.

Run this once before your first launch so that the first question is not
waiting on a multi-gigabyte download:

    python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def main() -> None:
    from huggingface_hub import snapshot_download

    for repo in (settings.embedding_model, settings.llm_model):
        print(f"\nDownloading {repo} ...")
        path = snapshot_download(
            repo_id=repo,
            allow_patterns=[
                "*.json", "*.txt", "*.model", "*.safetensors",
                "*.bin", "*.py", "*.md",
            ],
        )
        print(f"  cached at {path}")

    print("\nBoth models are cached. Start the server with: uvicorn app.main:app")


if __name__ == "__main__":
    main()
