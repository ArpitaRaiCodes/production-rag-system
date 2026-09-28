"""Central configuration. Every value can be overridden with an environment
variable or an entry in a local .env file (see .env.example)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Runtime settings for the whole application."""

    # ---- server -----------------------------------------------------------
    host: str = _env("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)
    cors_origins: list[str] = [
        o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    api_key: str | None = os.getenv("API_KEY") or None

    # ---- storage ----------------------------------------------------------
    data_dir: Path = Path(_env("DATA_DIR", str(BASE_DIR / "data")))
    chroma_dir: Path = Path(_env("CHROMA_DIR", str(BASE_DIR / "data" / "chroma")))
    upload_dir: Path = Path(_env("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
    collection_name: str = _env("COLLECTION_NAME", "knowledge_base")

    # ---- ingestion --------------------------------------------------------
    chunk_size: int = _env_int("CHUNK_SIZE", 900)          # characters
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 150)    # characters
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 50)
    keep_original_files: bool = _env_bool("KEEP_ORIGINAL_FILES", True)

    # ---- embeddings -------------------------------------------------------
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 32)
    # bge models want this prefix on the *query* side only
    query_prefix: str = _env(
        "QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
    )

    # ---- generation -------------------------------------------------------
    llm_model: str = _env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    device: str = _env("DEVICE", "auto")          # auto | cuda | mps | cpu
    dtype: str = _env("DTYPE", "auto")            # auto | float16 | bfloat16 | float32
    max_new_tokens: int = _env_int("MAX_NEW_TOKENS", 512)
    temperature: float = _env_float("TEMPERATURE", 0.3)
    top_p: float = _env_float("TOP_P", 0.9)
    load_llm_on_startup: bool = _env_bool("LOAD_LLM_ON_STARTUP", False)
    hf_home: str | None = os.getenv("HF_HOME") or None

    # ---- retrieval --------------------------------------------------------
    top_k: int = _env_int("TOP_K", 3)
    min_score: float = _env_float("MIN_SCORE", 0.15)  # cosine similarity floor

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    if settings.hf_home:
        os.environ.setdefault("HF_HOME", settings.hf_home)
    return settings


settings = get_settings()
