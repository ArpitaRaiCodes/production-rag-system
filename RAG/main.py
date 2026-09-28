"""FastAPI application: document management + retrieval + chat."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import llm, rag, vectorstore
from app.config import BASE_DIR, settings
from app.ingestion import (
    SUPPORTED_EXTENSIONS,
    EmptyDocument,
    UnsupportedFileType,
    build_chunks,
)
from app.schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentInfo,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    UploadResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
logger = logging.getLogger("rag")

WEB_DIR = BASE_DIR / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    vectorstore.get_collection()  # opens/creates the store at boot
    if settings.load_llm_on_startup:
        logger.info("Preloading the language model")
        llm.load_model()
    yield


app = FastAPI(
    title="Personal RAG",
    description="Upload your documents, ask questions, get cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op unless API_KEY is set in the environment."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )


# ---------------------------------------------------------------------------
# Health and stats
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    data = vectorstore.stats()
    return HealthResponse(
        status="ok",
        llm_loaded=llm.is_loaded(),
        documents=data["documents"],
        chunks=data["chunks"],
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["system"])
def get_stats() -> StatsResponse:
    return StatsResponse(**vectorstore.stats())


@app.post("/api/warmup", tags=["system"], dependencies=[Depends(require_api_key)])
def warmup() -> dict:
    """Load the model now so the first question is not slow."""
    llm.load_model()
    return {"loaded": True, "model": settings.llm_model}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@app.get("/api/documents", response_model=list[DocumentInfo], tags=["documents"])
def get_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in vectorstore.list_documents()]


@app.post(
    "/api/documents",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"{suffix or 'That file type'} is not supported. Use one of: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )

    doc_id = uuid.uuid4().hex[:12]
    stored_path = settings.upload_dir / f"{doc_id}{suffix}"

    size = 0
    limit = settings.max_upload_mb * 1024 * 1024
    with stored_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                stored_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File is larger than the {settings.max_upload_mb} MB limit.",
                )
            out.write(chunk)

    try:
        chunks = build_chunks(stored_path)
        count = vectorstore.add_document(
            doc_id=doc_id,
            filename=filename,
            chunks=[{"text": c.text, "index": c.index, "page": c.page} for c in chunks],
        )
    except (UnsupportedFileType, EmptyDocument) as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        stored_path.unlink(missing_ok=True)
        logger.exception("Indexing failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    if not settings.keep_original_files:
        stored_path.unlink(missing_ok=True)

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunks=count,
        message=f"Indexed {filename} into {count} searchable chunks.",
    )


@app.delete(
    "/api/documents/{doc_id}",
    response_model=DeleteResponse,
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)
def delete_document(doc_id: str) -> DeleteResponse:
    removed = vectorstore.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document with id {doc_id}")

    for leftover in settings.upload_dir.glob(f"{doc_id}.*"):
        leftover.unlink(missing_ok=True)

    return DeleteResponse(
        doc_id=doc_id,
        chunks_removed=removed,
        message=f"Removed {removed} chunks. That document is no longer searchable.",
    )


@app.delete(
    "/api/documents",
    response_model=DeleteResponse,
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)
def clear_all(confirm: bool = False) -> DeleteResponse:
    """Wipe the vector database. Requires ?confirm=true."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to delete every document in the vector database.",
        )
    removed = vectorstore.reset_collection()
    if settings.upload_dir.exists():
        shutil.rmtree(settings.upload_dir, ignore_errors=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return DeleteResponse(
        doc_id=None,
        chunks_removed=removed,
        message=f"Vector database cleared. {removed} chunks removed.",
    )


# ---------------------------------------------------------------------------
# Retrieval and chat
# ---------------------------------------------------------------------------
@app.post("/api/search", response_model=SearchResponse, tags=["chat"])
def search(payload: SearchRequest) -> SearchResponse:
    """Top-k passages, no generation. Useful for debugging retrieval."""
    hits = vectorstore.search(payload.query, top_k=payload.top_k)
    return SearchResponse(query=payload.query, results=hits)


@app.post("/api/ask", response_model=AskResponse, tags=["chat"])
def ask(payload: AskRequest) -> AskResponse:
    result = rag.answer(
        payload.question,
        top_k=payload.top_k,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
    )
    return AskResponse(**result)


@app.post("/api/ask/stream", tags=["chat"])
def ask_stream(payload: AskRequest) -> StreamingResponse:
    """Server-sent events: one `sources` event, then `token` events, then `done`."""

    def event_stream():
        try:
            sources, tokens = rag.stream_answer(
                payload.question,
                top_k=payload.top_k,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
            )
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            for piece in tokens:
                yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming failed")
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Web interface
# ---------------------------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
