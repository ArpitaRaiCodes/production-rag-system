"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    max_new_tokens: int | None = Field(default=None, ge=16, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    rank: int
    score: float
    text: str
    doc_id: str | None = None
    filename: str | None = None
    page: int | None = None
    chunk_index: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool


class SearchResponse(BaseModel):
    query: str
    results: list[Source]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    uploaded_at: str | None = None
    chunks: int
    characters: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    message: str


class DeleteResponse(BaseModel):
    doc_id: str | None = None
    chunks_removed: int
    message: str


class StatsResponse(BaseModel):
    collection: str
    documents: int
    chunks: int
    embedding_model: str
    llm_model: str
    storage_path: str


class HealthResponse(BaseModel):
    status: str
    llm_loaded: bool
    documents: int
    chunks: int
