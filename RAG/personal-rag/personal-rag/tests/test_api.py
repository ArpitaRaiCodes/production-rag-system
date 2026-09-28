"""End-to-end tests for the HTTP layer, with the models stubbed out.

    pytest -q
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

POLICY = (
    "The refund window is thirty days from delivery. "
    "Customers must include the original packaging with every return. "
    "Refunds are issued to the original payment method within five business days. "
    "Shipping charges are not refundable under any circumstances."
).encode()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app.config import settings

    root = tmp_path_factory.mktemp("rag")
    settings.chroma_dir = root / "chroma"
    settings.upload_dir = root / "uploads"
    settings.data_dir = root
    settings.collection_name = "test_kb"
    settings.ensure_dirs()

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def upload(client, name: str, body: bytes):
    return client.post(
        "/api/documents", files={"file": (name, io.BytesIO(body), "text/plain")}
    )


def test_health_reports_ok(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_loaded"] is False


def test_upload_then_search_then_delete(client):
    created = upload(client, "policy.txt", POLICY)
    assert created.status_code == 201
    doc_id = created.json()["doc_id"]
    assert created.json()["chunks"] >= 1

    listed = client.get("/api/documents").json()
    assert any(d["doc_id"] == doc_id for d in listed)

    found = client.post("/api/search", json={"query": "refund window", "top_k": 3})
    assert found.status_code == 200
    results = found.json()["results"]
    assert results and results[0]["filename"] == "policy.txt"
    assert len(results) <= 3

    removed = client.delete(f"/api/documents/{doc_id}")
    assert removed.status_code == 200
    assert removed.json()["chunks_removed"] >= 1

    assert client.get("/api/documents").json() == []
    assert client.post("/api/search", json={"query": "refund"}).json()["results"] == []


def test_deleting_one_file_leaves_the_others(client):
    a = upload(client, "a.txt", POLICY).json()["doc_id"]
    b = upload(client, "b.txt", b"Warranty claims are handled by the service centre.")
    b_id = b.json()["doc_id"]

    client.delete(f"/api/documents/{a}")
    remaining = client.get("/api/documents").json()
    assert [d["doc_id"] for d in remaining] == [b_id]

    client.delete("/api/documents?confirm=true")


def test_deleting_an_unknown_document_is_a_404(client):
    assert client.delete("/api/documents/doesnotexist").status_code == 404


def test_unsupported_file_type_is_rejected(client):
    res = client.post(
        "/api/documents",
        files={"file": ("payload.exe", io.BytesIO(b"nope"), "application/octet-stream")},
    )
    assert res.status_code == 415


def test_clear_all_requires_confirmation(client):
    upload(client, "policy.txt", POLICY)
    assert client.delete("/api/documents").status_code == 400
    cleared = client.delete("/api/documents?confirm=true")
    assert cleared.status_code == 200
    assert client.get("/api/stats").json()["chunks"] == 0


def test_streaming_answer_emits_sources_then_tokens(client, monkeypatch):
    from app import llm

    monkeypatch.setattr(llm, "stream", lambda *a, **k: iter(["Thirty ", "days [1]."]))
    upload(client, "policy.txt", POLICY)

    with client.stream(
        "POST", "/api/ask/stream", json={"question": "refund window", "top_k": 3}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: sources" in body
    assert body.count("event: token") == 2
    assert body.rstrip().endswith("event: done\ndata: {}")
    client.delete("/api/documents?confirm=true")
