"""Test fixtures.

The suite runs without downloading any model: the embedding functions are
replaced with a deterministic hashing vectoriser, which is enough to exercise
storage, retrieval ordering and the HTTP layer.
"""

from __future__ import annotations

import math
import re
import sys
from hashlib import blake2b
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DIM = 256


def fake_vector(text: str) -> list[float]:
    vector = [0.0] * DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        slot = int(blake2b(token.encode(), digest_size=4).hexdigest(), 16) % DIM
        vector[slot] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture(scope="session", autouse=True)
def stub_embeddings():
    from app import vectorstore

    vectorstore.embed_documents = lambda texts: [fake_vector(t) for t in texts]
    vectorstore.embed_query = fake_vector
    yield
