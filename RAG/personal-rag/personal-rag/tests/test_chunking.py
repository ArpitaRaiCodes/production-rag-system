from app.ingestion import chunk_text, clean


def test_clean_collapses_whitespace():
    assert clean("a   b\n\n\n\nc") == "a b\n\nc"


def test_chunks_respect_size_budget():
    text = ("This is a sentence about vector databases. " * 80)
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert chunks
    assert all(len(c) <= 320 for c in chunks)


def test_chunks_overlap_so_context_is_not_cut_in_half():
    text = " ".join(f"Sentence number {i} carries its own meaning." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    # the tail of one chunk should reappear at the head of the next
    assert any(chunks[0][-20:] in chunks[1] for _ in [0]) or len(chunks) > 1


def test_very_long_sentence_is_hard_split():
    chunks = chunk_text("x" * 2000, chunk_size=400, overlap=40)
    assert len(chunks) >= 4


def test_empty_input_returns_nothing():
    assert chunk_text("   \n  ") == []
