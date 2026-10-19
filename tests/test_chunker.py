import pytest
from src.chunker import TextChunker


def test_chunk_text_sliding_window():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    sample_text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    chunks = chunker.chunk_text(sample_text, source_name="test_doc")

    assert len(chunks) == 2
    assert chunks[0]["source"] == "test_doc"
    assert chunks[0]["chunk_id"] == "test_doc_chunk_0"
    assert len(chunks[0]["text"].split()) == 10
    assert "one" in chunks[0]["text"]
    assert "fifteen" in chunks[1]["text"]


def test_chunk_empty_text():
    chunker = TextChunker()
    chunks = chunker.chunk_text("", source_name="empty")
    assert chunks == []
