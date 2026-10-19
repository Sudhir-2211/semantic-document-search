import pytest
from src.search_engine import SemanticSearchEngine


def test_search_engine_end_to_end():
    engine = SemanticSearchEngine()
    docs = [
        {
            "title": "circuit_breaker.txt",
            "content": "The Circuit Breaker pattern prevents cascading failures across distributed microservices by failing fast when dependencies degrade.",
        },
        {
            "title": "vector_search.txt",
            "content": "FAISS provides fast dense vector nearest neighbor indexing using IndexFlatIP for cosine similarity ranking.",
        },
    ]

    total_chunks = engine.index_documents(docs)
    assert total_chunks == 2
    assert engine.indexer.index.ntotal == 2

    # Query for circuit breaker
    results = engine.search("How does circuit breaker protect microservices?", top_k=2)
    assert len(results) >= 1
    assert results[0]["source"] == "circuit_breaker.txt"
    assert results[0]["similarity_score"] > 0.0

    # Query with empty string
    empty_res = engine.search("")
    assert empty_res == []
