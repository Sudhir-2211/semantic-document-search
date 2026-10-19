import os
import pytest
import numpy as np
from src.indexer import VectorIndexer


def test_indexer_add_search_and_persistence(tmp_path):
    dim = 384
    indexer = VectorIndexer(dimension=dim)

    # Create dummy embeddings
    v1 = np.zeros((1, dim), dtype=np.float32)
    v1[0, 0] = 1.0  # Points along dim 0

    v2 = np.zeros((1, dim), dtype=np.float32)
    v2[0, 1] = 1.0  # Points along dim 1

    chunks = [
        {"chunk_id": "c1", "source": "doc1", "text": "Content about microservices"},
        {"chunk_id": "c2", "source": "doc2", "text": "Content about databases"},
    ]
    stacked_vecs = np.vstack([v1, v2])

    indexer.add_chunks(chunks, stacked_vecs)
    assert indexer.index.ntotal == 2

    # Query matching v1 exactly
    results = indexer.search(v1[0], top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert pytest.approx(results[0]["similarity_score"], abs=1e-4) == 1.0
    assert pytest.approx(results[1]["similarity_score"], abs=1e-4) == 0.0

    # Test Save & Load
    idx_file = str(tmp_path / "test.index")
    meta_file = str(tmp_path / "test.json")
    indexer.save(idx_file, meta_file)

    loaded_indexer = VectorIndexer(dimension=dim)
    loaded_indexer.load(idx_file, meta_file)
    assert loaded_indexer.index.ntotal == 2
    assert len(loaded_indexer.metadata) == 2
