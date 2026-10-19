import pytest
import numpy as np
from src.embedder import DenseEmbedder


def test_embedder_dimension_and_normalization():
    embedder = DenseEmbedder(dimension=384)
    texts = [
        "Distributed database replication and consensus protocols.",
        "Convolutional neural networks for image classification.",
    ]
    vecs = embedder.embed_texts(texts)

    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32

    # L2 norm of every vector should be approximately 1.0
    norms = np.linalg.norm(vecs, axis=1)
    for n in norms:
        assert pytest.approx(n, abs=1e-4) == 1.0


def test_embed_single_query():
    embedder = DenseEmbedder(dimension=384)
    q_vec = embedder.embed_query("Microservice circuit breaker pattern")
    assert q_vec.shape == (384,)
    assert pytest.approx(np.linalg.norm(q_vec), abs=1e-4) == 1.0
