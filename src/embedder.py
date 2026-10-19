import numpy as np
from typing import List, Union, Optional
from sklearn.feature_extraction.text import HashingVectorizer


class DenseEmbedder:
    """
    High-performance vector embedder producing L2-normalized 384-dimensional dense vectors.
    Supports SentenceTransformers when available, with a fast zero-network Hashing vectorizer fallback.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384, use_transformer: Optional[bool] = None):
        import os
        self.dimension = dimension
        self.model_name = model_name
        self._st_model = None
        self._hashing_vectorizer = HashingVectorizer(n_features=dimension, norm="l2", alternate_sign=False)

        if use_transformer is None:
            use_transformer = os.getenv("USE_TRANSFORMERS", "false").lower() in ("true", "1")

        if use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model_name)
            except Exception:
                self._st_model = None

    @property
    def is_transformer(self) -> bool:
        return self._st_model is not None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self._st_model is not None:
            try:
                embeddings = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.array(embeddings, dtype=np.float32)
            except Exception:
                pass

        # Fallback dense normalized projection
        sparse_vecs = self._hashing_vectorizer.transform(texts)
        dense_vecs = sparse_vecs.toarray().astype(np.float32)

        # Ensure L2 normalization
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense_vecs / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
