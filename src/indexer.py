import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss


class VectorIndexer:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(chunks) == 0:
            return
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")

        # Ensure float32 C-contiguous
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vecs)
        self.metadata.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        q_vec = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                res = dict(self.metadata[idx])
                res["similarity_score"] = round(float(score), 4)
                results.append(res)
        return results

    def save(self, index_path: str, metadata_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, index_path: str, metadata_path: str):
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file does not exist.")

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.dimension = self.index.d
