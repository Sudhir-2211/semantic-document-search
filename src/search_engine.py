import os
from typing import List, Dict, Any, Optional
from src.chunker import TextChunker
from src.embedder import DenseEmbedder
from src.indexer import VectorIndexer


class SemanticSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        self.embedder = DenseEmbedder(model_name=model_name, dimension=dimension)
        self.indexer = VectorIndexer(dimension=dimension)

    def index_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Accepts list of dicts: [{'title': '...', 'content': '...'}]
        Chunks, embeds, and updates FAISS index.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_text(doc["content"], source_name=doc.get("title", "untitled"))
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.indexer.add_chunks(all_chunks, embeddings)
        return len(all_chunks)

    def index_directory(self, dir_path: str) -> int:
        docs = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".txt", ".md")):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        docs.append({"title": file, "content": f.read()})
        return self.index_documents(docs)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        q_vec = self.embedder.embed_query(query)
        raw_results = self.indexer.search(q_vec, top_k=top_k)

        filtered = [r for r in raw_results if r["similarity_score"] >= min_score]
        return filtered

    def save(self, index_file: str, metadata_file: str):
        self.indexer.save(index_file, metadata_file)

    def load(self, index_file: str, metadata_file: str):
        self.indexer.load(index_file, metadata_file)
