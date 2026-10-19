from typing import List, Dict, Any


class TextChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source_name: str = "doc") -> List[Dict[str, Any]]:
        words = text.strip().split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            window_words = words[i:i + self.chunk_size]
            chunk_str = " ".join(window_words)
            chunks.append({
                "chunk_id": f"{source_name}_chunk_{chunk_idx}",
                "source": source_name,
                "text": chunk_str,
                "word_count": len(window_words),
            })
            chunk_idx += 1
            if i + self.chunk_size >= len(words):
                break

        return chunks
