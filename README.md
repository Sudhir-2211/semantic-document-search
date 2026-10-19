# Semantic Document Search

A high-performance dense vector retrieval and semantic document search engine built with **Python**, **Sentence Transformers**, **FAISS (Facebook AI Similarity Search)**, and **Streamlit**.

---

## Features

- **Document Chunking Engine**: Text chunker using sliding-window token spans with customizable overlap to preserve semantic context across paragraph boundaries.
- **Dense Vector Embedding**:
  - Computes 384-dimensional dense vectors with strict L2-normalization.
  - Seamlessly utilizes HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) with a zero-network dense hash fallback for local offline testing.
- **FAISS Vector Indexing**:
  - Leverages `IndexFlatIP` (Inner Product) over L2-normalized vectors to compute exact cosine similarity without distance distortion.
  - Disk serialization: saves and reloads FAISS binary indices (`.index`) alongside structured JSON chunk metadata.
- **Dual Query Interfaces**:
  - **Interactive Streamlit Web Dashboard**: Query input with live similarity badges, dynamic Top-K and threshold sliders, and sample search prompts.
  - **Command-Line Interface (CLI)**: Fast headless indexing and command-line search.
- **Synthetic Technical Corpus**: Pre-populated with technical articles covering microservices, FAISS indexing, OAuth2.0 security, ML model monitoring, and Redis caching.

---

## Project Structure

```text
semantic-document-search/
├── data/
│   └── documents/            # Sample technical documentation corpus
├── src/
│   ├── chunker.py            # Sliding-window document chunking
│   ├── embedder.py           # Dense vector embedder (SentenceTransformers & fallback)
│   ├── indexer.py            # FAISS vector indexer & serialization
│   └── search_engine.py      # Core semantic query coordinator
├── tests/
│   ├── test_chunker.py       # Chunking & overlap tests
│   ├── test_embedder.py      # Embedding shape & L2 normalization tests
│   ├── test_indexer.py       # Vector indexing & persistence tests
│   └── test_search_engine.py # End-to-end semantic query tests
├── app.py                    # Streamlit web application
├── cli.py                    # Headless CLI runner
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+

### 2. Installation

```bash
cd semantic-document-search
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

python -m pip install -r requirements.txt
```

### 3. Running Semantic Search via CLI

Index the documents and run a natural language query:

```bash
python cli.py --query "How does circuit breaker protect microservices from cascading failure?" --top-k 3
```

### 4. Running the Interactive Web App

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Automated Tests

Run the PyTest test suite:

```bash
python -m pytest tests -v
```

Tests verify:
- Sliding-window word tokenization and boundary preservation
- 384-dimensional vector embedding generation and unit-length L2 normalization
- FAISS `IndexFlatIP` indexing, top-k ranking, and disk save/load roundtrips
- Semantic query matching on test documents

---

## Security Considerations

1. **Path Traversal**: When indexing user-uploaded folders, sanitize paths to ensure ingestion remains constrained within designated corpus directories.
2. **Denial of Service (DoS)**: Extremely long query texts should be truncated to maximum token lengths before vector encoding to avoid resource exhaustion.

---

## Known Limitations

- **Exact Nearest Neighbor**: Uses `IndexFlatIP` (exact brute-force cosine search); for collections exceeding 1,000,000 documents, approximate indexing (`IndexIVFFlat` or `IndexHNSWFlat`) should be enabled for sub-millisecond scaling.
- **Multilingual Support**: Default embeddings are optimized for English; multilingual retrieval requires multilingual models (e.g. `paraphrase-multilingual-MiniLM-L12-v2`).
