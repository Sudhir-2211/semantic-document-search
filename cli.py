import argparse
from src.search_engine import SemanticSearchEngine


def main():
    parser = argparse.ArgumentParser(description="Semantic Document Search with FAISS & Embeddings")
    parser.add_argument("--index-dir", default="data/documents", help="Directory of documents to index")
    parser.add_argument("--query", type=str, help="Search query string")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to retrieve")
    args = parser.parse_args()

    engine = SemanticSearchEngine()
    print(f"Indexing documents in '{args.index_dir}'...")
    total_chunks = engine.index_directory(args.index_dir)
    print(f"Indexed {total_chunks} chunks successfully.")

    if args.query:
        print(f"\nSearching for: '{args.query}' (top {args.top_k} results)\n" + "-" * 50)
        results = engine.search(args.query, top_k=args.top_k)
        for i, res in enumerate(results, start=1):
            print(f"[{i}] Score: {res['similarity_score']:.4f} | Source: {res['source']}")
            snippet = res['text'][:220] + "..." if len(res['text']) > 220 else res['text']
            print(f"    Excerpt: {snippet}\n")
    else:
        print("Tip: Run with --query 'your search question' to execute a semantic search.")


if __name__ == "__main__":
    main()
