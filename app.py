import streamlit as st
import os
from src.search_engine import SemanticSearchEngine

st.set_page_config(
    page_title="Semantic Document Search",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Semantic Document Search Engine")
st.markdown("Dense vector retrieval powered by **FAISS** (IndexFlatIP) and **Sentence Embeddings**.")

@st.cache_resource
def get_engine():
    engine = SemanticSearchEngine()
    if os.path.exists("data/documents"):
        engine.index_directory("data/documents")
    return engine

engine = get_engine()

# Sidebar
st.sidebar.header("Search & Index Settings")
top_k = st.sidebar.slider("Number of Results (Top K):", min_value=1, max_value=10, value=3)
min_score = st.sidebar.slider("Minimum Similarity Score:", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Index Statistics")
st.sidebar.write(f"- Total Chunks in Index: **{engine.indexer.index.ntotal}**")
st.sidebar.write(f"- Vector Dimension: **{engine.indexer.dimension}**")
st.sidebar.write(f"- Backend: **{'Transformer (HF)' if engine.embedder.is_transformer else 'Dense Hash L2'}**")

# Query Input
query = st.text_input(
    "Enter a natural language search query:",
    placeholder="e.g. How does circuit breaker handle downstream microservice failures?",
)

sample_queries = [
    "How does circuit breaker handle downstream microservice failures?",
    "What is the difference between IndexFlatIP and HNSW in FAISS?",
    "How does OAuth PKCE prevent authorization code interception?",
    "What statistical tests detect data drift in machine learning models?",
    "How to mitigate cache stampede thundering herd in Redis?",
]

st.markdown("**Sample Search Queries:**")
cols = st.columns(len(sample_queries))
for idx, q_text in enumerate(sample_queries):
    if cols[idx].button(f"Query {idx+1}", help=q_text):
        query = q_text

if query:
    results = engine.search(query, top_k=top_k, min_score=min_score)

    st.markdown(f"### Results for: *'{query}'* ({len(results)} matches)")
    if results:
        for res in results:
            score = res["similarity_score"]
            badge_color = "#10b981" if score > 0.5 else ("#3b82f6" if score > 0.25 else "#94a3b8")

            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid rgba(255,255,255,0.1); border-left: 5px solid {badge_color}; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; background: rgba(255,255,255,0.03);">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.85rem; color: #94a3b8;">
                        <span>📄 <strong>Source:</strong> {res['source']}</span>
                        <span style="background: rgba(255,255,255,0.1); padding: 0.15rem 0.5rem; border-radius: 4px; color: {badge_color}; font-weight: 700;">Score: {score:.4f}</span>
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.5; color: #e2e8f0;">
                        {res['text']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No document chunks met the current similarity threshold. Try adjusting the query or lowering the threshold.")
