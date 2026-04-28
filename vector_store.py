import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔥 Strong embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# =========================
# CREATE VECTOR STORE
# =========================
def create_vector_store(chunks):

    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"✅ Indexed {len(chunks)} chunks")

    return index


# =========================
# SEARCH QUERY (NO RERANK)
# =========================
def search_query(query, index, chunks, top_k=10):

    print(f"\n🔍 Query: {query}")

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for i, idx in enumerate(indices[0]):

        if idx == -1:
            continue

        results.append({
            "id": idx,
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(scores[0][i])
        })

    # 🔍 DEBUG: Show raw retrieval
    print("\n📊 Retrieved Results (Before Rerank):\n")
    for r in results:
        print(f"Score: {r['score']:.4f} | Page: {r['page']}")
        print(r["text"][:150])
        print("-" * 50)

    return results