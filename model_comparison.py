from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def compare_models(chunks, query):

    models = [
        "BAAI/bge-base-en-v1.5",
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1"
    ]

    results_data = []

    texts = [chunk["text"] for chunk in chunks]

    for model_name in models:

        print(f"\n🔍 Testing Model: {model_name}")

        model = SentenceTransformer(model_name)

        # embeddings
        embeddings = model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        # FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # query embedding
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding, 5)

        top_score = float(scores[0][0])
        avg_score = float(np.mean(scores[0]))

        print(f"Top Score: {top_score:.4f}")
        print(f"Average Score: {avg_score:.4f}")

        results_data.append({
            "model": model_name,
            "top_score": top_score,
            "avg_score": avg_score
        })

    return results_data