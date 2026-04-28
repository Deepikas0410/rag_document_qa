from sentence_transformers import CrossEncoder
import numpy as np

def compare_rerankers(query, results):

    models = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ]

    comparison = []

    for model_name in models:

        print(f"\n🔍 Testing Reranker: {model_name}")

        model = CrossEncoder(model_name)

        pairs = [(query, r["text"]) for r in results]

        scores = model.predict(pairs)

        top_score = float(max(scores))
        avg_score = float(np.mean(scores))

        print(f"Top Rerank Score: {top_score:.4f}")
        print(f"Avg Rerank Score: {avg_score:.4f}")

        comparison.append({
            "model": model_name,
            "top_score": top_score,
            "avg_score": avg_score
        })

    return comparison