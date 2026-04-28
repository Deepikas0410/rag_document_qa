from sentence_transformers import CrossEncoder

# 🔥 Strong reranker model
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


def rerank(query, results, top_k=3):

    print("\n⚡ Applying Reranking...\n")

    pairs = [(query, r["text"]) for r in results]

    scores = reranker_model.predict(pairs)

    for i, score in enumerate(scores):
        results[i]["rerank_score"] = float(score)

    # Sort by rerank score
    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    print("🏆 Top Results After Reranking:\n")

    for r in results[:top_k]:
        print(f"Rerank Score: {r['rerank_score']:.4f} | Page: {r['page']}")
        print(r["text"][:150])
        print("-" * 50)

    return results[:top_k]