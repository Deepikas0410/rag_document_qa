from pdf_loader import extract_text_from_pdf
from chunker import split_text_into_chunks
from vector_store import create_vector_store, search_query
from reranker import rerank
from llm import generate_answer, generate_questions_from_document
from query_rewriter import rewrite_query
from model_comparison import compare_models
from reranker_comparison import compare_rerankers

import matplotlib.pyplot as plt
import numpy as np


def main():

    print("\n📄 RAG Document QA System\n")

    pdf_path = input("Enter PDF file path: ")

    # =====================================================
    # Step 1: Load PDF
    # =====================================================
    pages = extract_text_from_pdf(pdf_path)
    print(f"✅ Pages Loaded: {len(pages)}")

    # =====================================================
    # 🔥 NEW: Generate Questions from Document
    # =====================================================
    print("\n🧠 Generating questions from document...\n")

    questions = generate_questions_from_document(pages)

    print("📌 Suggested Questions:\n")
    for i, q in enumerate(questions):
        print(f"{i+1}. {q}")

    # =====================================================
    # Step 2: Chunking
    # =====================================================
    chunks = split_text_into_chunks(pages)
    print(f"\n✅ Chunks Created: {len(chunks)}")

    # =====================================================
    # 🔥 Model Comparison (MISSING PART)
    # =====================================================

    print("\n📊 Running Model Comparison...\n")

    sample_query = "What is the main topic of this document?"

    comparison_results = compare_models(chunks, sample_query)

    # =========================
    # Display as Table
    # =========================

    print("\n📊 Model Comparison Table:\n")

    print("{:<35} {:<15} {:<15}".format("Model", "Top Score", "Avg Score"))
    print("-" * 65)

    for r in comparison_results:
        print("{:<35} {:<15.4f} {:<15.4f}".format(
            r["model"], r["top_score"], r["avg_score"]
        ))    



    models = [r["model"] for r in comparison_results]
    top_scores = [r["top_score"] for r in comparison_results]
    avg_scores = [r["avg_score"] for r in comparison_results]

    x = np.arange(len(models))  # positions
    width = 0.35  # bar width

    plt.figure()

    bars1 = plt.bar(x - width/2, top_scores, width, label="Top Score")
    bars2 = plt.bar(x + width/2, avg_scores, width, label="Avg Score")

    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Embedding Model Comparison")

    plt.xticks(x, models, rotation=30)
    plt.legend()
    plt.tight_layout()

    # ✅ Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval,
                    f"{float(yval):.4f}",
                    ha='center', va='bottom')

    plt.show()

    # =====================================================
    # Step 3: Vector Store
    # =====================================================
    index = create_vector_store(chunks)
    print("✅ Vector Store Created\n")


    # =====================================================
    # Step 4: Query Loop (UPDATED)
    # =====================================================
    while True:

        choice = input("\nEnter question number (or type 'quit'): ")

        if choice.lower() == "quit":
            break

        if not choice.isdigit() or int(choice) > len(questions):
            print("❌ Invalid choice")
            continue

        query = questions[int(choice) - 1]["question"]

        print(f"\n💬 Selected Question: {query}")

        print(f"\n🔍 Original Query: {query}")

        # =================================================
        # Step 5: Query Rewriting
        # =================================================
        processed_query = rewrite_query(query)

        if processed_query != query:
            print(f"✏️ Corrected Query: {processed_query}")

        # =================================================
        # Step 6: Retrieval
        # =================================================
        results = search_query(processed_query, index, chunks, top_k=10)

        print("\n📊 Retrieved Results (Before Rerank):\n")
        for r in results:
            print(f"Score: {r['score']:.4f} | Page: {r['page']}")
            print(r["text"][:150])
            print("-" * 50)


        # =================================================
        # 🔥 NEW: Reranker Comparison (ADD HERE)
        # =================================================
        # Sort by Top Score (descending)
        reranker_results = compare_rerankers(processed_query, results)

        reranker_results = sorted(reranker_results, key=lambda x: x["top_score"], reverse=True)

        print("\n📊 Reranker Comparison Table:\n")

        print("{:<45} {:<15} {:<15}".format("Model", "Top Score", "Avg Score"))
        print("-" * 80)

        for r in reranker_results:
            print("{:<45} {:<15.4f} {:<15.4f}".format(
                r["model"], r["top_score"], r["avg_score"]
            ))


        # =========================
        # Reranker Graph (Top + Avg)
        # =========================

        models = [r["model"] for r in reranker_results]
        top_scores = [float(r["top_score"]) for r in reranker_results]
        avg_scores = [float(r["avg_score"]) for r in reranker_results]

        x = np.arange(len(models))
        width = 0.35

        plt.figure()

        bars1 = plt.bar(x - width/2, top_scores, width, label="Top Score")
        bars2 = plt.bar(x + width/2, avg_scores, width, label="Avg Score")

        plt.xlabel("Reranker Models")
        plt.ylabel("Score")
        plt.title("Reranker Model Comparison")

        plt.xticks(x, models, rotation=30)
        plt.legend()
        plt.tight_layout()

        # ✅ Add values on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval,
                    f"{float(yval):.4f}",
                    ha='center', va='bottom')

        plt.show()
        # =================================================
        # Step 7: Reranking
        # =================================================
        results = rerank(processed_query, results, top_k=3)

        # =================================================
        # Step 8: Build Context
        # =================================================
        context = " ".join([r["text"] for r in results])

        # =================================================
        # Step 9: Generate Answer
        # =================================================
        answer = generate_answer(processed_query, context)

        # =================================================
        # Step 10: Citations
        # =================================================
        pages_used = sorted(list(set([r["page"] for r in results])))
        citation = ", ".join([f"Page {p}" for p in pages_used])

        print("\n🧠 Answer:\n")
        print(f"{answer} [{citation}]")

        # =================================================
        # Step 11: Show Source Chunks
        # =================================================
        print("\n📊 Top Retrieved Chunks:\n")
        for r in results:
            print(f"Page: {r['page']} | Score: {r['score']:.4f}")
            print(r["text"])
            print("\n----------------------\n")


if __name__ == "__main__":
    main()