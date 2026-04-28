import streamlit as st
import time
import re

from pdf_loader import extract_text_from_pdf
from chunker import split_text_into_chunks
from vector_store import create_vector_store, search_query
from reranker import rerank
from llm import generate_answer, generate_questions_from_document, is_answerable
from query_rewriter import rewrite_query
from pdf_viewer import render_pdf_page
import pandas as pd
from model_comparison import compare_models


import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


from sentence_transformers import SentenceTransformer, util

# Load once (important)
@st.cache_resource
def load_eval_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

eval_model = load_eval_model()


def retrieval_accuracy(results):
    scores = [r["score"] for r in results]
    return sum(scores) / len(scores)


def answer_similarity(answer, context):
    emb1 = eval_model.encode(answer, convert_to_tensor=True)
    emb2 = eval_model.encode(context, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))


def faithfulness(answer, context):
    emb1 = eval_model.encode(answer, convert_to_tensor=True)
    emb2 = eval_model.encode(context, convert_to_tensor=True)
    score = float(util.cos_sim(emb1, emb2))
    return score


def query_coverage(query, answer):
    q_words = set(query.lower().split())
    a_words = set(answer.lower().split())
    return len(q_words & a_words) / len(q_words)


@st.cache_data
def get_model_comparison_results(chunks):
    sample_query = "What is the main topic of this document?"
    return compare_models(chunks, sample_query)


# --------------------------
# 🔥 ADD: Ranking Function
# --------------------------
def simple_rank(q_list):
    def score(q):
        text = q["question"].lower() if isinstance(q, dict) else str(q).lower()

        if "what is" in text:
            return 10
        elif "difference" in text or "compare" in text:
            return 9
        elif "why" in text or "how" in text:
            return 8
        else:
            return 7

    return sorted(q_list, key=score, reverse=True)


# --------------------------
# Normalize
# --------------------------
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())


# --------------------------
# Extract clean sentence
# --------------------------
def get_clean_text(text):
    sentences = text.split(".")
    clean = ".".join(sentences[:2]).strip()
    if not clean.endswith("."):
        clean += "."
    return clean


# --------------------------
# Session
# --------------------------
if "current_query" not in st.session_state:
    st.session_state.current_query = None


# --------------------------
# Animation
# --------------------------
def thinking_animation():
    placeholder = st.empty()
    for dots in ["⏳ Thinking.", "⏳ Thinking..", "⏳ Thinking..."]:
        placeholder.markdown(f"<div class='bot'>{dots}</div>", unsafe_allow_html=True)
        time.sleep(0.3)
    placeholder.empty()


# --------------------------
# Config
# --------------------------
st.set_page_config(page_title="Document QA System", layout="wide")


# --------------------------
# Toggle
# --------------------------
top_left, top_right = st.columns([9,1])
with top_right:
    dark_mode = st.toggle("🌙", label_visibility="collapsed")


# --------------------------
# Colors
# --------------------------
if dark_mode:
    bg = "#0f172a"
    user_bg = "linear-gradient(135deg,#7f1d1d,#b91c1c)"
    bot_bg = "linear-gradient(135deg,#111827,#1e293b)"
    text = "#e5e7eb"
    evidence_bg = "#93c5fd"
    suggestion_color = "#ffffff"
else:
    bg = "#f9fafb"
    user_bg = "linear-gradient(135deg,#fee2e2,#fecaca)"
    bot_bg = "linear-gradient(135deg,#ffffff,#e5e7eb)"
    text = "#111827"
    evidence_bg = "#bfdbfe"
    suggestion_color = "#000000"


# --------------------------
# CSS
# --------------------------
st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {bg};
    color: {text};
}}
</style>
""", unsafe_allow_html=True)


# --------------------------
# Header
# --------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>
        📄 Document Question Answering System
    </h1>
    """,
    unsafe_allow_html=True
)


# --------------------------
# Session Init
# --------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = None


# --------------------------
# Upload
# --------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and st.session_state.index is None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🔄 Processing document..."):

        pages = extract_text_from_pdf("temp.pdf")
        chunks = split_text_into_chunks(pages)
        index = create_vector_store(chunks)

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success("✅ Document uploaded successfully!")


if st.session_state.index is not None:

    with st.expander("📊 View Model Comparisons"):

        import matplotlib.pyplot as plt
        import numpy as np

        with st.spinner("Running model comparisons..."):

            # =========================
            # 🔹 Embedding
            # =========================
            st.subheader("🔹 Embedding Model Comparison")

            sample_query = "What is the main topic of this document?"
            comparison_results = get_model_comparison_results(st.session_state.chunks)

            df = pd.DataFrame(comparison_results)
            st.table(df)

            models = [r["model"] for r in comparison_results]
            top_scores = [r["top_score"] for r in comparison_results]
            avg_scores = [r["avg_score"] for r in comparison_results]

            x = np.arange(len(models))
            width = 0.35

            fig, ax = plt.subplots(figsize=(4.5, 2.8))

            bars1 = ax.bar(x - width/2, top_scores, width, label="Top Score")
            bars2 = ax.bar(x + width/2, avg_scores, width, label="Avg Score")

            # ✅ Add headroom
            ax.set_ylim(0, max(top_scores) * 1.25)

            # ✅ Add values
            for bars in [bars1, bars2]:
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        yval + 0.01,
                        f"{float(yval):.4f}",
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

            ax.set_title("Embedding Model Comparison", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=25, fontsize=8)
            ax.legend(fontsize=8)

            plt.tight_layout(pad=0.5)
            st.pyplot(fig)
# --------------------------
# Chat
# --------------------------
if st.session_state.index:

    if "questions" not in st.session_state:

        pages = extract_text_from_pdf("temp.pdf")

        with st.spinner("🧠 Generating questions..."):
            questions = generate_questions_from_document(pages)


        # 🔥 FILTER ONLY ANSWERABLE QUESTIONS
        filtered_questions = []

        for q in questions:
            if is_answerable(q["question"], st.session_state.index, st.session_state.chunks):
                filtered_questions.append(q)

        # fallback (avoid empty)
        if filtered_questions:
            questions = filtered_questions    

        # 🔥 ADD: fallback if empty
        if not questions:
            st.warning("⚠️ No questions generated. Using default questions.")
            questions = [{"question": "What is the main topic of this document?"}]

        # 🔥 Rank
        questions = simple_rank(questions)

        # 🔥 LIMIT TO TOP 15
        questions = questions[:15]

        st.session_state.questions = questions


    # --------------------------
    # Questions UI
    # --------------------------
    selected_question_obj = st.selectbox(
        "📌 Suggested Questions from Document:",
        st.session_state.questions,
        format_func=lambda x: x["question"] if isinstance(x, dict) else x
    )

    selected_question = selected_question_obj["question"] if isinstance(selected_question_obj, dict) else selected_question_obj


    # 🔥 Manual input
    user_input = st.text_input("✍️ Or type your own question:")

    if user_input:
        selected_question = user_input


    # Button
    if st.button("Get Answer"):
        st.session_state.current_query = selected_question


    if st.session_state.current_query:

        query = st.session_state.current_query

        # 🔥 ADD THIS HERE
        processed_query = rewrite_query(query)

        st.session_state.messages.append({"role": "user", "content": query})

        try:
            # 🔥 IMPROVE SHORT / DEFINITION QUERIES
            if len(query.split()) <= 3:
                processed_query = f"Explain the concept of {query} in the document"
        except:
            processed_query = query


        results = search_query(processed_query, st.session_state.index, st.session_state.chunks, top_k=10)
        results = rerank(processed_query, results, top_k=5)

        # 🔥 SORT RESULTS BY SCORE (DESCENDING)
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # 🔥 SMART FILTER
        top_score = results[0]["score"]

        # keep results close to best score
        results = [r for r in results if r["score"] >= top_score - 0.1]

        # fallback if too few
        if len(results) < 2:
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]

        context = " ".join([r["text"] for r in results])

        thinking_animation()

        answer = generate_answer(processed_query, context)

        # =========================
        # 📊 Evaluation Metrics
        # =========================
        st.subheader("📊 Evaluation Metrics")

        retrieval_acc = retrieval_accuracy(results)
        sim_score = answer_similarity(answer, context)
        faith = faithfulness(answer, context)
        coverage = query_coverage(query, answer)

        metrics_df = pd.DataFrame({
            "Metric": ["Retrieval Accuracy", "Answer Similarity", "Faithfulness", "Query Coverage"],
            "Score": [retrieval_acc, sim_score, faith, coverage]
        })

        st.table(metrics_df)

        pages_used = sorted(set([r["page"] for r in results]))
        citation = ", ".join([f"Page {p}" for p in pages_used])

        not_found = "not found" in answer.lower()

        if not_found:
            final_answer = f"<b>🤖 Answer:</b><br><br>{answer}"
        else:
            final_answer = f"<b>🤖 Answer:</b><br><br>{answer} <b>[{citation}]</b>"


        # 🔥 REMOVE DUPLICATE PAGES
        seen_pages = set()
        evidence = []

        for r in results:
            if r["page"] in seen_pages:
                continue

            seen_pages.add(r["page"])

            img = render_pdf_page("temp.pdf", r["page"])
            if img:
                evidence.append({**r, "image": img})


        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "evidence": evidence
        })

        st.session_state.current_query = None


    # --------------------------
    # Display
    # --------------------------
    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(f"<div class='user'>{msg['content']}</div>", unsafe_allow_html=True)

        else:
            st.markdown(f"<div class='bot'>{msg['content']}</div>", unsafe_allow_html=True)

            if "evidence" in msg and "not found" not in msg["content"].lower():

                with st.expander("🔍 View Sources"):

                    for ev in msg["evidence"]:

                        with st.container():

                            clean_text = get_clean_text(ev["text"])

                            st.markdown(f"""
                            <div class="evidence">
                            📄 <b>Page {ev['page']}</b> | Score: {ev['score']:.2f}
                            <br><br>
                            <i>{clean_text}</i>
                            </div>
                            """, unsafe_allow_html=True)

                            col1, col2, col3 = st.columns([1,2,1])
                            with col2:
                                st.image(ev["image"], width="stretch")
                                st.markdown(
                                    f"<div class='caption'>📄 Page {ev['page']} Preview</div>",
                                    unsafe_allow_html=True
                                )   