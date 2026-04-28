from dotenv import load_dotenv
from groq import Groq
import os
import re
from vector_store import search_query

# =========================================================
# ✅ Initialize Groq Client (SAFE METHOD)
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not set in environment variables")

client = Groq(api_key=api_key)

def is_answerable(question, index, chunks):

    # ✅ allow short queries (definitions)
    if len(question.split()) <= 3:
        return True

    results = search_query(question, index, chunks, top_k=3)

    if not results or results[0]["score"] < 0.2:
        return False

    return True


def is_high_quality(question):

    q = question.lower()

    weak_patterns = [
        "who is", "author", "authors",
        "research team", "affiliation",
        "organization", "university",
        "google", "paper", "published",
        "name of", "who are"
    ]

    for w in weak_patterns:
        if w in q:
            return False

    strong_patterns = [
        "what is", "how", "why",
        "difference", "advantages",
        "working", "architecture",
        "mechanism", "role",
        "function", "model",
        "performance", "training"
    ]

    for s in strong_patterns:
        if s in q:
            return True

    return False

# =========================================================
# ✅ 1. ANSWER GENERATION FUNCTION (RAG)
# =========================================================

def generate_answer(query, context):

    prompt = f"""
You are a document question answering system.

Use the provided context to answer the question.

IMPORTANT:
- The answer may not be explicitly defined but described indirectly
- If the concept is explained in the context, extract and explain it
- Do NOT expect exact definition words
- Use descriptive sentences from the context

- Only say "Answer not found in the document" if absolutely nothing relevant is present

- Be clear and concise

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# =========================================================
# 🔥 2. QUESTION GENERATION FUNCTION (FIXED)
# =========================================================

def generate_questions_from_document(pages):

    from chunker import split_text_into_chunks

    chunks = split_text_into_chunks(pages)

    all_questions = []

    # 🔥 STEP 1: Generate questions from chunks
    for chunk in chunks[:6]:

        chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk
        chunk_text = chunk_text[:1200]   # avoid token overflow

        prompt = f"""
You are a professor preparing exam questions.

Generate 6 to 8 meaningful questions.

IMPORTANT:
- Each question MUST end with '?'
- Return ONLY questions
- No explanations

Text:
{chunk_text}

Questions:
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )

            output = response.choices[0].message.content.strip()

            if not output:
                continue

            lines = output.split("\n")

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()

                if line.startswith("-") or line.startswith("•"):
                    line = line[1:].strip()

                if len(line) < 10:
                    continue

                if not line.endswith("?"):
                    line += "?"

                all_questions.append({
                    "question": line
                })

        except Exception as e:
            print("Chunk error:", e)

    # 🔥 STEP 2: Remove duplicates
    unique_dict = {}
    for q in all_questions:
        unique_dict[q["question"]] = q

    unique_questions = list(unique_dict.values())

    # 🔥 FILTER HIGH QUALITY QUESTIONS
    filtered_questions = []

    for q in unique_questions:
        if is_high_quality(q["question"]):
            filtered_questions.append(q)

    # fallback if too strict
    if len(filtered_questions) < 10:
        filtered_questions = unique_questions

    unique_questions = filtered_questions

    # 🔥 STEP 3: If less → generate from full document
    if len(unique_questions) < 15:

        full_text = " ".join([p["text"] for p in pages[:2]])
        full_text = full_text[:2000]

        prompt = f"""
Generate important questions from this document.

IMPORTANT:
- Questions must be meaningful
- Each must end with '?'
- Generate at least 10 questions

Text:
{full_text}

Questions:
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )

            extra_output = response.choices[0].message.content.strip()

            lines = extra_output.split("\n")

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()

                if len(line) > 10:
                    if not line.endswith("?"):
                        line += "?"

                    unique_questions.append({
                        "question": line
                    })

        except Exception as e:
            print("Fallback error:", e)

    # 🔥 STEP 4: Final cleanup
    final_dict = {}
    for q in unique_questions:
        final_dict[q["question"]] = q

    final_questions = list(final_dict.values())

    return final_questions[:15]

# =========================================================
# 🔥 3. QUESTION RANKING (UNCHANGED)
# =========================================================

def rank_questions(questions):

    ranked = []

    for q in questions:

        question_text = q["question"]

        prompt = f"""
Rate the importance of the following question for understanding a document.

Give a score from 1 to 10.

Question:
{question_text}

Score:
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            score_text = response.choices[0].message.content.strip()

            score = int(''.join(filter(str.isdigit, score_text))[:2] or 5)

            ranked.append({
                "question": question_text,
                "score": score
            })

        except:
            ranked.append({
                "question": question_text,
                "score": 5
            })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked

