import os
from groq import Groq
from difflib import get_close_matches

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🔥 simple fallback dictionary
COMMON_WORDS = [
    "what", "is", "transformer", "encoder", "decoder",
    "self", "attention", "model", "define", "explain"
]

def simple_spell_fix(query):
    words = query.split()
    corrected = []

    for word in words:
        match = get_close_matches(word.lower(), COMMON_WORDS, n=1, cutoff=0.7)
        if match:
            corrected.append(match[0])
        else:
            corrected.append(word)

    return " ".join(corrected)


def rewrite_query(query):

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"Fix spelling only:\n{query}"
            }],
            temperature=0
        )

        corrected = response.choices[0].message.content.strip()

        # 🔥 fallback if API didn't correct
        if corrected.lower() == query.lower():
            return simple_spell_fix(query)

        return corrected

    except:
        # 🔥 fallback if API fails
        return simple_spell_fix(query)