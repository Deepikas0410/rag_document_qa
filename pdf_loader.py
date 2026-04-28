import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if text.strip():  # avoid empty pages
            pages.append({
                "page": page_num + 1,
                "text": text.strip()
            })

    doc.close()
    return pages


# =========================================================
# 🔥 NEW: Convert pages → full text (for question generation)
# =========================================================

def get_full_text(pages):
    return " ".join([p["text"] for p in pages])