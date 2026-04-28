from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text_into_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_id = 0

    for page in pages:

        split_chunks = splitter.split_text(page["text"])

        for chunk in split_chunks:

            if chunk.strip():
                chunks.append({
                    "id": chunk_id,
                    "page": page["page"],
                    "text": chunk
                })
                chunk_id += 1

    return chunks