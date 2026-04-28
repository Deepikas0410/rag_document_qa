# Document Question Answering using RAG

This project is an AI-powered document question answering system that uses Retrieval-Augmented Generation (RAG) to generate context-aware answers from PDF documents.

## Features
- Extracts and processes text from PDF documents
- Splits content into chunks for efficient retrieval
- Uses semantic search to find relevant context
- Generates answers using Large Language Models (LLMs)
- Simple and interactive interface using Streamlit

## Tech Stack
- Python
- Streamlit
- LLM APIs (OpenAI / similar)
- (Optional: LangChain / FAISS if you used them)

## How it Works
1. Upload PDF document
2. Text is extracted and split into smaller chunks
3. Each chunk is converted into embeddings
4. Relevant chunks are retrieved based on user query
5. LLM generates an answer using retrieved context
