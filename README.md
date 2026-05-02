# 🧠 AI Knowledge Assistant

A RAG (Retrieval Augmented Generation) system that answers questions strictly based on your uploaded documents — no hallucinations, no general AI knowledge.

## How it works

1. Upload a PDF document
2. The system extracts text, splits it into chunks and converts them to vector embeddings using `sentence-transformers`
3. Embeddings are stored in a FAISS vector database
4. When you ask a question, it finds the most semantically similar chunks
5. Those chunks are passed to Gemini LLM which answers strictly from that context

## Tech Stack

- **FastAPI** — REST API backend
- **FAISS** — vector similarity search
- **sentence-transformers** — text embeddings (`all-MiniLM-L6-v2`)
- **Google Gemini API** — LLM for answer generation
- **PyPDF2** — PDF text extraction
- **Vanilla JS + HTML** — frontend UI

## Setup

```bash
# Clone the repo
git clone https://github.com/sarkayyyy/knowledge-assistant.git
cd knowledge-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-multipart pypdf2 faiss-cpu sentence-transformers google-generativeai python-dotenv

# Add your Gemini API key
echo GEMINI_API_KEY=your_key_here > .env

# Run the server
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings
                                              ↓
User Question → Embedding → FAISS Search → Top 5 Chunks
                                              ↓
                                        Gemini LLM → Answer
```