import os
import pickle
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index(filepaths):
    all_chunks = []
    all_sources = []

    for filepath in filepaths:
        print(f"Processing: {filepath}")
        text = extract_text_from_pdf(filepath)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_sources.extend([os.path.basename(filepath)] * len(chunks))

    print(f"Total chunks: {len(all_chunks)}")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open("index_data.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

    faiss.write_index(index, "faiss.index")
    print("Index built successfully!")