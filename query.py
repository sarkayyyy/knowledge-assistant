import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv                                          #loads API key from .env file

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")

def load_index():                                                       #loads index ---> vector database
    index = faiss.read_index("faiss.index")
    with open("index_data.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["sources"]

def search(query, index, chunks, sources, top_k=5):                     #finds most relevant text chunks for a query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i < len(chunks):
            results.append({
                "chunk": chunks[i],
                "source": sources[i]
            })
    return results

def answer(query):
    index, chunks, sources = load_index()
    relevant_chunks = search(query, index, chunks, sources)

    # Build context from retrieved chunks
    context = ""
    for r in relevant_chunks:
        context += f"[From: {r['source']}]\n{r['chunk']}\n\n"

    prompt = f"""You are a knowledge assistant. Answer the user's question ONLY using the context below.
If the answer is not in the context, say "This information is not in the uploaded documents."

Context:
{context}

Question: {query}
Answer:"""

    response = gemini.generate_content(prompt)
    return response.text