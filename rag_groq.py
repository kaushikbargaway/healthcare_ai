import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------
# Load Vector DB
# -------------------------
index = faiss.read_index("vector_db/medical_index.faiss")
chunks = np.load("vector_db/chunks.npy", allow_pickle=True)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Retrieve chunks
# -------------------------
def retrieve_chunks(query, k=3):
    vec = embed_model.encode([query])
    D, I = index.search(vec, k)
    return [chunks[i] for i in I[0]]

# -------------------------
# Call Groq LLM
# -------------------------
def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# -------------------------
# RAG PIPELINE
# -------------------------
def generate_answer(query):
    docs = retrieve_chunks(query)
    context = "\n".join(docs)

    prompt = f"""
You are a medical assistant.
Use ONLY the context below.
If unsure, say you do not know.

Context:
{context}

User Symptoms:
{query}

Provide:
1. Possible causes
2. Explanation
3. General advice
"""

    return call_llm(prompt)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    q = input("Enter symptoms: ")
    print("\n--- Medical Assistant Response ---\n")
    print(generate_answer(q))
