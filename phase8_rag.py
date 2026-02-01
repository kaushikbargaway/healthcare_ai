import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load vector DB
index = faiss.read_index("vector_db/medical_index.faiss")
chunks = np.load("vector_db/chunks.npy", allow_pickle=True)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM client
client = OpenAI()

def retrieve(query, k=3):
    q_vec = embed_model.encode([query])
    D, I = index.search(q_vec, k)
    return [chunks[i] for i in I[0]]

def generate_answer(query):
    context = "\n".join(retrieve(query))

    prompt = f"""
You are a medical assistant.
Use only the information from the context.
If unsure, say you don't know.

Context:
{context}

User symptoms:
{query}

Provide:
1. Possible causes
2. Explanation
3. General advice
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    q = input("Enter symptoms: ")
    print(generate_answer(q))