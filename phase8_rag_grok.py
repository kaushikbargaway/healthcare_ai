import os
import requests
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

# -------------------------------
# Load Vector Database
# -------------------------------

index = faiss.read_index("vector_db/medical_index.faiss")
chunks = np.load("vector_db/chunks.npy", allow_pickle=True)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Retrieve Relevant Chunks
# -------------------------------

def retrieve_chunks(query, k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# -------------------------------
# Call Grok API
# -------------------------------

def call_grok(prompt):
    url = "https://api.x.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "grok-1",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    result = response.json()

    print("RAW GROK RESPONSE:")
    print(result)

    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    else:
        return "Grok API Error: " + str(result)

# -------------------------------
# RAG PIPELINE
# -------------------------------

def generate_answer(query):
    docs = retrieve_chunks(query)
    context = "\n".join(docs)

    prompt = f"""
You are a medical assistant.
Use ONLY the context below.
If the answer is not in context, say you are not sure.

Context:
{context}

User Symptoms:
{query}

Provide:
1. Possible causes
2. Simple explanation
3. General advice
"""

    answer = call_grok(prompt)
    return answer

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    user_query = input("Enter symptoms: ")
    response = generate_answer(user_query)
    print("\n--- Medical Assistant Response ---\n")
    print(response)
