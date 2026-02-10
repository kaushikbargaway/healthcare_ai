import os
import joblib
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# -------------------------------
# Load Environment
# -------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# Load Models
# -------------------------------

# Severity classifier
severity_model = joblib.load("models/severity_model.pkl")

# Embedding model (same as training)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Vector DB
index = faiss.read_index("vector_db/medical_index.faiss")
chunks = np.load("vector_db/chunks.npy", allow_pickle=True)

# -------------------------------
# Severity Prediction
# -------------------------------

def predict_severity(text):
    vec = embed_model.encode([text])
    severity = severity_model.predict(vec)[0]
    return severity

def severity_label(sev):
    return {0: "Low", 1: "Moderate", 2: "High"}[sev]

# -------------------------------
# RAG Retrieval
# -------------------------------

def retrieve_chunks(query, k=3):
    vec = embed_model.encode([query])
    D, I = index.search(vec, k)
    return [chunks[i] for i in I[0]]

# -------------------------------
# Groq LLM Call
# -------------------------------

def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# -------------------------------
# HYBRID PIPELINE
# -------------------------------

def analyze_symptoms(user_input):
    # 1. Severity prediction
    sev = predict_severity(user_input)
    sev_text = severity_label(sev)

    # 2. RAG explanation
    docs = retrieve_chunks(user_input)
    context = "\n".join(docs)

    prompt = f"""
You are a medical assistant.
Use ONLY the context below.
Do not diagnose.
Do not prescribe medicines.
If unsure, say you are not certain.

Context:
{context}

User Symptoms:
{user_input}

Provide:
1. Possible causes
2. Simple explanation
3. Safe general advice
"""

    explanation = call_llm(prompt)

    return {
        "severity_level": sev_text,
        "rag_response": explanation
    }

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    symptoms = input("Enter symptoms: ")
    result = analyze_symptoms(symptoms)

    print("\n--- AI Healthcare Assistant ---\n")
    print(f"Severity Level: {result['severity_level']}\n")
    print(result["rag_response"])

    print("\n--------------------------------")
    print("DISCLAIMER:")
    print("This system is for educational purposes only.")
    print("It does NOT provide medical diagnosis or treatment.")
    print("Always consult a qualified healthcare professional.")
    print("--------------------------------")