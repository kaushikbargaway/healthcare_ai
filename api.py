import os
import joblib
import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import uuid



# -------------------------------
# Load Environment & Models
# -------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

severity_model = joblib.load("models/severity_model.pkl")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vector_db/medical_index.faiss")
chunks = np.load("vector_db/chunks.npy", allow_pickle=True)

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="AI Healthcare Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],      # allow all methods (GET, POST, OPTIONS)
    allow_headers=["*"],      # allow all headers

)

# -------------------------------
# Request Schema
# -------------------------------
class SymptomRequest(BaseModel):
    symptoms: str

class FollowUpRequest(BaseModel):
    base_response: str
    severity_level: str
    user_question: str

# -------------------------------
# Helper Functions
# -------------------------------
def rule_based_severity(text):
    text = text.lower()
    if any(x in text for x in ["chest pain", "shortness of breath", "seizure", "stroke"]):
        return 2
    if any(x in text for x in ["runny nose", "sneezing", "itching"]):
        return 0
    return None

def predict_severity(text):
    vec = embed_model.encode([text])
    return severity_model.predict(vec)[0]

def retrieve_chunks(query, k=3):
    vec = embed_model.encode([query])
    _, I = index.search(vec, k)
    return [chunks[i] for i in I[0]]

def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/analyze")
def analyze_symptoms(request: SymptomRequest):
    user_input = request.symptoms

    rule_sev = rule_based_severity(user_input)
    sev = rule_sev if rule_sev is not None else predict_severity(user_input)

    severity_label = {0: "Low", 1: "Moderate", 2: "High"}[sev]

    docs = retrieve_chunks(user_input)
    context = "\n".join(docs)

    prompt = f"""
You are a medical assistant.
Use ONLY the context below.
Do not diagnose or prescribe medication.

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
        "severity_level": severity_label,
        "response": explanation,
        "disclaimer": "Educational use only. Consult a healthcare professional."
    }

@app.post("/followup")
def follow_up(request: FollowUpRequest):
    prompt = f"""
You are a healthcare assistant chatbot.

Base Medical Analysis:
Severity Level: {request.severity_level}
{request.base_response}

User Question:
{request.user_question}

Rules:
- Do NOT add new medical conditions
- Do NOT change severity level
- Do NOT diagnose
- Only explain or clarify the base analysis
- Keep response simple and safe
"""

    answer = call_llm(prompt)

    return {
        "answer": answer,
        "disclaimer": "This response is for educational purposes only."
    }

@app.post("/download-report")
def download_report(request: SymptomRequest):
    # Reuse analyze logic
    result = analyze_symptoms(request)

    file_id = str(uuid.uuid4())
    file_path = f"reports/{file_id}.pdf"
    os.makedirs("reports", exist_ok=True)

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>AI Healthcare Assistant – Health Summary Report</b>", styles["Title"]))
    content.append(Paragraph(f"<b>Symptoms:</b> {request.symptoms}", styles["Normal"]))
    content.append(Paragraph(f"<b>Severity Level:</b> {result['severity_level']}", styles["Normal"]))
    content.append(Paragraph("<b>Explanation:</b>", styles["Heading2"]))
    content.append(Paragraph(result["response"].replace("\n", "<br/>"), styles["Normal"]))
    content.append(Paragraph("<b>Disclaimer:</b> Educational use only. Consult a healthcare professional.", styles["Italic"]))

    doc.build(content)

    return FileResponse(file_path, filename="health_report.pdf")

@app.post("/explain-report")
def explain_medical_report(file: UploadFile = File(...)):
    text = ""

    try:
        if file.filename.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(file.file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            text = file.file.read().decode("utf-8")

        if not text.strip():
            return {
                "explanation": "Unable to extract readable text from the uploaded report. Please upload a text-based PDF or report.",
                "disclaimer": "Educational use only."
            }

        prompt = f"""
You are a healthcare explanation assistant.

Explain the following medical report in SIMPLE language.
Do NOT diagnose.
Do NOT suggest medicines.
Only explain what the terms generally mean.
Provide general lifestyle or awareness advice.

Medical Report:
{text[:4000]}
"""

        explanation = call_llm(prompt)

        return {
            "explanation": explanation,
            "disclaimer": "This explanation is for educational purposes only."
        }

    except Exception as e:
        return {
            "explanation": f"Error processing report: {str(e)}",
            "disclaimer": "Educational use only."
        }
    
@app.post("/download-explained-report")
def download_explained_report(payload: dict):
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    import uuid, os

    explanation = payload.get("explanation", "")

    if not explanation.strip():
        return {"error": "No explanation provided"}

    os.makedirs("reports", exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = f"reports/explained_report_{file_id}.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Medical Report Explanation</b>", styles["Title"]))
    content.append(Paragraph(
        "This document provides a simplified explanation of an uploaded medical report.",
        styles["Normal"]
    ))
    content.append(Paragraph("<br/><b>Explanation:</b>", styles["Heading2"]))
    content.append(Paragraph(explanation.replace("\n", "<br/>"), styles["Normal"]))
    content.append(Paragraph(
        "<br/><i>Disclaimer: This explanation is for educational purposes only and does not replace professional medical advice.</i>",
        styles["Italic"]
    ))

    doc.build(content)

    return FileResponse(file_path, filename="medical_report_explanation.pdf")
