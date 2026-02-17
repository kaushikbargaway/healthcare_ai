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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import uuid
from pydantic import BaseModel
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, TableStyle
)
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime


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

class ReportRequest(BaseModel):
    name: str
    dob: str | None = None
    email: str | None = None
    symptoms: str
    severity_level: str
    analysis: str

class ExplanationRequest(BaseModel):
    explanation: str

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
def download_report(request: ReportRequest):

    os.makedirs("reports", exist_ok=True)

    unique_id = str(uuid.uuid4())[:6].upper()
    timestamp = datetime.now()
    report_id = f"MEDAI-{timestamp.strftime('%Y%m%d')}-{unique_id}"

    file_path = f"reports/{report_id}.pdf"

    # Better margins
    doc = SimpleDocTemplate(
        file_path,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom Styles
    section_title = ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        spaceAfter=8
    )

    normal_text = styles["Normal"]

    content = []

    # ---------- HEADER TABLE ----------
    header_data = []

    if os.path.exists("logo.png"):
        logo = Image("logo.png", width=1*inch, height=1*inch)
    else:
        logo = ""

    header_text = Paragraph(
        "<b>MedAI Healthcare System</b><br/>"
        "AI-Powered Clinical Decision Support",
        styles["Title"]
    )

    header_data.append([logo, header_text])

    header_table = Table(header_data, colWidths=[1.2*inch, 4.5*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))

    content.append(header_table)
    content.append(Spacer(1, 0.3 * inch))

    # ---------- META ----------
    content.append(Paragraph(
        f"<b>Report ID:</b> {report_id}", normal_text))
    content.append(Paragraph(
        f"<b>Generated On:</b> {timestamp.strftime('%d %b %Y | %H:%M')}",
        normal_text))

    content.append(Spacer(1, 0.3 * inch))

    # Divider
    content.append(Table([[""]], colWidths=[6*inch],
                         style=[('LINEABOVE', (0,0), (-1,-1), 1, colors.grey)]))
    content.append(Spacer(1, 0.3 * inch))

    # ---------- PATIENT INFO ----------
    content.append(Paragraph("PATIENT INFORMATION", section_title))
    content.append(Paragraph(f"<b>Name:</b> {request.name}", normal_text))

    if request.dob:
        content.append(Paragraph(f"<b>Date of Birth:</b> {request.dob}", normal_text))

    if request.email:
        content.append(Paragraph(f"<b>Email:</b> {request.email}", normal_text))

    content.append(Spacer(1, 0.3 * inch))

    # ---------- SYMPTOMS ----------
    content.append(Paragraph("SYMPTOMS", section_title))
    content.append(Paragraph(request.symptoms, normal_text))

    content.append(Spacer(1, 0.2 * inch))

    content.append(Paragraph(
        f"<b>Severity Level:</b> {request.severity_level}",
        normal_text
    ))

    content.append(Spacer(1, 0.3 * inch))

    # ---------- ANALYSIS ----------
    content.append(Paragraph("CLINICAL ANALYSIS", section_title))
    content.append(Spacer(1, 0.1 * inch))

    formatted_analysis = request.analysis.replace("\n", "<br/>")
    content.append(Paragraph(formatted_analysis, normal_text))

    content.append(Spacer(1, 0.4 * inch))

    # ---------- DISCLAIMER ----------
    content.append(Table([[""]], colWidths=[6*inch],
                         style=[('LINEABOVE', (0,0), (-1,-1), 1, colors.grey)]))
    content.append(Spacer(1, 0.2 * inch))

    content.append(Paragraph(
        "Disclaimer: This report is generated by an AI system for educational "
        "purposes only. It does not replace professional medical advice.",
        styles["Italic"]
    ))

    doc.build(content)

    return FileResponse(file_path, filename=f"{report_id}.pdf")


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
def download_explained_report(request: ExplanationRequest):

    os.makedirs("reports", exist_ok=True)

    unique_id = str(uuid.uuid4())[:6].upper()
    timestamp = datetime.now()
    report_id = f"MEDAI-REP-{timestamp.strftime('%Y%m%d')}-{unique_id}"

    file_path = f"reports/{report_id}.pdf"

    doc = SimpleDocTemplate(
        file_path,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    section_title = ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        spaceAfter=8
    )

    normal_text = styles["Normal"]

    content = []

    # ---------- HEADER ----------
    header_data = []

    if os.path.exists("logo.png"):
        logo = Image("logo.png", width=1*inch, height=1*inch)
    else:
        logo = ""

    header_text = Paragraph(
        "<b>MedAI Healthcare System</b><br/>"
        "AI-Powered Medical Report Simplification",
        styles["Title"]
    )

    header_data.append([logo, header_text])

    header_table = Table(header_data, colWidths=[1.2*inch, 4.5*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))

    content.append(header_table)
    content.append(Spacer(1, 0.3 * inch))

    # ---------- META ----------
    content.append(Paragraph(
        f"<b>Report ID:</b> {report_id}", normal_text))
    content.append(Paragraph(
        f"<b>Generated On:</b> {timestamp.strftime('%d %b %Y | %H:%M')}",
        normal_text))

    content.append(Spacer(1, 0.3 * inch))

    content.append(Table([[""]], colWidths=[6*inch],
                         style=[('LINEABOVE', (0,0), (-1,-1), 1, colors.grey)]))
    content.append(Spacer(1, 0.3 * inch))

    # ---------- EXPLANATION ----------
    content.append(Paragraph("MEDICAL REPORT EXPLANATION", section_title))
    content.append(Spacer(1, 0.2 * inch))

    for line in request.explanation.split("\n"):
        if line.strip():
            content.append(Paragraph(line.strip(), normal_text))
            content.append(Spacer(1, 0.12 * inch))

    content.append(Spacer(1, 0.4 * inch))

    content.append(Table([[""]], colWidths=[6*inch],
                         style=[('LINEABOVE', (0,0), (-1,-1), 1, colors.grey)]))
    content.append(Spacer(1, 0.2 * inch))

    content.append(Paragraph(
        "Disclaimer: This explanation is generated by an AI system for "
        "educational purposes only. It does not constitute medical diagnosis "
        "or treatment. Always consult a licensed healthcare professional.",
        styles["Italic"]
    ))

    doc.build(content)

    return FileResponse(
        path=file_path,
        filename=f"{report_id}.pdf",
        media_type="application/pdf"
    )
