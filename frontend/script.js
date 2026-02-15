let severityLevel = "";
let baseResponse = "";
let uploadedExplanation = "";

/* Symptom Analysis */
async function analyze() {
    const symptoms = document.getElementById("symptoms").value;

    const res = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ symptoms })
    });

    const data = await res.json();

    severityLevel = data.severity_level;
    baseResponse = data.response;

    const output = document.getElementById("analysisOutput");
    output.style.display = "block";
    output.innerHTML =
        `<div class="${severityLevel.toLowerCase()}">
            Severity: ${severityLevel}
         </div><br>${baseResponse}`;

    document.getElementById("downloadBtn").style.display = "inline-block";
    document.getElementById("chatSection").style.display = "block";

}

/* Download Symptom Report */
async function downloadReport() {
    const symptoms = document.getElementById("symptoms").value;

    const res = await fetch("http://127.0.0.1:8000/download-report", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ symptoms })
    });

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "health_report.pdf";
    a.click();
}

/* Upload Medical Report */
async function uploadReport() {
    const fileInput = document.getElementById("reportFile");
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/explain-report", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    uploadedExplanation = data.explanation;

    const output = document.getElementById("reportOutput");
    output.style.display = "block";
    output.innerText = uploadedExplanation;

    document.getElementById("downloadExplainBtn").style.display = "inline-block";
}

/* Download Explained Report */
async function downloadExplainedReport() {
    const res = await fetch("http://127.0.0.1:8000/download-explained-report", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ explanation: uploadedExplanation })
    });

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "medical_report_explanation.pdf";
    a.click();
}

/* Follow-up Chat */
async function sendFollowUp() {
    const question = document.getElementById("followupInput").value;
    if (!question) return;

    const chatBox = document.getElementById("chatBox");

    // User bubble
    chatBox.innerHTML += `
        <div class="chat-bubble user-bubble">
            ${question}
        </div>
    `;

    document.getElementById("followupInput").value = "";

    const res = await fetch("http://127.0.0.1:8000/followup", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            base_response: baseResponse,
            severity_level: severityLevel,
            user_question: question
        })
    });

    const data = await res.json();

    // Assistant bubble
    chatBox.innerHTML += `
        <div class="chat-bubble bot-bubble">
            ${data.answer}
        </div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;
}
