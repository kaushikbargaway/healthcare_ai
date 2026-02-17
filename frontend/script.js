let severityLevel = "";
let baseResponse = "";
let uploadedExplanation = "";

let analysisHTML = "";
let reportHTML = "";

/* SYMPTOM ANALYSIS */
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

    analysisHTML = `
        <div class="result-card" id="analysisCard">
            <h3>Symptom Analysis Result</h3>
            <div class="${severityLevel.toLowerCase()}">
                Severity Level: ${severityLevel}
            </div>
            <hr>
            ${baseResponse.replace(/\n/g, "<br>")}
            <br><br>
            <button onclick="downloadReport()">Download Report</button>
        </div>
    `;

    renderResults();
}

/* REPORT EXPLAINER */
async function uploadReport() {
    const file = document.getElementById("reportFile").files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/explain-report", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    uploadedExplanation = data.explanation;

    reportHTML = `
        <div class="result-card" id="reportCard">
            <h3>Medical Report Explanation</h3>
            ${uploadedExplanation.replace(/\n/g, "<br>")}
            <br><br>
            <button onclick="downloadExplainedReport()">Download Explanation</button>
        </div>
    `;

    renderResults();
}

/* Render Logic */
function renderResults() {
    const container = document.getElementById("resultContainer");

    if (analysisHTML && reportHTML) {
        container.style.gridTemplateColumns = "1fr 1fr";
        container.innerHTML = analysisHTML + reportHTML;
    } else {
        container.style.gridTemplateColumns = "1fr";
        container.innerHTML = analysisHTML || reportHTML;
    }
}

/* Floating Chat */
function toggleChat() {
    const chatWindow = document.getElementById("chatWindow");
    chatWindow.style.display =
        chatWindow.style.display === "flex" ? "none" : "flex";
}

async function sendFloatingMessage() {
    const input = document.getElementById("floatingInput");
    const message = input.value.trim();
    if (!message) return;

    const chatBox = document.getElementById("floatingChatBox");

    chatBox.innerHTML += `
        <div class="chat-bubble user-bubble">${message}</div>
    `;

    input.value = "";

    if (!baseResponse) {
        chatBox.innerHTML += `
            <div class="chat-bubble bot-bubble">
                Please analyze your symptoms first so I can provide accurate guidance.
            </div>
        `;
        chatBox.scrollTop = chatBox.scrollHeight;
        return;
    }

    const res = await fetch("http://127.0.0.1:8000/followup", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            base_response: baseResponse,
            severity_level: severityLevel,
            user_question: message
        })
    });

    const data = await res.json();

    chatBox.innerHTML += `
        <div class="chat-bubble bot-bubble">${data.answer}</div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;
}
