const form = document.getElementById("grievanceForm");
const chatWindow = document.getElementById("chatWindow");
const userMessage = document.getElementById("userMessage");
const sendBtn = document.getElementById("sendBtn");

let grievanceId = null;

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const name = document.getElementById("name").value.trim();
    const email = document.getElementById("email").value.trim();
    const grievance = document.getElementById("grievance").value.trim();

    if (!name || !email || !grievance) return;

    const response = await fetch("/submit_grievance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, grievance })
    });

    const data = await response.json();
    grievanceId = data.grievance_id;

    alert(`✅ Grievance submitted successfully! Your ID: ${grievanceId}`);

    // Enable chatbot
    userMessage.disabled = false;
    sendBtn.disabled = false;

    form.reset();
});

sendBtn.addEventListener("click", async () => {
    const message = userMessage.value.trim();
    if (!message) return;

    appendMessage("user", message);

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grievance_id: grievanceId, message })
    });

    const data = await response.json();
    appendMessage("bot", data.reply);

    userMessage.value = "";
});

function appendMessage(sender, message) {
    const msgDiv = document.createElement("div");
    msgDiv.className = sender === "bot" ? "bot-msg" : "user-msg";
    msgDiv.textContent = message;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}
