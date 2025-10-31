import os
import json
import base64
import certifi
from flask import Flask, request, jsonify
from pymongo import MongoClient
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ----------------------- CONFIGURATION -----------------------
app = Flask(__name__)

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# MongoDB connection
try:
    client = MongoClient(
        "mongodb+srv://<YOUR_USERNAME>:<YOUR_PASSWORD>@ac-psjom9p.nk9m4o6.mongodb.net/?retryWrites=true&w=majority",
        tls=True,
        tlsCAFile=certifi.where()
    )
    db = client["grievances"]
    collection = db["submissions"]
    print("✅ Connected to MongoDB successfully.")
except Exception as e:
    print("❌ MongoDB connection error:", e)

# Groq multimodal client
groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------------- VECTOR STORE -----------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

def build_vector_store():
    global vector_store
    kb_dir = "knowledgebase"
    files = [
        os.path.join(kb_dir, "consumer_act.json"),
        os.path.join(kb_dir, "sectoral_grievance.json"),
        os.path.join(kb_dir, "judgments.json")
    ]

    all_docs = []
    for f in files:
        if not os.path.exists(f):
            print(f"⚠️ Missing file: {f}")
            continue
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            for item in data:
                text = " ".join([str(v) for v in item.values()])
                all_docs.append(Document(page_content=text, metadata=item))

    if not all_docs:
        print("❌ No knowledgebase files found. Context retrieval will not work.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_docs)
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    print(f"✅ Vector store built successfully with {len(split_docs)} chunks.")

# Build vector store at startup
build_vector_store()

# ----------------------- HELPER FUNCTIONS -----------------------
def retrieve_context(query_text, k=5):
    if vector_store is None:
        return ""
    docs = vector_store.similarity_search(query_text, k=k)
    context = "\n".join([d.page_content for d in docs])
    print("\n--- Retrieved Context ---")
    for i, d in enumerate(docs, 1):
        print(f"[Doc {i}]:", d.page_content[:250], "...\n")
    print("-------------------------\n")
    return context

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

# ----------------------- ROUTES -----------------------

# HTML Frontend
@app.route("/")
def home():
    return HTML_PAGE

@app.route("/submit_grievance", methods=["POST"])
def submit_grievance():
    data = request.json
    name, email, grievance = data.get("name"), data.get("email"), data.get("grievance")

    if not name or not grievance:
        return jsonify({"status": "error", "message": "Name and grievance are required."}), 400

    record = {"name": name, "email": email, "grievance": grievance}
    result = collection.insert_one(record)
    grievance_id = str(result.inserted_id)
    return jsonify({"status": "success", "grievance_id": grievance_id})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    grievance_id = data.get("grievance_id")
    message = data.get("message")
    image_id = data.get("image_id")

    if not grievance_id:
        return jsonify({"status": "need_grievance_id", "reply": "Please provide your grievance ID to continue."})

    record = collection.find_one({"_id": {"$eq": grievance_id}})
    if not record:
        return jsonify({"status": "error", "message": "Grievance ID not found."})

    context = retrieve_context(record.get("grievance", ""))

    if image_id:
        image_path = os.path.join("uploads", image_id)
        base64_image = encode_image_to_base64(image_path)
        content = [
            {"type": "text", "text": f"User grievance: {record.get('grievance')}\n\nContext: {context}\n\nUser message: {message}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    else:
        content = [{"type": "text", "text": f"User grievance: {record.get('grievance')}\n\nContext: {context}\n\nUser message: {message}"}]

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    )

    reply = completion.choices[0].message.content
    return jsonify({"status": "success", "reply": reply})

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image provided."})

    img = request.files["image"]
    os.makedirs("uploads", exist_ok=True)
    img_path = os.path.join("uploads", img.filename)
    img.save(img_path)
    return jsonify({"status": "success", "image_id": img.filename})

# ----------------------- SYSTEM PROMPT -----------------------
SYSTEM_PROMPT = """
You are a Consumer Grievance Assistance Chatbot for India. 
Guide users in addressing consumer grievances across sectors like Airlines, Banking, Insurance, Telecom, Real Estate, Electricity, E-Commerce, and Food Safety.

Rules:
1. Greet politely and ask about their grievance.
2. Only handle consumer grievance or related queries. Decline unrelated ones with: "I can only help with consumer-related issues."
3. Ask one question at a time to gather relevant details (cause, date, opposing party, desired relief).
4. Warn if the issue is older than 2 years.
5. Suggest step-by-step remedies clearly:
   Step 1: ...
   Step 2: ...
6. Mention National Consumer Helpline (1800-11-4000) and e-daakhil portal if applicable.
7. Use retrieved context only for factual accuracy; do not reveal it.
8. Always respond in a clear, user-friendly, and structured format.
"""

# ----------------------- HTML PAGE -----------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Consumer Grievance Chatbot — Multimodal</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:linear-gradient(180deg,#f3f6ff,#eef7f3);--card:#ffffff;--accent:#0066ff;--accent2:#00c1b3}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Segoe UI,Roboto,"Helvetica Neue",Arial;background:var(--bg);color:#111}
.container{max-width:1000px;margin:36px auto;padding:28px;border-radius:14px;background:var(--card);box-shadow:0 10px 30px rgba(16,24,40,0.08)}
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:12px}
.logo{height:48px;width:48px;border-radius:10px;background:linear-gradient(135deg,var(--accent),#4f8dff);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700}
h1{margin:0;font-size:20px}
.grid{display:grid;grid-template-columns:420px 1fr;gap:20px}
.card{background:#f9fbff;border-radius:12px;padding:16px;border:1px solid #eef3ff}
.field{margin-bottom:12px}
input[type="text"],input[type="email"],textarea{width:100%;padding:12px;border-radius:8px;border:1px solid #dbe7ff;font-size:14px}
textarea{min-height:110px;resize:vertical}
.btn{display:inline-block;padding:12px 18px;border-radius:10px;background:linear-gradient(90deg,var(--accent),#2ab0ff);color:#fff;border:none;cursor:pointer;font-weight:600}
.small{font-size:13px;color:#4b5563}
#chat {height:520px;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px;background:linear-gradient(180deg,#ffffff,#fbfdff);border-radius:10px;border:1px solid #eef6ff}
.bubble{max-width:78%;padding:12px 16px;border-radius:12px;font-size:14px}
.user{align-self:flex-end;background:linear-gradient(180deg,#0066ff,#2ab0ff);color:#fff;border-bottom-right-radius:2px}
.bot{align-self:flex-start;background:linear-gradient(180deg,#0ea25a,#2ccb8f);color:#fff;border-bottom-left-radius:2px}
.meta{font-size:12px;color:#6b7280;margin-top:6px}
.controls{display:flex;gap:10px;margin-top:12px;align-items:center}
.file-input{display:flex;gap:8px;align-items:center}
.preview{max-width:180px;border-radius:8px;border:1px solid #e6eefc;padding:6px;background:#fff}
.footer-note{margin-top:8px;font-size:13px;color:#6b7280}
.badge{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;color:#0f172a;font-weight:600}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="brand">
      <div class="logo">CG</div>
      <div>
        <h1>Consumer Grievance Chatbot</h1>
        <div class="small">Multimodal support — text + image. Powered by RAG + Groq</div>
      </div>
    </div>
    <div class="badge">Indian Consumer Assistance</div>
  </div>

  <div class="grid">
    <div>
      <div class="card">
        <h3 style="margin-top:0">Submit a new grievance</h3>
        <div class="field"><input id="name" type="text" placeholder="Your full name" /></div>
        <div class="field"><input id="email" type="email" placeholder="Your email (optional)" /></div>
        <div class="field"><textarea id="grievance" placeholder="Describe your grievance in natural language"></textarea></div>
        <div style="display:flex;gap:10px;align-items:center">
          <button class="btn" onclick="submitGrievance()">Submit & Start Chat</button>
          <div class="small">After submission you'll get a unique grievance ID (shown in popup)</div>
        </div>
      </div>

      <div style="height:16px"></div>

      <div class="card">
        <h3 style="margin-top:0">Attach an image during chat</h3>
        <div class="small">You can upload a photo (pothole, damaged product, food package). The bot will analyze it.</div>
        <div style="height:12px"></div>
        <input id="image-file" type="file" accept="image/*" />
        <div class="footer-note">Image analysis uses a vision model if available. If not, you'll be asked to briefly describe the image.</div>
      </div>
    </div>

    <div>
      <div class="card" style="height:100%;display:flex;flex-direction:column">
        <h3 style="margin-top:0">Chat</h3>
        <div id="chat" aria-live="polite"></div>

        <div class="controls">
          <input id="chat-message" type="text" placeholder="Say hello or type your message..." style="flex:1;padding:10px;border-radius:8px;border:1px solid #e6eefc" />
          <button class="btn" onclick="sendMessage()">Send</button>
          <button class="btn" onclick="uploadImageAndSend()" style="background:linear-gradient(90deg,#00bfa5,#00c1b3)">Upload & Send</button>
        </div>
        <div class="footer-note">Tip: If you start with "hello" the bot will ask for grievance ID. Provide it to link the chat to your submission.</div>
      </div>
    </div>
  </div>
</div>

<script>
let grievanceId = '';
function addMessage(sender, text, meta='') {
  const chat=document.getElementById('chat');
  const div=document.createElement('div');
  div.className='bubble '+(sender==='user'?'user':'bot');
  div.textContent=text;chat.appendChild(div);
  if(meta){const m=document.createElement('div');m.className='meta';m.textContent=meta;chat.appendChild(m);}
  chat.scrollTop=chat.scrollHeight;
}
function submitGrievance(){
  const name=document.getElementById('name').value.trim();
  const email=document.getElementById('email').value.trim();
  const grievance=document.getElementById('grievance').value.trim();
  if(!name||!grievance){alert('Please enter your name and grievance.');return;}
  fetch('/submit_grievance',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,email,grievance})})
  .then(r=>r.json()).then(data=>{
    if(data.status==='success'){grievanceId=data.grievance_id;alert('Grievance submitted — your ID: '+grievanceId);
      addMessage('bot','Hello! Please tell me your grievance or start with hello.','Grievance ID: '+grievanceId);}
    else alert('Error: '+(data.message||'unknown'));}).catch(e=>alert('Network error'));}
async function sendMessage(){
  const msgEl=document.getElementById('chat-message');const message=msgEl.value.trim();
  if(!message)return;addMessage('user',message);msgEl.value='';
  const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({grievance_id:grievanceId,message})});
  const data=await res.json();if(data.status==='success')addMessage('bot',data.reply);else addMessage('bot','Error: '+(data.message||'unknown'));}
async function uploadImageAndSend(){
  const input=document.getElementById('image-file');
  if(!input.files||input.files.length===0){alert('Choose an image first');return;}
  const file=input.files[0];const form=new FormData();form.append('image',file);
  const res=await fetch('/upload_image',{method:'POST',body:form});const result=await res.json();
  if(result.status!=='success'){alert('Upload failed');return;}
  grievanceId=grievanceId||'';addMessage('user','[image attached: '+file.name+']');
  const res2=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({grievance_id:grievanceId,message:'Image attached',image_id:result.image_id})});
  const data2=await res2.json();if(data2.status==='success')addMessage('bot',data2.reply);else addMessage('bot','Error: '+(data2.message||'unknown'));}
</script>
</body></html>
"""

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
