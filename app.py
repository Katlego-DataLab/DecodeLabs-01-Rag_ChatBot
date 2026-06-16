## ────────────────────────────────────────────────────
## DecodeLab RAG Chatbot Streamlit UI
## Author : Katlego Mathebula
## ────────────────────────────────────────────────────

import streamlit as st
import faiss
from datetime import datetime
import pytz
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="DecodeLab AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#0f172a !important; color:#cbd5e1; }
footer { visibility:hidden; }
header { visibility:hidden; }
[data-testid="stHeader"] { display:none; }
.block-container { padding:1.5rem 2rem !important; max-width:100% !important; }

.stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] { background:#0f172a !important; }

.cursor-glow { position:fixed; width:600px; height:600px; border-radius:50%; background:radial-gradient(circle,rgba(100,255,218,0.06) 0%,transparent 70%); pointer-events:none; transform:translate(-50%,-50%); z-index:0; top:0; left:0; }

[data-testid="stSidebar"] { background:#0d1424 !important; border-right:1px solid #1e293b; }
[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
[data-testid="stSidebarContent"] { padding:2rem 1.5rem; }

.pill { display:inline-flex; align-items:center; gap:5px; font-size:11px; font-weight:500; padding:4px 12px; border-radius:20px; border:1px solid; margin:2px; }
.pill-teal  { background:rgba(100,255,218,0.08); border-color:rgba(100,255,218,0.3); color:#64ffda !important; }
.pill-blue  { background:rgba(96,165,250,0.08);  border-color:rgba(96,165,250,0.3);  color:#60a5fa !important; }
.pill-slate { background:rgba(148,163,184,0.08); border-color:rgba(148,163,184,0.3); color:#94a3b8 !important; }

.topic-tag { display:inline-block; background:rgba(100,255,218,0.05); border:1px solid rgba(100,255,218,0.15); color:#64ffda !important; font-size:11px; padding:3px 10px; border-radius:10px; margin:2px; }

.msg-user { display:flex; justify-content:flex-end; margin:0.6rem 0; }
.msg-bot  { display:flex; justify-content:flex-start; margin:0.6rem 0; gap:10px; align-items:flex-start; }
.bubble-user { background:linear-gradient(135deg,rgba(100,255,218,0.15),rgba(96,165,250,0.15)); border:1px solid rgba(100,255,218,0.25); color:#e2e8f0; padding:10px 16px; border-radius:16px 16px 4px 16px; max-width:75%; font-size:0.875rem; line-height:1.6; }
.bubble-bot  { background:#111827; border:1px solid #1e293b; color:#cbd5e1; padding:10px 16px; border-radius:16px 16px 16px 4px; max-width:75%; font-size:0.875rem; line-height:1.6; }
.bot-avatar  { width:30px; height:30px; border-radius:50%; background:rgba(100,255,218,0.1); border:1px solid rgba(100,255,218,0.3); display:flex; align-items:center; justify-content:center; font-size:13px; flex-shrink:0; margin-top:2px; }
.src-chip { background:#0f172a; border:1px solid #1e293b; color:#475569; font-size:10px; padding:2px 8px; border-radius:8px; margin:2px; display:inline-block; }

.chat-box { background:#0d1424; border:1px solid #1e293b; border-radius:14px; padding:1.2rem; height:360px; overflow-y:auto; margin-bottom:0.8rem; }
.chat-box::-webkit-scrollbar-thumb { background:#1e293b; border-radius:2px; }

.stChatInput > div { background:#0f172a !important; border:1px solid #1e293b !important; border-radius:12px !important; }
.stChatInput > div:focus-within { border-color:#64ffda !important; }
.stChatInput textarea { color:#64ffda !important; font-size:0.875rem !important; background:#0f172a !important; caret-color:#64ffda !important; }
.stChatInput textarea::placeholder { color:rgba(100,255,218,0.35) !important; }
</style>

<div class="cursor-glow" id="glow"></div>
<script>
document.addEventListener('mousemove',(e)=>{
    const g=document.getElementById('glow');
    if(g){g.style.left=e.clientX+'px';g.style.top=e.clientY+'px';}
});
</script>
""", unsafe_allow_html=True)

## ── CONFIG ────────────────────────────────────────────────────
DATA_PATH   = "data/knowledge_base.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 5
CHUNK_SIZE  = 3

## ── RAG FUNCTIONS (no LLM — retrieval only, stable) ──────────
def load_and_chunk(path, chunk_size=CHUNK_SIZE):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = []
    for section in raw.split("---"):
        section = section.strip()
        if not section or section.startswith("#"):
            continue
        sentences = [s.strip() for s in section.split("\n") if s.strip() and not s.startswith("TOPIC:")]
        for i in range(0, len(sentences), max(1, chunk_size - 1)):
            chunk = " ".join(sentences[i: i + chunk_size])
            if chunk:
                chunks.append(chunk)
    return chunks

def build_index(chunks, embed_model):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, embed_model):
    qv = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)
    scores, indices = index.search(qv, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score > 0.1:
            results.append(chunks[idx])
    return results

def build_answer(context_chunks):
    ## No LLM needed — just return the retrieved chunks as the answer
    if not context_chunks:
        return "I couldn't find relevant information for that question. Try asking about ML, Python, NLP, RAG, or Data Science!"
    return " ".join(context_chunks[:3])

def rule_based(user_input, user_name, waiting_for):
    now = datetime.now().strftime("%H:%M")
    if waiting_for == "name":
        name = user_input.replace("my name is ", "").title() if user_input.startswith("my name is ") else user_input.title()
        return f"What a beautiful name, {name}! 🌟 How can I help you?", name, ""
    if waiting_for == "location":
        return f"Oh {user_input.title()}! That sounds wonderful 🌍", user_name, ""
    if waiting_for == "hobby":
        return f"That's so cool that you enjoy {user_input.title()}! 🎉", user_name, ""
    if user_input in ["hi", "hello", "hey"]:
        return "Hello! I'm DecodeLab AI 🤖 What's your name?", user_name, "name"
    if any(kw in user_input for kw in ["your name", "who built you", "who made you", "who created you"]):
        return "I'm DecodeLab AI, built with 🤍 by Katlego Mathebula!", user_name, ""
    if "how are you" in user_input:
        return (f"I'm doing great, thanks for asking {user_name}! 😊" if user_name else "I'm doing great! 😊"), user_name, ""
    if "time" in user_input:
        return f"The current time is {now} ⏰", user_name, ""
    if "where are you from" in user_input:
        return "I'm from the digital world 💻 Where are you from?", user_name, "location"
    if "hobbies" in user_input or "fun" in user_input:
        return "I love learning and helping people! 🚀 What about you?", user_name, "hobby"
    if "help" in user_input:
        return "Ask me about:\n🧠 Machine Learning · 🤖 Deep Learning\n📊 Data Science · 💬 NLP · 🔍 RAG · 🐍 Python", user_name, ""
    return None, user_name, waiting_for

## ── CACHED LOADER (embedding only — no LLM to crash) ─────────
@st.cache_resource(show_spinner=False)
def load_embed_and_index():
    embed_model = SentenceTransformer(EMBED_MODEL)
    chunks = load_and_chunk(DATA_PATH)
    index = build_index(chunks, embed_model)
    return embed_model, chunks, index

## ── SESSION STATE ─────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "user_name"   not in st.session_state: st.session_state.user_name   = ""
if "waiting_for" not in st.session_state: st.session_state.waiting_for = ""

with st.spinner("Loading AI models..."):
    embed_model, chunks, index = load_embed_and_index()

## ════ SIDEBAR ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h1 style='font-family:Space Grotesk,sans-serif; font-size:1.6rem; color:#e2e8f0; margin:0;'>DecodeLab<br>AI Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64ffda; font-size:0.85rem; font-weight:500; margin:0.3rem 0 1rem;'>AI Engineer · RAG System</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; font-size:0.8rem; line-height:1.7;'>Ask me anything about machine learning, AI, data science, and Python.</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#475569; font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase;'>Topics</p>", unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="topic-tag">Machine Learning</span>
        <span class="topic-tag">Deep Learning</span>
        <span class="topic-tag">NLP</span>
        <span class="topic-tag">RAG</span>
        <span class="topic-tag">Transformers</span>
        <span class="topic-tag">FAISS</span>
        <span class="topic-tag">Python</span>
        <span class="topic-tag">Data Science</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color:#475569; font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase;'>Try asking</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#475569; line-height:2.2;'>
        → What is machine learning?<br>
        → Explain neural networks<br>
        → What is RAG?<br>
        → How does backpropagation work?
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='font-size:11px; color:#334155;'>Built with 🤍 by <span style='color:#64ffda; font-weight:600;'>Katlego Mathebula</span><br>DecodeLab Internship</p>", unsafe_allow_html=True)

## ════ MAIN CONTENT ════════════════════════════════════════════
st.markdown("<p style='font-size:11px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:#64ffda;'>Chat Interface</p>", unsafe_allow_html=True)

st.markdown(f"""
<div style='margin-bottom:1rem;'>
    <span class="pill pill-teal">● RAG Ready</span>
    <span class="pill pill-blue">● {len(chunks)} Chunks Indexed</span>
    <span class="pill pill-slate">● FAISS · cosine similarity</span>
</div>
""", unsafe_allow_html=True)

## chat history
chat_html = '<div class="chat-box" id="chat-box">'
if not st.session_state.messages:
    chat_html += '<div style="text-align:center;padding:3rem 0;color:#334155;"><div style="font-size:2rem;">🤖</div><div style="font-size:0.85rem;margin-top:0.5rem;">Say <strong style="color:#64ffda;">hello</strong> to get started!</div></div>'

for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f'<div class="msg-user"><div class="bubble-user">{msg["content"]}</div></div>'
    else:
        sources_html = ""
        if msg.get("sources"):
            chips = "".join(f'<span class="src-chip">{s[:50]}...</span>' for s in msg["sources"])
            sources_html = f'<div style="margin-top:6px;">{chips}</div>'
        chat_html += f'<div class="msg-bot"><div class="bot-avatar">🤖</div><div><div class="bubble-bot">{msg["content"]}</div>{sources_html}</div></div>'

chat_html += '<div id="chat-end"></div></div>'
st.markdown(chat_html, unsafe_allow_html=True)

## input
if prompt := st.chat_input("Ask me about AI, ML, Data Science..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    response, st.session_state.user_name, st.session_state.waiting_for = rule_based(
        prompt.lower(), st.session_state.user_name, st.session_state.waiting_for
    )

    sources = []
    if response:
        answer = response
    else:
        with st.spinner("🔍 Searching knowledge base..."):
            sources = retrieve(prompt, index, chunks, embed_model)
            answer = build_answer(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()
