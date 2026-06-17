## ────────────────────────────────────────────────────
## DecodeLab RAG Chatbot, Streamlit UI
## Author : Katlego Mathebula
## Stack  : HuggingFace Sentence Transformers + FAISS + Streamlit
## ────────────────────────────────────────────────────

import numpy as np
import faiss
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer

## ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="DecodeLab AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

## ── CONFIG ────────────────────────────────────────────────────
DATA_PATH   = "data/knowledge_base.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 5
CHUNK_SIZE  = 3

## ── SESSION STATE ─────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "user_name"   not in st.session_state: st.session_state.user_name   = ""
if "waiting_for" not in st.session_state: st.session_state.waiting_for = ""

## ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}

/* Hide sidebar and its toggle completely */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="header"] { display: none !important; }

.block-container {
    padding: 2rem 3rem 2rem 3rem !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* ── PAGE TITLE ── */
.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 38px;
    font-weight: 700;
    color: #64ffda;
    letter-spacing: 0.04em;
    text-align: center;
    margin-bottom: 0.2rem;
    text-transform: uppercase;
}
.page-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #475569;
    text-align: center;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

/* ── Status chips ── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-bottom: 1.6rem; }
.chip { font-size: 11px; padding: 4px 12px; border-radius: 999px; font-weight: 500; }
.chip-green  { background: #0f3d2e; color: #4ade80; border: 1px solid #16a34a; }
.chip-blue   { background: #0e2a4a; color: #60a5fa; border: 1px solid #2563eb; }
.chip-purple { background: #2d1b4e; color: #c084fc; border: 1px solid #7c3aed; }

/* ── Chat box ── */
.chat-box {
    background: #0b1628;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    min-height: 380px;
    max-height: 460px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.msg-user {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 8px 0 8px auto;
    max-width: 72%;
    width: fit-content;
    font-size: 14px;
    line-height: 1.5;
    margin-left: auto;
}
.msg-bot {
    background: #1a2740;
    color: #cdd6f4;
    border: 1px solid #1e3a5f;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    margin: 8px auto 8px 0;
    max-width: 80%;
    width: fit-content;
    font-size: 14px;
    line-height: 1.6;
}
.source-chip {
    display: inline-block;
    background: #0f2a4a;
    color: #64b5f6;
    border: 1px solid #1e4a7a;
    border-radius: 999px;
    font-size: 10px;
    padding: 2px 8px;
    margin: 4px 3px 0 0;
}
.empty-state { text-align: center; color: #334155; margin-top: 100px; }
.empty-state-icon { font-size: 52px; display: block; margin-bottom: 12px; }
.empty-state-text { font-size: 16px; }
.empty-state-text span { color: #64ffda; font-weight: 600; }

/* ── Input ── */
[data-testid="stTextInput"] input {
    background: #0b1628 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 14px;
    padding: 12px 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #64ffda !important;
    box-shadow: 0 0 0 2px rgba(100,255,218,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #475569 !important; }

/* ── Send button ── */
[data-testid="stFormSubmitButton"] button {
    background: #064e3b !important;
    color: #64ffda !important;
    border: 1px solid #10b981 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    height: 48px !important;
    min-width: 52px !important;
}
[data-testid="stFormSubmitButton"] button:hover { background: #065f46 !important; }

/* ── Info section below input ── */
.info-section {
    margin-top: 2rem;
    border-top: 1px solid #1e3a5f;
    padding-top: 1.6rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
}
.info-card {
    background: #0d1526;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
}
.info-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64ffda;
    margin-bottom: 0.8rem;
}
.topic-pill {
    display: inline-block;
    background: #0e2a4a;
    color: #60a5fa;
    border: 1px solid #1e4a7a;
    border-radius: 999px;
    font-size: 11px;
    padding: 3px 10px;
    margin: 3px 3px 3px 0;
}
.info-text { font-size: 13px; color: #64748b; line-height: 1.6; }
.author-name { color: #64ffda; font-weight: 600; font-size: 15px; display: block; margin-bottom: 4px; }
.author-role { color: #475569; font-size: 12px; margin-bottom: 10px; display: block; }
.author-link {
    display: inline-block;
    font-size: 11px;
    color: #60a5fa;
    background: #0e2a4a;
    border: 1px solid #1e4a7a;
    border-radius: 999px;
    padding: 3px 10px;
    margin: 2px 3px 2px 0;
    text-decoration: none;
}

footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
""", unsafe_allow_html=True)

## ── RAG FUNCTIONS ─────────────────────────────────────────────
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

@st.cache_resource
def load_models_and_index():
    embed_model = SentenceTransformer(EMBED_MODEL)
    chunks      = load_and_chunk(DATA_PATH)
    embeddings  = embed_model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embed_model, index, chunks

def retrieve(query, index, chunks, embed_model):
    q = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, TOP_K)
    return [chunks[i] for s, i in zip(scores[0], indices[0]) if i != -1 and s > 0.1]

def rule_based(user_input):
    name = st.session_state.user_name
    wf   = st.session_state.waiting_for
    now  = datetime.now().strftime("%H:%M")
    inp  = user_input.strip()
    low  = inp.lower()

    if wf == "name":
        n = inp.replace("my name is ", "").title() if low.startswith("my name is ") else inp.title()
        st.session_state.user_name   = n
        st.session_state.waiting_for = ""
        return f"What a beautiful name, {n}! How can I help you?"
    if wf == "location":
        st.session_state.waiting_for = ""
        return f"Oh {inp.title()}! That sounds like a wonderful place 🌍"
    if wf == "hobby":
        st.session_state.waiting_for = ""
        return f"That's so cool that you enjoy {inp.title()}! 🎉"
    if low in ["hi", "hello", "hey"]:
        st.session_state.waiting_for = "name"
        return "Hello! I am DecodeLab RAG Chatbot 🤖\nWhat's your name?"
    if any(k in low for k in ["your name", "who built you", "who made you", "who created you"]):
        return "I am DecodeLab RAG Chatbot, built with 🤍 by Katlego Mathebula!"
    if "how are you" in low:
        return f"I'm doing great, thanks for asking{' ' + name if name else ''}!"
    if "time" in low:
        return f"The current time is {now} ⏰"
    if "where are you from" in low:
        st.session_state.waiting_for = "location"
        return "I'm from the digital world 💻\nHow about you — where are you from?"
    if "hobbies" in low or "what do you do for fun" in low:
        st.session_state.waiting_for = "hobby"
        return "I love learning and helping people! 🚀\nWhat about you — what do you do for fun?"
    if "help" in low:
        return "You can ask me anything from my knowledge base!\nTopics: Python, ML, Data Science, NLP, RAG, Vector Databases.\nOr just say hello! 😊"
    return None

## ── PAGE HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="page-title">DecodeLab AI Chatbot</div>
<div class="page-subtitle">AI Engineer · RAG System · Built Raw in Python</div>
""", unsafe_allow_html=True)

## Status chips
try:
    _, _, chunks_loaded = load_models_and_index()
    num_chunks = len(chunks_loaded)
    st.markdown(f"""
    <div class="chip-row">
        <span class="chip chip-green">● RAG Ready</span>
        <span class="chip chip-blue">● {num_chunks} Chunks Indexed</span>
        <span class="chip chip-purple">● FAISS · cosine similarity</span>
    </div>
    """, unsafe_allow_html=True)
except Exception:
    st.markdown("""
    <div class="chip-row">
        <span class="chip chip-purple">● Loading RAG system...</span>
    </div>
    """, unsafe_allow_html=True)

## ── CHAT HISTORY ──────────────────────────────────────────────
chat_html = '<div class="chat-box">'
if not st.session_state.messages:
    chat_html += """
    <div class="empty-state">
        <span class="empty-state-icon">🤖</span>
        <div class="empty-state-text">Say <span>hello</span> to get started!</div>
    </div>"""
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f'<div class="msg-user">{msg["content"]}</div>'
        else:
            content = msg["content"].replace("\n", "<br>")
            sources_html = ""
            if msg.get("sources"):
                for chunk in msg["sources"]:
                    preview = chunk[:55].replace("<", "&lt;") + "..."
                    sources_html += f'<span class="source-chip">{preview}</span>'
            chat_html += f'<div class="msg-bot">🤖 {content}{"<br><br>" + sources_html if sources_html else ""}</div>'
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

## ── INPUT FORM ────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([11, 1])
    with col_input:
        user_input = st.text_input("message", label_visibility="collapsed",
                                   placeholder="Ask me about AI, ML, Data Science...")
    with col_send:
        submitted = st.form_submit_button("↑")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    with st.spinner(""):
        rule_reply = rule_based(user_input.strip())
        if rule_reply:
            st.session_state.messages.append({"role": "assistant", "content": rule_reply, "sources": []})
        else:
            try:
                embed_model, index, chunks = load_models_and_index()
                retrieved = retrieve(user_input.strip(), index, chunks, embed_model)
                answer = " ".join(retrieved[:2]) if retrieved else "I couldn't find relevant information for that question."
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": retrieved})
            except Exception:
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, something went wrong. Please try again.", "sources": []})
    st.rerun()

## ── INFO SECTION (below input) ────────────────────────────────
st.markdown("""
<div class="info-section">

  <div class="info-card">
    <div class="info-card-title">📚 What you can search on this chatbot</div>
    <div>
      <span class="topic-pill">Machine Learning</span>
      <span class="topic-pill">Deep Learning</span>
      <span class="topic-pill">NLP</span>
      <span class="topic-pill">RAG</span>
      <span class="topic-pill">Transformers</span>
      <span class="topic-pill">FAISS</span>
      <span class="topic-pill">Python</span>
      <span class="topic-pill">Data Science</span>
      <span class="topic-pill">Neural Networks</span>
      <span class="topic-pill">Vector Databases</span>
      <span class="topic-pill">AI vs ML vs DL</span>
    </div>
    <br>
    <div class="info-text">Try asking: <em>"What is machine learning?", "Explain neural networks", "What is RAG?", "How does backpropagation work?"</em></div>
  </div>

  <div class="info-card">
    <div class="info-card-title"> Built with 🤍 by</div>
    <span class="author-name">Katlego Mathebula</span>
    <span class="author-role">Junior Data Scientist & ML Engineer · DecodeLab 2026</span>
    <div class="info-text" style="margin-bottom: 10px;">
      Built raw in Python, no LangChain. Uses HuggingFace Sentence Transformers + FAISS for real-time semantic search across a custom AI/ML knowledge base.
    </div>
    <a class="author-link" href="https://github.com/Katlego-DataLab" target="_blank">⌥ GitHub</a>
    <a class="author-link" href="https://katlego-datalab.github.io/Website-updated-/" target="_blank">⌥ Portfolio</a>
  </div>

</div>
""", unsafe_allow_html=True)
