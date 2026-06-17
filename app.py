## ────────────────────────────────────────────────────
## DecodeLab RAG Chatbot — Streamlit UI
## Author : Katlego Mathebula
## Stack  : HuggingFace Sentence Transformers + FAISS + Streamlit
## ────────────────────────────────────────────────────

import os
import numpy as np
import faiss
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer

## ── PAGE CONFIG (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="DecodeLab AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

## ── CONFIG ────────────────────────────────────────────────────
DATA_PATH   = "data/knowledge_base.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 5
CHUNK_SIZE  = 3

## ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Base theme ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit sidebar toggle arrow ── */
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1526 !important;
    border-right: 1px solid #1e3a5f !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* ── Persistent toggle button (lives in main area) ── */
#sidebar-toggle-container {
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 99999;
}
.toggle-btn {
    background: #1e3a5f;
    border: 1px solid #2d5a8e;
    color: #64ffda;
    font-size: 20px;
    width: 38px;
    height: 38px;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s;
}
.toggle-btn:hover {
    background: #2d5a8e;
}

/* ── Main content area ── */
[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
}
.block-container {
    padding-top: 1rem !important;
    max-width: 100% !important;
}

/* ── Header ── */
.chat-header {
    font-family: 'Space Grotesk', sans-serif;
    color: #64ffda;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── Status chips ── */
.chip-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}
.chip {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 500;
}
.chip-green  { background: #0f3d2e; color: #4ade80; border: 1px solid #16a34a; }
.chip-blue   { background: #0e2a4a; color: #60a5fa; border: 1px solid #2563eb; }
.chip-purple { background: #2d1b4e; color: #c084fc; border: 1px solid #7c3aed; }

/* ── Chat bubbles ── */
.chat-box {
    background: #0b1628;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    min-height: 360px;
    max-height: 480px;
    overflow-y: auto;
    margin-bottom: 1rem;
    scroll-behavior: smooth;
}
.msg-user {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 8px 0 8px auto;
    max-width: 75%;
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
    line-height: 1.5;
}
.msg-bot-icon {
    font-size: 16px;
    margin-right: 6px;
    vertical-align: middle;
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
.empty-state {
    text-align: center;
    color: #334155;
    margin-top: 80px;
}
.empty-state-icon { font-size: 48px; display: block; margin-bottom: 10px; }
.empty-state-text { font-size: 16px; }
.empty-state-text span { color: #64ffda; font-weight: 600; }

/* ── Input area ── */
[data-testid="stTextInput"] input {
    background: #0b1628 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
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
    font-weight: 600 !important;
    height: 48px;
    min-width: 52px;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #065f46 !important;
}

/* ── Sidebar content ── */
.sidebar-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #64ffda;
    line-height: 1.2;
    padding: 0.5rem 0 0.2rem 0;
}
.sidebar-sub {
    font-size: 11px;
    color: #64748b;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.sidebar-desc {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}
.sidebar-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 1rem 0 0.5rem 0;
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
    cursor: pointer;
}
.try-item {
    font-size: 12px;
    color: #64748b;
    padding: 4px 0;
    cursor: pointer;
}
.try-arrow { color: #64ffda; margin-right: 4px; }
.sidebar-footer {
    font-size: 11px;
    color: #334155;
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1e3a5f;
}

/* ── Streamlit override ── */
.stButton button { background: transparent; border: none; }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
""", unsafe_allow_html=True)

## ── SESSION STATE ─────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "user_name"   not in st.session_state: st.session_state.user_name   = ""
if "waiting_for" not in st.session_state: st.session_state.waiting_for = ""
if "sidebar_open" not in st.session_state: st.session_state.sidebar_open = True

## ── PERSISTENT TOGGLE BUTTON (injected via HTML — always visible) ──
st.markdown("""
<div id="sidebar-toggle-container">
  <button class="toggle-btn" onclick="toggleSidebar()" title="Toggle sidebar">☰</button>
</div>
<script>
function toggleSidebar() {
    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
    const btn     = window.parent.document.querySelector('[data-testid="collapsedControl"]');
    if (sidebar) {
        const isVisible = sidebar.style.display !== 'none' &&
                          !sidebar.classList.contains('st-emotion-cache-hidden');
        if (btn) btn.click();
    }
}
// Auto-show sidebar on load
window.parent.document.addEventListener('DOMContentLoaded', () => {
    const ctrl = window.parent.document.querySelector('[data-testid="collapsedControl"]');
    if (ctrl) ctrl.style.display = 'none';
});
</script>
""", unsafe_allow_html=True)

## ── HELPER: TOGGLE via session state + rerun ──────────────────
## The toggle button below is the TRUE fix — a real Streamlit button
## in a fixed-position container achieved via CSS injection.
## It stays visible because it is NOT inside st.sidebar.

col_toggle, col_spacer = st.columns([1, 20])
with col_toggle:
    if st.button("☰", key="sidebar_toggle", help="Toggle sidebar"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

## ── RAG FUNCTIONS ─────────────────────────────────────────────
def load_and_chunk(path: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = []
    for section in raw.split("---"):
        section = section.strip()
        if not section or section.startswith("#"):
            continue
        sentences = [s.strip() for s in section.split("\n")
                     if s.strip() and not s.startswith("TOPIC:")]
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


def retrieve(query: str, index, chunks, embed_model, top_k: int = TOP_K):
    q = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, top_k)
    return [chunks[i] for s, i in zip(scores[0], indices[0]) if i != -1 and s > 0.1]


def rule_based(user_input: str) -> str | None:
    name   = st.session_state.user_name
    wf     = st.session_state.waiting_for
    now    = datetime.now().strftime("%H:%M")
    inp    = user_input.strip()

    if wf == "name":
        n = inp.replace("my name is ", "").title() if inp.startswith("my name is ") else inp.title()
        st.session_state.user_name   = n
        st.session_state.waiting_for = ""
        return f"What a beautiful name, {n}! How can I help you?"

    if wf == "location":
        st.session_state.waiting_for = ""
        return f"Oh {inp.title()}! That sounds like a wonderful place 🌍"

    if wf == "hobby":
        st.session_state.waiting_for = ""
        return f"That's so cool that you enjoy {inp.title()}! 🎉"

    low = inp.lower()

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
        return ("You can ask me anything from my knowledge base!\n"
                "Topics: Python, ML, Data Science, NLP, RAG, Vector Databases.\n"
                "Or just say hello! 😊")

    return None


## ── SIDEBAR (conditional on session state) ────────────────────
if st.session_state.sidebar_open:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">DecodeLab<br>AI Chatbot</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">AI Engineer · RAG System</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-desc">Ask me anything about machine learning, AI, data science, and Python.</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Topics</div>', unsafe_allow_html=True)
        st.markdown("""
        <div>
          <span class="topic-pill">Machine Learning</span>
          <span class="topic-pill">Deep Learning</span>
          <span class="topic-pill">NLP</span>
          <span class="topic-pill">RAG</span>
          <span class="topic-pill">Transformers</span>
          <span class="topic-pill">FAISS</span>
          <span class="topic-pill">Python</span>
          <span class="topic-pill">Data Science</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Try Asking</div>', unsafe_allow_html=True)
        suggestions = [
            "What is machine learning?",
            "Explain neural networks",
            "What is RAG?",
            "How does backpropagation work?",
        ]
        for s in suggestions:
            st.markdown(f'<div class="try-item"><span class="try-arrow">→</span>{s}</div>',
                        unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-footer">
            Built with 🤍 by Katlego Mathebula
        </div>
        """, unsafe_allow_html=True)

## ── MAIN CHAT AREA ────────────────────────────────────────────
st.markdown('<div class="chat-header">Chat Interface</div>', unsafe_allow_html=True)

## Status chips
num_chunks = 0
try:
    _, _, chunks_loaded = load_models_and_index()
    num_chunks = len(chunks_loaded)
    rag_status = f'<span class="chip chip-green">● RAG Ready</span>'
    chunk_status = f'<span class="chip chip-blue">● {num_chunks} Chunks Indexed</span>'
except Exception:
    rag_status   = '<span class="chip chip-purple">● Loading...</span>'
    chunk_status = ''

st.markdown(f"""
<div class="chip-row">
    {rag_status}
    {chunk_status}
    <span class="chip chip-purple">● FAISS · cosine similarity</span>
</div>
""", unsafe_allow_html=True)

## Chat history display
chat_html = '<div class="chat-box" id="chat-box">'
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
            if "sources" in msg and msg["sources"]:
                for chunk in msg["sources"]:
                    preview = chunk[:55].replace("<", "&lt;") + "..."
                    sources_html += f'<span class="source-chip">{preview}</span>'
            chat_html += f'<div class="msg-bot"><span class="msg-bot-icon">🤖</span>{content}{("<br><br>" + sources_html) if sources_html else ""}</div>'

chat_html += '</div>'
chat_html += '<script>var cb = document.getElementById("chat-box"); if(cb) cb.scrollTop = cb.scrollHeight;</script>'
st.markdown(chat_html, unsafe_allow_html=True)

## Input form
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([10, 1])
    with col_input:
        user_input = st.text_input(
            label="message",
            label_visibility="collapsed",
            placeholder="Ask me about AI, ML, Data Science...",
            key="user_input",
        )
    with col_send:
        submitted = st.form_submit_button("↑")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    with st.spinner(""):
        rule_reply = rule_based(user_input.strip())

        if rule_reply:
            st.session_state.messages.append({
                "role": "assistant",
                "content": rule_reply,
                "sources": [],
            })
        else:
            try:
                embed_model, index, chunks = load_models_and_index()
                retrieved = retrieve(user_input.strip(), index, chunks, embed_model)
                if retrieved:
                    answer = " ".join(retrieved[:2])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved,
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't find relevant information in my knowledge base for that question.",
                        "sources": [],
                    })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, something went wrong loading the RAG system. Please try again.",
                    "sources": [],
                })

    st.rerun()
