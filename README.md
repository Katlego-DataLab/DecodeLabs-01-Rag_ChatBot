# 🤖 DecodeLab RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=for-the-badge)
![RAG](https://img.shields.io/badge/Architecture-RAG-blueviolet?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

> A hybrid AI chatbot combining rule-based logic with Retrieval Augmented Generation (RAG) — built raw in Python without LangChain, with a sleek dark Streamlit UI.

Built as part of the **DecodeLab Batch 2026 Internship Programme** by **Katlego Mathebula**.

---

## Demo

![DecodeLab AI Chatbot Screenshot](screenshot.png)

> Dark-themed chat interface with sidebar navigation, topic tags, and real-time semantic search across a custom AI/ML knowledge base.

---

##  What is RAG?

**Retrieval Augmented Generation (RAG)** is an AI architecture that enhances chatbot responses by:

1. **Retrieving** relevant documents from a knowledge base using semantic search
2. **Augmenting** the prompt with that retrieved context
3. **Generating** a grounded, factual answer

Unlike a standard chatbot that relies purely on training data, RAG grounds responses in real documents — reducing hallucinations and enabling domain-specific knowledge.

---

##  Architecture — Hybrid Two-Layer System

```
User Input
    │
    ▼
┌─────────────────────────┐
│   Layer 1: Rule-Based   │  ── match found? ──▶ respond directly
│   (greetings, identity) │
└─────────────────────────┘
    │ no match
    ▼
┌─────────────────────────┐
│   Sentence Transformer  │  Embeds query → dense vector
│   all-MiniLM-L6-v2      │
└─────────────────────────┘
    │ query vector
    ▼
┌─────────────────────────┐
│   FAISS Index           │  Cosine similarity → top-K chunks
│   IndexFlatIP           │
└─────────────────────────┘
    │ retrieved context chunks
    ▼
┌─────────────────────────┐
│   Answer Builder        │  Returns most relevant chunks
│   (retrieval-only RAG)  │  as the chatbot response
└─────────────────────────┘
    │
    ▼
Chatbot Response
```

**Layer 1 — Rule-Based:** Handles greetings, identity questions (`who built you?`), time, location, hobbies, and help commands. Fast and reliable for conversational interactions.

**Layer 2 — RAG:** Kicks in when no rule matches. Semantically searches the knowledge base and returns the most relevant information using vector similarity.

---

##  Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| UI Framework | `Streamlit` | Web interface and chat layout |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Converts text to dense vectors |
| Vector Search | `FAISS (IndexFlatIP)` | Fast cosine similarity search |
| Chunking | Sliding window (overlap=1) | Splits knowledge base into retrievable pieces |
| Styling | Custom CSS + Google Fonts | Dark navy UI with teal accents |
| Language | Python 3.11 | Core logic and pipeline |

---

## Libraries Explained

### `streamlit`
The web framework that powers the entire chatbot UI. Streamlit turns Python scripts into interactive web apps with no frontend code needed. Used here for the chat interface, sidebar, spinner, session state, and page config.

### `sentence-transformers`
A library built on top of HuggingFace Transformers that produces high-quality sentence embeddings. The model `all-MiniLM-L6-v2` converts text into 384-dimensional vectors that capture semantic meaning so "machine learning" and "ML algorithms" are understood as related even without matching keywords.

### `faiss-cpu`
Facebook AI Similarity Search — an open-source library for fast nearest-neighbour search in high-dimensional vector spaces. Used here with `IndexFlatIP` (inner product on normalised vectors = cosine similarity) to find the most semantically similar chunks to a user's query.

### `numpy`
The foundational numerical computing library in Python. Used here for array manipulation when working with embeddings and FAISS index operations.

### `transformers` *(installed but not used in final version)*
HuggingFace's library for working with pre-trained language models. Initially used for LLM-based answer generation with `google/flan-t5-base` — removed in the final version due to stability issues (see Challenges section below).

### `torch` *(dependency)*
PyTorch — the deep learning framework that powers both `sentence-transformers` and `transformers` under the hood.

---

## Project Structure

```
Project 1 -Rag_ChatBot/
│
├── app.py                    # ✅ Main Streamlit UI (final version)
├── rag_chatbot.py            # Terminal chatbot with full RAG pipeline
├── build_knowledge_base.py   # Pulls Wikipedia articles via HuggingFace Datasets
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── data/
    └── knowledge_base.txt    # Custom AI/ML knowledge base
```

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Katlego-DataLab/decodelab-rag-chatbot.git
cd decodelab-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
python -m streamlit run app.py
```

> Opens automatically at `http://localhost:8501`

### 4. Run the terminal chatbot (optional)
```bash
python rag_chatbot.py
```

---

##  Requirements

```
streamlit
sentence-transformers
faiss-cpu
numpy
transformers
torch
```

Install all at once:
```bash
pip install streamlit sentence-transformers faiss-cpu numpy transformers torch
```

---

##  Knowledge Base Format

Edit `data/knowledge_base.txt` to add your own topics. Use `---` to separate sections:

```
---
TOPIC: Your Topic Name
First fact or sentence about the topic.
Second fact here.
Third fact here.
---
```

The chunker uses a sliding window with overlap of 1 sentence, so no context is ever lost between chunks. Each chunk is embedded and stored in the FAISS index at startup.

**Current topics covered:**
- Python Programming
- Machine Learning
- Deep Learning
- Data Science
- Natural Language Processing (NLP)
- RAG (Retrieval Augmented Generation)
- Neural Networks
- Vector Databases

---

##  How the RAG Pipeline Works

### Step 1: Indexing (at startup)
```
knowledge_base.txt
    → split by "---" into sections
    → split sections into sentences
    → sliding window chunking (size=3, overlap=1)
    → SentenceTransformer encodes chunks → embeddings
    → FAISS normalises + stores embeddings
```

### Step 2: Querying (on each user message)
```
User query
    → SentenceTransformer encodes query → query vector
    → FAISS cosine similarity search → top-5 chunks (score > 0.1)
    → top chunks joined and returned as answer
```

---

## ⚠️ Challenges & Lessons Learned

This project involved significant debugging and iteration. Below is an honest account of every major issue encountered and how it was resolved.

### ❌ Problem 1: Wrong pipeline type for flan-t5
**Error:** The LLM was loaded with `pipeline("text-generation", ...)` but `google/flan-t5-base` is an encoder-decoder model that requires `pipeline("text2text-generation", ...)`

**Result:** The model echoed the entire prompt back in its response instead of generating a clean answer.

**Fix attempted:** Changed to `"text2text-generation"` — but this caused a new error.

---

### ❌ Problem 2: `text2text-generation` not recognised
**Error:**
```
KeyError: "Unknown task text2text-generation, available tasks are [...]"
```
**Cause:** The installed version of `transformers` was too new and had removed or reorganised the `text2text-generation` pipeline key.

**Fix attempted:** Reverted to `"text-generation"` with manual prompt stripping using `raw[len(prompt):]` instead of `.replace(prompt, "")`.

---

### ❌ Problem 3: `.replace(prompt, "")` causing crashes
**Error:** When the LLM output didn't contain the prompt verbatim, `.replace()` returned the full raw output including the prompt — making responses unreadable and causing downstream crashes.

**Fix:** Switched to slicing: `answer = raw[len(prompt):].strip()` which always correctly removes exactly the prompt portion.

---

### ❌ Problem 4: App crashing on second question
**Error:** `Connection error — Is Streamlit still running?`

**Cause:** The LLM (`flan-t5-base`) was consuming too much RAM and timing out on generation, causing the entire Streamlit process to die.

**Fix attempted:** Added `try/except` blocks around LLM calls to catch errors gracefully and fall back to retrieved chunks.

---

### ❌ Problem 5: Streamlit `secrets.toml` permission error
**Error:**
```
PermissionError: [Errno 13] Permission denied: '.streamlit\secrets.toml'
```
**Cause:** A corrupted or locked Streamlit secrets file in the project folder.

**Fix:** Deleted the file manually:
```bash
del ".streamlit\secrets.toml"
```

---

### ❌ Problem 6: `streamlit` not recognised as a command
**Error:**
```
streamlit : The term 'streamlit' is not recognized...
```
**Cause:** Streamlit was installed but its executable was not on the Windows PowerShell PATH.

**Fix:** Always use the module flag:
```bash
python -m streamlit run app.py
```

---

### Final Solution: Removed LLM entirely
After multiple LLM-related crashes, the decision was made to remove the LLM generation step and use retrieval-only RAG. The chatbot now returns the top retrieved chunks directly as answers.

**Result:** Zero crashes, fast responses, stable deployment, and still fully functional as a RAG system because the knowledge base chunks are already dense and informative.

**Lesson:** For local deployment on limited RAM, retrieval-only RAG is more robust than generation-based RAG with large models. LLM generation should be offloaded to an API (e.g. OpenAI, Anthropic, or HuggingFace Inference API) rather than run locally.

---

## Key Concepts Demonstrated

- **Semantic search** — finds meaning, not just keywords
- **Vector embeddings** — text converted to high-dimensional numerical vectors
- **FAISS cosine similarity** — fast nearest-neighbour search at scale
- **Sliding window chunking** — overlapping chunks preserve context across boundaries
- **Hybrid architecture** — rule-based layer + neural retrieval working together
- **Session state management** — Streamlit session state preserves conversation context
- **Streamlit caching** — `@st.cache_resource` ensures models load only once

---

## UI Design

The interface features a custom dark navy theme inspired by modern developer portfolios:

- **Background:** `#0f172a` (deep navy)
- **Accent:** `#64ffda` (teal green)
- **Typography:** Inter (body) + Space Grotesk (headings) via Google Fonts
- **Cursor glow effect:** Radial gradient that follows the mouse
- **Chat bubbles:** Gradient user bubbles + dark bot bubbles with source chips

---

##  Author

**Katlego Mathebula**
Junior Data Scientist & ML Engineer
GitHub: [Katlego-DataLab](https://github.com/Katlego-DataLab)
Portfolio: [katlego-datalab.github.io](https://katlego-datalab.github.io/Website-updated-/)

Built with 🤍 as part of the **DecodeLab Batch 2026 Internship Programme**

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
