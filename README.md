#  DecodeLab RAG Chatbot

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=for-the-badge)
![RAG](https://img.shields.io/badge/Architecture-RAG-blueviolet?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> A hybrid AI chatbot combining rule-based logic with Retrieval Augmented Generation (RAG) — built raw in Python without LangChain.

---

##  Architecture — Hybrid RAG

This chatbot uses a two-layer architecture:

**Layer 1 — Rule-Based** (your original chatbot logic)
Handles greetings, state tracking, identity questions, and personal chat.

**Layer 2 — RAG** (kicks in when no rule matches)
Semantically searches a custom knowledge base and generates a grounded answer using a HuggingFace LLM.

```
User input
    │
    ▼
┌─────────────────────┐
│   Rule-Based Layer  │  ── matches? ──▶ respond directly
└─────────────────────┘
    │ no match
    ▼
┌─────────────────────┐
│  Embedding Model    │  sentence-transformers/all-MiniLM-L6-v2
└─────────────────────┘
    │ query vector
    ▼
┌─────────────────────┐
│   FAISS Index       │  cosine similarity search → top-k chunks
└─────────────────────┘
    │ retrieved context
    ▼
┌─────────────────────┐
│   LLM (flan-t5)     │  context + query → grounded answer
└─────────────────────┘
    │
    ▼
Chatbot response
```

---

##  Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | `FAISS` (IndexFlatIP — cosine similarity) |
| Language Model | `google/flan-t5-base` (HuggingFace) |
| Framework | Raw Python — no LangChain |
| Chunking | Sliding window with overlap |

---

##  Project Structure

```
rag_chatbot/
│
├── rag_chatbot.py          # Main chatbot with full RAG pipeline
├── requirements.txt        # Dependencies
├── README.md               # This file
└── data/
    └── knowledge_base.txt  # Your custom dataset (replace with your own!)
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/decodelab-rag-chatbot.git
cd decodelab-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the chatbot
python rag_chatbot.py
```

> Models download automatically on first run and are cached locally.

---

## Using Your Own Data

Edit `data/knowledge_base.txt` — use the `---` separator between topics:

```
---
TOPIC: Your Topic Name
Your first fact or sentence here.
Your second fact here.
---
```

The chunker splits on `---` sections and uses a sliding window so no context is lost between chunks.

---

##  Key Concepts Demonstrated

- **Semantic search** — finds meaning, not just keywords
- **Vector embeddings** — text converted to high-dimensional vectors
- **FAISS cosine similarity** — fast nearest-neighbour search
- **Prompt engineering** — context-aware prompt assembly
- **Hybrid architecture** — rule-based + neural retrieval working together
- **State tracking** — remembers conversation context across turns

---

##  Author

**Katlego Mathebula**
Built with 🤍 as part of the DecodeLab Internship Programme
