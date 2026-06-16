## ────────────────────────────────────────────────────
## DecodeLab RAG Chatbot, Built Raw (No LangChain)
## Author : Katlego Mathebula
## Stack  : HuggingFace Transformers + FAISS + Sentence Transformers
## ────────────────────────────────────────────────────

import os
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline

## ── CONFIG ────────────────────────────────────────────────────
DATA_PATH   = "data/knowledge_base.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "google/flan-t5-base"
TOP_K       = 5
CHUNK_SIZE  = 3


## ── STEP 1: LOAD AND CHUNK DOCUMENTS ─────────────────────────
def load_and_chunk(path: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = []
    for section in raw.split("---"):
        section = section.strip()
        if not section or section.startswith("#"):
            continue

        sentences = [s.strip() for s in section.split("\n") if s.strip() and not s.startswith("TOPIC:")]

        for i in range(0, len(sentences), max(1, chunk_size - 1)):
            chunk = " ".join(sentences[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)

    print(f"[✓] Loaded {len(chunks)} chunks from knowledge base")
    return chunks


## ── STEP 2: BUILD VECTOR INDEX ────────────────────────────────
def build_index(chunks: list[str], embed_model: SentenceTransformer):
    print("Embedding knowledge base, this only happens once...")
    embeddings = embed_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[✓] FAISS index built — {index.ntotal} vectors stored (dim={dim})")
    return index, embeddings


## ── STEP 3: RETRIEVE RELEVANT CHUNKS ─────────────────────────
def retrieve(query: str, index, chunks: list[str], embed_model: SentenceTransformer, top_k: int = TOP_K) -> list[str]:
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score > 0.1:
            retrieved.append(chunks[idx])

    return retrieved


## ── STEP 4: GENERATE ANSWER ───────────────────────────────────
def generate_answer(query: str, context_chunks: list[str], llm) -> str:
    if not context_chunks:
        return "I couldn't find relevant information in my knowledge base for that question."

    context = "\n".join(f"- {chunk}" for chunk in context_chunks)

    prompt = (
        f"You are a helpful AI assistant. Using the context below, answer the question "
        f"in full, complete sentences. Be detailed and informative.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Detailed answer:"
    )

    result = llm(prompt, max_new_tokens=300, truncation=True)
    answer = result[0].get("generated_text", "").strip() if isinstance(result, list) else str(result).strip()

    return answer if answer else "I found relevant context but couldn't form a clear answer."


## ── STEP 5: RULE-BASED LAYER ──────────────────────────────────
def rule_based_response(user_input: str, user_name: str, waiting_for: str) -> tuple[str | None, str, str]:
    now = datetime.now().strftime("%H:%M")

    if waiting_for == "name":
        user_name = user_input.replace("my name is ", "").title() if user_input.startswith("my name is ") else user_input.title()
        return f"What a beautiful name, {user_name}! How can I help you?", user_name, ""

    if waiting_for == "location":
        return f"Oh {user_input.title()}! That sounds like a wonderful place 🌍", user_name, ""

    if waiting_for == "hobby":
        return f"That's so cool that you enjoy {user_input.title()}! 🎉", user_name, ""

    if user_input in ["hi", "hello", "hey"]:
        return "Hello! I am DecodeLab RAG Chatbot 🤖\nWhat's your name?", user_name, "name"

    if any(kw in user_input for kw in ["your name", "who built you", "who made you", "who created you"]):
        return "I am DecodeLab RAG Chatbot, built with 🤍 by Katlego Mathebula!", user_name, ""

    if "how are you" in user_input:
        msg = f"I'm doing great, thanks for asking {user_name}!" if user_name else "I'm doing great, thanks for asking!"
        return msg, user_name, ""

    if "time" in user_input:
        return f"The current time is {now} ⏰", user_name, ""

    if "where are you from" in user_input:
        return "I'm from the digital world 💻 I exist to assist you!\nHow about you — where are you from?", user_name, "location"

    if "hobbies" in user_input or "what do you do for fun" in user_input:
        return "I love learning and helping people! 🚀\nWhat about you — what do you do for fun?", user_name, "hobby"

    if "help" in user_input:
        return (
            "You can ask me anything from my knowledge base!\n"
            "Topics I know about: Python, ML, Data Science, NLP, RAG, Vector Databases.\n"
            "Or just say hello! 😊"
        ), user_name, ""

    return None, user_name, waiting_for


## ── MAIN CHATBOT LOOP ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("       DecodeLab RAG Chatbot, Powered by HuggingFace")
    print("       Type 'bye' to exit")
    print("=" * 55)

    print("\nLoading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"[✓] Embedding model ready: {EMBED_MODEL}")

    print("Loading language model (this may take a minute first time)...")
    llm = pipeline("text2text-generation", model=LLM_MODEL)
    print(f"[✓] LLM ready: {LLM_MODEL}\n")

    chunks = load_and_chunk(DATA_PATH)
    index, _ = build_index(chunks, embed_model)

    user_name   = ""
    waiting_for = ""

    print("\n[✓] RAG system ready! Start chatting.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["bye", "goodbye", "exit", "quit"]:
            farewell = f"It was great chatting with you, {user_name}! 🤍" if user_name else "It was great chatting with you! 🤍"
            print(f"Chatbot: {farewell}")
            print("Chatbot: Wishing you an amazing day, bye for now!")
            break

        response, user_name, waiting_for = rule_based_response(
            user_input.lower(), user_name, waiting_for
        )

        if response:
            print(f"Chatbot: {response}\n")
            continue

        print("Chatbot: [searching knowledge base...]\n")

        retrieved_chunks = retrieve(user_input, index, chunks, embed_model)

        if retrieved_chunks:
            print(f"[DEBUG] Retrieved {len(retrieved_chunks)} relevant chunk(s):")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"  {i}. {chunk[:80]}...")
            print()

        answer = generate_answer(user_input, retrieved_chunks, llm)
        print(f"Chatbot: {answer}\n")


if __name__ == "__main__":
    main()
