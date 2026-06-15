## ────────────────────────────────────────────────────
## DecodeLab RAG — Knowledge Base Builder
## Pulls real AI/ML/Tech articles from HuggingFace Wikipedia
## and converts them into knowledge_base.txt for the RAG chatbot
## 
## Author: Katlego Mathebula
## ────────────────────────────────────────────────────

from datasets import load_dataset
import re
import os

## ── CONFIG ────────────────────────────────────────────────────
OUTPUT_PATH   = "data/knowledge_base.txt"
MAX_ARTICLES  = 50        ## how many articles to pull (increase for richer knowledge)
MAX_CHARS     = 2000      ## max characters to keep per article (keeps chunks focused)

## AI/ML/Tech topics to filter from Wikipedia
AI_ML_TOPICS = [
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "transformer",
    "reinforcement learning",
    "computer vision",
    "artificial intelligence",
    "large language model",
    "generative ai",
    "retrieval augmented generation",
    "vector database",
    "word embedding",
    "gradient descent",
    "backpropagation",
    "convolutional neural network",
    "recurrent neural network",
    "attention mechanism",
    "bert",
    "gpt",
    "diffusion model",
    "random forest",
    "support vector machine",
    "data science",
    "feature engineering",
    "overfitting",
    "regularisation",
    "hyperparameter",
    "python programming",
    "scikit-learn",
    "tensorflow",
    "pytorch",
]

## ── HELPERS ───────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove wiki markup, citations, and extra whitespace."""
    text = re.sub(r'\[.*?\]', '', text)        ## remove citations [1], [2]
    text = re.sub(r'==+.*?==+', '', text)      ## remove section headers
    text = re.sub(r'\n{3,}', '\n\n', text)     ## collapse blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)    ## collapse spaces
    return text.strip()


def is_relevant(title: str, text: str) -> bool:
    """Check if the article is about an AI/ML/Tech topic."""
    combined = (title + " " + text[:500]).lower()
    return any(topic in combined for topic in AI_ML_TOPICS)


def truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """Keep only the intro section (most dense with definitions)."""
    ## Take first max_chars characters, cut at last full sentence
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_dot = truncated.rfind('.')
    return truncated[:last_dot + 1] if last_dot > 0 else truncated


## ── MAIN ──────────────────────────────────────────────────────
def build_knowledge_base():
    print("=" * 55)
    print("  DecodeLab — Knowledge Base Builder")
    print("  Source: Wikipedia (HuggingFace Datasets)")
    print("=" * 55)

    print("\n[⏳] Loading Wikipedia dataset in streaming mode...")
    print("     (streaming = no full download needed)\n")

    ## Stream Wikipedia, no need to download the whole 20GB dataset
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    os.makedirs("data", exist_ok=True)

    articles_saved = 0
    articles_scanned = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("## DecodeLab AI Knowledge Base\n")
        f.write("## Source: Wikipedia via HuggingFace Datasets\n")
        f.write("## Topics: AI, Machine Learning, Deep Learning, NLP, Data Science\n\n")

        for article in dataset:
            articles_scanned += 1

            title = article["title"]
            text  = article["text"]

            ## Filter to AI/ML/Tech articles only
            if not is_relevant(title, text):
                continue

            clean = clean_text(text)
            intro = truncate(clean)

            if len(intro) < 100:   ## skip stubs
                continue

            ## Write in the format the RAG chatbot expects
            f.write("---\n")
            f.write(f"TOPIC: {title}\n")
            f.write(intro + "\n")

            articles_saved += 1
            print(f"[✓] ({articles_saved}/{MAX_ARTICLES}) Saved: {title}")

            if articles_saved >= MAX_ARTICLES:
                break

            ## Print progress every 1000 scanned
            if articles_scanned % 1000 == 0:
                print(f"    [scanning... {articles_scanned} articles checked, {articles_saved} saved]")

    print(f"\n[✓] Done! {articles_saved} articles saved to '{OUTPUT_PATH}'")
    print(f"    Scanned {articles_scanned} Wikipedia articles total")
    print(f"\n[→] Now run: python rag_chatbot.py")


if __name__ == "__main__":
    build_knowledge_base()
