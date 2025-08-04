from loader import load_pdf_text
from text_splitter import smart_split_text
from embedder import create_vector_store
from qa_engine import build_qa_chain
import os
import json

RAW_PDF_PATH = "data/raw/ai_guide.pdf"
CHUNKS_PATH = "data/processed/ai_guide_chunks.json"

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()

    print("🔍 Loading PDF:", RAW_PDF_PATH)
    text = load_pdf_text(RAW_PDF_PATH)

    print("✂️ Splitting text into chunks...")
    chunks = smart_split_text(text)
    print(f"✅ Total chunks: {len(chunks)}")

    # Save chunks for debug/inspection
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Embed and create vector store
    create_vector_store(chunks)
    print("📦 Vector store created.")

    # Build the QA system
    qa = build_qa_chain()

    while True:
        query = input("❓ Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = qa.invoke(query)
        print("💡 Answer:", answer)
