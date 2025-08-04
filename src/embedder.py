from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List

def create_vector_store(chunks: List[str], persist_directory: str = "db") -> Chroma:
    """
    Converts text chunks into vector embeddings using HuggingFace and stores them in ChromaDB.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vector_db.persist()
    return vector_db
