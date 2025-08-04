from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

load_dotenv()

def build_qa_chain(persist_directory: str = "db") -> RetrievalQA:
    """
    Builds a QA chain using ChromaDB as retriever and OpenRouter's LLM for answering.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa
