"""
knowledge_base.py
------------------
Builds a FAISS vector store from the sample "client website" documents
in /data (services, FAQ, pricing) using Gemini embeddings.

This is the RETRIEVAL half of the agentic chatbot: when the agent decides
a question is factual ("what does X cost", "how long does Y take"), it
calls `retrieve()` from this module to fetch the most relevant document
chunks, which are then passed to the LLM as grounding context.
"""

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Create a .env file (see .env.example) "
        "and set GEMINI_API_KEY=your_key_here."
    )


def _load_documents():
    """Load every .txt file in /data as a LangChain Document."""
    docs = []
    for file_path in sorted(DATA_DIR.glob("*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def build_vectorstore() -> FAISS:
    """
    Chunk the knowledge-base documents and embed them into a FAISS index.

    Chunking (via RecursiveCharacterTextSplitter) keeps each retrieved
    piece small and topic-focused, so the LLM gets concise, relevant
    context instead of an entire document dumped into the prompt.
    """
    documents = _load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY
    )

    return FAISS.from_documents(chunks, embeddings)


def retrieve(vectorstore: FAISS, query: str, k: int = 3) -> str:
    """
    Semantic search over the knowledge base. Returns the top-k relevant
    chunks concatenated into a single context string ready to drop into
    a prompt.
    """
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in results)
