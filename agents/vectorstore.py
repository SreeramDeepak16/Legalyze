import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./database/core_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "0").lower() in ("1", "true", "yes")


# -------------------------------------------------------
# Imports  (this is the correct import for your versions)
# -------------------------------------------------------
from langchain_chroma import Chroma
from langchain_core.documents import Document  # type: ignore


def make_doc(text: str, metadata: dict):
    return Document(page_content=text, metadata=metadata)


# -------------------------------------------------------
# Internal helper
# -------------------------------------------------------
def _make_chroma(embedder):
    return Chroma(
        collection_name="core_memory",
        persist_directory=CHROMA_DIR,
        embedding_function=embedder,
    )


# -------------------------------------------------------
# Main initialization logic
# -------------------------------------------------------
def _init_embedder_and_vectordb():
    if USE_LOCAL:
        embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
        vectordb = _make_chroma(embedder)
        return vectordb, embedder

    try:
        embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
        embedder.embed_query("test")  # sanity check

        vectordb = _make_chroma(embedder)
        return vectordb, embedder

    except Exception as e:
        embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
        vectordb = _make_chroma(embedder)
        return vectordb, embedder


_vdb, _embedder = _init_embedder_and_vectordb()


def init_vectorstore():
    return _vdb, _embedder
