import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "0").lower() in ("1", "true", "yes")


# -------------------------------------------------------
# Imports  (this is the correct import for your versions)
# -------------------------------------------------------
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # type: ignore


def make_doc(text: str, metadata: dict):
    return Document(page_content=text, metadata=metadata)


# -------------------------------------------------------
# Local embedder
# -------------------------------------------------------
class LocalSentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vecs = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

    def embed_query(self, text: str):
        vec = self._model.encode([text], show_progress_bar=False)[0]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)


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
        local_model = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        embedder = LocalSentenceTransformerEmbedder(model_name=local_model)
        vectordb = _make_chroma(embedder)
        print("vectorstore: Using LOCAL sentence-transformers embedder:", local_model)
        return vectordb, embedder

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
        embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        embedder.embed_query("test")  # sanity check

        vectordb = _make_chroma(embedder)
        print("vectorstore: Using GoogleGenerativeAIEmbeddings (model=%s)" % EMBED_MODEL)
        return vectordb, embedder

    except Exception as e:
        print("vectorstore: Google embeddings failed (%s). Falling back to local." % e)
        embedder = LocalSentenceTransformerEmbedder()
        vectordb = _make_chroma(embedder)
        return vectordb, embedder


_vdb, _embedder = _init_embedder_and_vectordb()


def init_vectorstore():
    return _vdb, _embedder
