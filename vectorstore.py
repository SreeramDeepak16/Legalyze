import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")  # google short name or full (langchain wrapper may want short)
USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "0").lower() in ("1", "true", "yes")

# Imports for vector DB
from langchain_community.vectorstores import Chroma  # type: ignore
from langchain.docstore.document import Document  # type: ignore

def make_doc(text: str, metadata: dict):
    return Document(page_content=text, metadata=metadata)

# Local embedder implementation (object with embed_documents & embed_query)
class LocalSentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Lazy import to avoid requiring sentence-transformers when not using local mode
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # returns list[list[float]]
        vecs = self._model.encode(texts, show_progress_bar=False)
        # ensure lists (not numpy arrays)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

    def embed_query(self, text: str):
        vec = self._model.encode([text], show_progress_bar=False)[0]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

# Factory that tries Google then falls back to local
def _init_embedder_and_vectordb():
    # Option A: force local
    if USE_LOCAL:
        local_model = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        embedder = LocalSentenceTransformerEmbedder(model_name=local_model)
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)
        print("vectorstore: Using LOCAL sentence-transformers embedder:", local_model)
        return vectordb, embedder

    # Try Google generative embeddings first
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
        # NOTE: langchain_google_genai's class exposes embed_documents + embed_query
        embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        # do a minimal test (small query) to ensure credentials/quotas are OK
        try:
            _ = embedder.embed_query("test")
        except Exception as e:
            # Reraise to be caught by outer try so we fall back
            raise RuntimeError(f"Google embedder test failed: {e}") from e

        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)
        print("vectorstore: Using GoogleGenerativeAIEmbeddings (EMBED_MODEL=%s)" % EMBED_MODEL)
        return vectordb, embedder

    except Exception as e:
        # Fallback to local
        print("vectorstore: Google embeddings not available or failed (%s). Falling back to local embedder." % str(e))
        local_model = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            embedder = LocalSentenceTransformerEmbedder(model_name=local_model)
        except Exception as e2:
            # Last-resort tiny mock embedder (deterministic hash vectors) if sentence-transformers not installable
            print("vectorstore: Failed to initialize sentence-transformers (%s). Using mock embedder." % str(e2))
            class MockEmbedder:
                def embed_documents(self, texts):
                    import hashlib, struct
                    out = []
                    for t in texts:
                        h = hashlib.sha256(t.encode("utf-8")).digest()
                        # make 384-d floats from sha digest (just deterministic)
                        vec = []
                        for i in range(0, 48):
                            chunk = h[(i*3) % len(h):(i*3)%len(h)+3]
                            val = sum(b << (8*j) for j,b in enumerate(chunk)) % 1000
                            vec.append(float(val) / 1000.0)
                        out.append(vec)
                    return out

                def embed_query(self, text):
                    return self.embed_documents([text])[0]

            embedder = MockEmbedder()

        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)
        return vectordb, embedder

# Public initializer
_vdb, _embedder = _init_embedder_and_vectordb()

def init_vectorstore():
    """
    Returns (vectordb, embedder).
    vectordb is a Chroma instance and embedder implements:
      - embed_documents(list[str]) -> list[list[float]]
      - embed_query(str) -> list[float]
    """
    return _vdb, _embedder
