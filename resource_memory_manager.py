import os
import re
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

load_dotenv()  # harmless

class ResourceMemoryManager:

    def __init__(self, auto_summarize: bool = True):
        self._clients_initialized = False
        self.embeddings = None
        self._vectorstore = None
        self.llm = None
        self.auto_summarize = auto_summarize

        # Chunking Config (safe to create now)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

    def _init_clients(self):
        if self._clients_initialized:
            return
        load_dotenv()
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )

            self._vectorstore = Chroma(
                collection_name='resource_memory',
                persist_directory='resource_db',
                embedding_function=self.embeddings,
            )

            if self.auto_summarize:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                )
            else:
                self.llm = None
        except Exception:
            self.embeddings = None
            self._vectorstore = None
            self.llm = None
        self._clients_initialized = True

    # ----------------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------------

    def _load_text_from_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            return "\n\n".join(p.page_content for p in pages)

        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        raise ValueError(f"Unsupported file type: {ext}")

    def _infer_type(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        return {
            ".pdf": "pdf_text",
            ".txt": "text",
            ".md": "markdown"
        }.get(ext, "unknown")

    def _summarize(self, text: str) -> str:
        # summarization uses llm if available
        if not self.auto_summarize:
            return text[:200] + "..." if len(text) > 200 else text

        self._init_clients()
        if not self.llm:
            return text[:200] + "..." if len(text) > 200 else text

        snippet = text[:4000]
        prompt = (
            "Summarize this document in 2–3 sentences:\n\n"
            f"{snippet}"
        )
        # use .predict if available, else .invoke or .run depending on client
        try:
            result = self.llm.predict(prompt)
        except AttributeError:
            try:
                result = self.llm.invoke(prompt)
            except Exception:
                return snippet[:200] + "..."
        except Exception:
            return snippet[:200] + "..."
        return result.strip()

    def _chunk_and_store(self, text: str, metadata: dict) -> List[str]:
        self._init_clients()
        if not self._vectorstore:
            raise RuntimeError("Vectorstore not initialized")
        docs = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        ids = self._vectorstore.add_texts(
            texts=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs]
        )

        self._vectorstore.persist()
        return ids

    # ----------------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------------

    def add_or_update(self, input_text: str) -> dict:
        # 1) Check for file paths
        file_match = re.search(r"[^\s]+\.(pdf|txt|md)", input_text, flags=re.I)
        if file_match:
            file_path = file_match.group(0)
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            raw = self._load_text_from_file(file_path)

            meta = {
                "title": os.path.basename(file_path),
                "resource_type": self._infer_type(file_path),
                "file_path": os.path.abspath(file_path),
                "summary": self._summarize(raw) if self.auto_summarize else ""
            }

            ids = self._chunk_and_store(raw, meta)
            return {"stored_ids": ids, "resource": meta}

        # 2) Check for URLs
        url_match = re.search(r"https?://\S+", input_text)
        if url_match:
            url = url_match.group(0)

            meta = {
                "title": url,
                "resource_type": "url",
                "url": url,
                "summary": f"Reference link: {url}"
            }

            ids = self._chunk_and_store(f"External resource link: {url}", meta)
            return {"stored_ids": ids, "resource": meta}

        # 3) Raw text
        meta = {
            "title": "text_snippet",
            "resource_type": "raw_text",
            "summary": self._summarize(input_text) if self.auto_summarize else ""
        }

        ids = self._chunk_and_store(input_text, meta)
        return {"stored_ids": ids, "resource": meta}

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.6):
        self._init_clients()
        if not self._vectorstore:
            return None

        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=1)

        if not results:
            return None

        doc, score = results[0]

        if score < threshold:
            return None

        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "relevance": score
        }

    def as_retriever(self):
        self._init_clients()
        if not self._vectorstore:
            raise RuntimeError("Vectorstore not initialized")
        return self._vectorstore.as_retriever()
