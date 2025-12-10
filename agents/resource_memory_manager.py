import os
import re
from typing import Optional, Dict, Any, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from dotenv import load_dotenv
load_dotenv()


class ResourceMemoryManager:
    """
    MIRIX-Compatible Resource Memory Manager
    Works with MetaMemoryManager (supports add_or_update + retrieve)
    """

    def __init__(
        self,
        auto_summarize: bool = True,
    ) -> None:
        

        # Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("RESOURCE_KEY")
        )

        # Vector DB (Chroma)
        self._vectorstore = Chroma(
            collection_name='resource_memory',
            persist_directory='resource_db',
            embedding_function=self.embeddings,
        )

        # LLM for summarization
        self.auto_summarize = auto_summarize
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("RESOURCE_KEY")
        ) if auto_summarize else None

        # Chunking config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

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
        if not self.llm:
            return text[:200] + "..." if len(text) > 200 else text

        snippet = text[:4000]
        prompt = (
            "Summarize this document in 2–3 sentences:\n\n"
            f"{snippet}"
        )
        resp = self.llm.invoke(prompt)
        return resp.content.strip()

    def _chunk_and_store(self, text: str, metadata: dict) -> List[str]:

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
    # PUBLIC API — MIRIX REQUIRED METHODS
    # ----------------------------------------------------------------------

    def add_or_update(self, input_text: str) -> dict:
        """
        Accepts:
        - direct file path (“load ./resume.pdf”)
        - URL (“here is my github: https://github.com/xyz”)
        - text (“my notes: …. “)

        Stores the resource inside Chroma.
        """

        # 1) Detect file paths
        file_match = re.findall(r"[^\s]+\.(pdf|txt|md)", input_text, flags=re.I)
        if file_match:
            # Extract actual file path (complete string)
            file_path = re.search(r"[^\s]+\.(pdf|txt|md)", input_text).group(0)
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

        # 2) Detect URLs
        url_match = re.findall(r"https?://\S+", input_text)
        if url_match:
            url = url_match[0]
            text = f"External resource link: {url}"

            meta = {
                "title": url,
                "resource_type": "url",
                "url": url,
                "summary": f"Reference link stored: {url}"
            }

            ids = self._chunk_and_store(text, meta)
            return {"stored_ids": ids, "resource": meta}

        # 3) Store raw text as a resource
        meta = {
            "title": "text_snippet",
            "resource_type": "raw_text",
            "summary": self._summarize(input_text) if self.auto_summarize else ""
        }

        ids = self._chunk_and_store(input_text, meta)
        return {"stored_ids": ids, "resource": meta}

    def retrieve(self, query: str, k: int = 5, threshold : float = 0.6):
        """
        Retrieve resource text related to user query.
        """
        #return self._vectorstore.similarity_search(query, k=k)
        results = self._vectorstore.similarity_search_with_relevance_scores(query, k=1)

        if not results:
            return None

        doc, score = results[0]

        # higher score = more similar (1.0 = identical)
        if score > threshold:
            return doc

        return None

    def as_retriever(self):
        return self._vectorstore.as_retriever()