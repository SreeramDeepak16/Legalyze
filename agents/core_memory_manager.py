import os
import time
import json
from dotenv import load_dotenv #type: ignore
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from agents.vectorstore import init_vectorstore, make_doc  # type: ignore

# configuration (can be set in .env)
MAX_BLOCKS = int(os.getenv("MAX_BLOCKS", 300))
REWRITE_THRESHOLD = float(os.getenv("REWRITE_THRESHOLD", 0.9))

# initialize vectorstore and embedder
_vdb, _embedder = init_vectorstore()

# create chat model (ensure short model name, not "models/...")
DEFAULT_GEMINI = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
chat = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI)


class CoreMemoryManager():
    """
    CoreMemoryManager stores two canonical high-priority documents:
      - type == "persona" -> stable identity, tone, behavior (one paragraph)
      - type == "human"   -> enduring facts/preferences (bullet points or short sentences)
    """

    def __init__(self, vectordb=_vdb, chat_model=chat):
        self.vdb = vectordb
        self.chat = chat_model

    def add_or_update(self, content: str, mtype: str = "fact"):
        """
        Add a memory document. mtype usually in {"fact","persona","human"}.
        After add, attempt rewrite if needed.
        """
        meta = {
            "type": mtype,
            "updated_at": int(time.time()),
        }
        doc = make_doc(content, meta)
        self.vdb.add_documents([doc])

        try:
            self.vdb.persist()
        except Exception:
            pass
        # check whether we need to compact the memories
        self.rewrite_if_needed()

    def retrieve(self, query: str, k: int = 5, thresold: float =0.5):
        """
        Query memories using vector similarity. Returns list[Document].
        """
        total = self.count()
        if total == 0:
            return []
        k = min(k, total)
        results = self.vdb.similarity_search_with_relevance_scores(query, k=k)
        if not results:
            return None
        doc,score = results[0]
        if score < thresold:
            return None
        return doc.page_content
    
    def count(self) -> int:
        """
        Count number of stored docs.
        """
        coll = self.vdb._collection.get(include=["metadatas"])
        c = len(coll["metadatas"])
        print(f"[DEBUG] CoreMemoryManager.count()->{c}")
        return len(coll["metadatas"])

    def _collect_user_docs(self, k: int = 200):
        """
        Collect up to k docs (used for rewrite). Returns list[Document].
        """
        return self.vdb.similarity_search("", k=k)

    def rewrite_if_needed(self):
        """
        If stored docs exceed the threshold, compress them into two canonical blocks:
          - persona: one paragraph describing stable identity / tone
          - human: bullet-list or short sentences of enduring facts/preferences
        """
        total = self.count()
        threshold = MAX_BLOCKS * REWRITE_THRESHOLD
        if total < threshold:
            return

        docs = self._collect_user_docs(k=200)
        if not docs:
            return

        merged = "\n".join(d.page_content for d in docs)

        system = SystemMessage(
            "You are a memory-compression assistant. "
            "Produce a JSON object with two keys exactly: "
            '"persona" (one short paragraph describing the stable identity, tone, and behavior) '
            'and "human" (a short list or short sentences of enduring facts/preferences). '
            "Return ONLY valid JSON and nothing else."
        )
        human = HumanMessage(
            "Compress the following facts into the required JSON form. Facts:\n\n" + merged
        )

        # get model output
        raw_resp = ""
        try:
            raw_resp = self.chat([system, human]).content
        except Exception:
            # if generation fails, bail out (leave existing docs intact)
            return

        # Try parsing JSON strictly, with defensive fallbacks
        persona_text = ""
        human_text = ""

        def _try_parse_json(s: str):
            try:
                return json.loads(s)
            except Exception:
                return None

        parsed = _try_parse_json(raw_resp)
        if parsed is None:
            # try extracting a JSON substring (between first '{' and last '}')
            try:
                start = raw_resp.index("{")
                end = raw_resp.rindex("}") + 1
                parsed = _try_parse_json(raw_resp[start:end])
            except Exception:
                parsed = None

        if isinstance(parsed, dict):
            persona_text = (parsed.get("persona") or "").strip()
            human_text = (parsed.get("human") or "").strip()
        else:
            # final fallback: treat entire model response as human facts (no persona)
            human_text = raw_resp.strip()

        # if nothing useful returned, abort (avoid deleting data)
        if not persona_text and not human_text:
            return

        # replace docs with canonical persona & human docs
        try:
            # empty filter → delete all docs
            self.vdb.delete(filter={})
        except Exception:
            pass

        now = int(time.time())
        docs_to_add = []
        if persona_text:
            docs_to_add.append(make_doc(persona_text, {"type": "persona", "updated_at": now}))
        if human_text:
            docs_to_add.append(make_doc(human_text, {"type": "human", "updated_at": now}))

        if docs_to_add:
            self.vdb.add_documents(docs_to_add)
            self.vdb.persist()
