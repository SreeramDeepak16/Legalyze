import uuid
from datetime import datetime, timedelta
import os

from langchain.tools import tool
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import initialize_agent, AgentType

# ---------------------------------------------------------
# CHROMA SETUP (deferred)
# ---------------------------------------------------------
EMBED = None
CHROMA = None

# ---------------------------------------------------------
# MEMORY MANAGER
# ---------------------------------------------------------
class EpisodicMemoryManager:

    def __init__(self):
        load_dotenv()
        self.db = None
        self._clients_initialized = False

    def _init_clients(self):
        if self._clients_initialized:
            return
        load_dotenv()
        try:
            global EMBED, CHROMA
            if EMBED is None:
                EMBED = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            if CHROMA is None:
                CHROMA = Chroma(
                    collection_name="episodic_memory",
                    embedding_function=EMBED,
                    persist_directory="episodic_chroma_store"
                )

            self.db = CHROMA
        except Exception:
            self.db = None
        self._clients_initialized = True

    def add_or_update(self, event_type=None, actor=None,
                      summary=None, details=None, event_id=None):

        self._init_clients()
        occurred_at = datetime.utcnow().isoformat()

        text = f"""
Event Type: {event_type}
Actor: {actor}
Summary: {summary}
Details: {details}
Occurred At: {occurred_at}
""".strip()

        metadata = {
            "event_type": event_type,
            "actor": actor,
            "summary": summary,
            "details": details,
            "occurred_at": occurred_at
        }

        # UPDATE EXISTING MEMORY
        if event_id:
            if self.db is None:
                raise RuntimeError("Vector DB not initialized")
            self.db.add_texts(
                texts=[text],
                metadatas=[metadata],
                ids=[event_id]
            )
            return event_id

        # CREATE NEW MEMORY
        new_id = str(uuid.uuid4())
        if self.db is None:
            raise RuntimeError("Vector DB not initialized")
        self.db.add_texts(
            texts=[text],
            metadatas=[metadata],
            ids=[new_id]
        )
        return new_id

    def retrieve(self, query, threshold=0.5):
        self._init_clients()
        if not self.db:
            return None
        results = self.db.similarity_search_with_relevance_scores(query, k=3)
        if not results:
            return None
        doc, score = results[0]
        if score < threshold:
            return None
        return [doc.metadata for doc, _ in results]

    def list_all(self):
        self._init_clients()
        if not self.db:
            return []
        data = self.db._collection.get()
        return data["metadatas"]

    def list_recent(self, days=7):
        self._init_clients()
        if not self.db:
            return []
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff = cutoff.isoformat()
        data = self.db._collection.get()
        return [
            m for m in data["metadatas"]
            if m.get("occurred_at") >= cutoff
        ]


memory = EpisodicMemoryManager()

# ---------------------------------------------------------
# FORMATTER
# ---------------------------------------------------------
def format_event(meta):
    try:
        occur = datetime.fromisoformat(meta["occurred_at"])
        nice = occur.strftime("%b %d, %Y — %I:%M %p")
    except:
        nice = meta["occurred_at"]

    return f"""
🕒 {nice}
📌 Type: {meta.get("event_type")}
🧑 Actor: {meta.get("actor")}
📝 Summary: {meta.get("summary")}
📄 Details: {meta.get("details")}
""".strip()


# ---------------------------------------------------------
# TOOLS
# ---------------------------------------------------------
@tool
def add_or_update_memory(event_type: str = None,
                         actor: str = None,
                         summary: str = None,
                         details: str = None,
                         event_id: str = None) -> str:
    """Add or update an episodic memory event."""
    mem_id = memory.add_or_update(
        event_type=event_type,
        actor=actor,
        summary=summary,
        details=details,
        event_id=event_id
    )
    return f"Memory saved with ID: {mem_id}"


@tool
def retrieve_memory(query: str) -> list:
    """Retrieve episodic memories based on similarity search."""
    results = memory.retrieve(query)
    return [format_event(r) for r in results] if results else ["No matching memories found"]


@tool
def list_recent_events(days: int = 7) -> list:
    """List events from the last X days."""
    events = memory.list_recent(days)
    return [format_event(e) for e in events] if events else ["No recent memories"]


@tool
def list_all_memories() -> list:
    """List all episodic memories."""
    events = memory.list_all()
    return [format_event(e) for e in events] if events else ["No memories stored"]


TOOLS = [
    add_or_update_memory,
    retrieve_memory,
    list_recent_events,
    list_all_memories
]


# ---------------------------------------------------------
# SYSTEM PROMPT & AGENT SETUP
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an Episodic Memory Agent.

Rules:
- If the user describes an event → call add_or_update_memory.
- If user says "search", "find", "about", "retrieve" → call retrieve_memory.
- If user says "recent", "today", "week" → call list_recent_events.
- If user says "all events", "show all" → call list_all_memories.

Respond ONLY with the correct tool call.
"""

# LLM and agent initialization are left to runtime (avoid at import)
llm = None
agent = None

def init_agent_for_runtime(api_key_env_name="GOOGLE_API_KEY"):
    """
    Call this from your app entrypoint to initialize the LLM and agent
    after environment is guaranteed to be set.
    """
    global llm, agent
    load_dotenv()
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0
        )
        agent = initialize_agent(
            tools=TOOLS,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        # Inject custom system behavior
        agent.agent.llm_chain.prompt.messages[0].prompt.template = SYSTEM_PROMPT
    except Exception as e:
        llm = None
        agent = None
        raise

# ---------------------------------------------------------
# RUN FUNCTION
# ---------------------------------------------------------
def run_agent(text: str):
    try:
        if agent is None:
            init_agent_for_runtime()
        return agent.run(text)
    except Exception as e:
        return f"⚠ Error: {str(e)}"
