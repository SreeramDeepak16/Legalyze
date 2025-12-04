import uuid
from datetime import datetime, timedelta
import os

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import create_agent


# ---------------------------------------------------------
# CHROMA SETUP
# ---------------------------------------------------------
EMBED = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CHROMA = Chroma(
    collection_name="episodic_memory",
    embedding_function=EMBED,
    persist_directory="episodic_chroma_store"
)


# ---------------------------------------------------------
# MEMORY MANAGER
# ---------------------------------------------------------
class EpisodicMemoryManager:

    def __init__(self):
        self.db = CHROMA

    def add_or_update(self, event_type=None, actor=None,
                      summary=None, details=None, event_id=None):

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

        # UPDATE EXISTING
        if event_id:
            self.db.add_texts(
                texts=[text],
                metadatas=[metadata],
                ids=[event_id]
            )
            return event_id

        # CREATE NEW MEMORY
        new_id = str(uuid.uuid4())
        self.db.add_texts(
            texts=[text],
            metadatas=[metadata],
            ids=[new_id]
        )
        return new_id

    # ---------------------------------------------------------
    # 🔥 RENAMED FROM `search()` TO `retrieve_memory()`
    # ---------------------------------------------------------
    def retrieve_memory(self, query):
        results = self.db.similarity_search(query, k=10)
        return [r.metadata for r in results]

    def list_all(self):
        data = self.db._collection.get()
        return data["metadatas"]

    def list_recent(self, days=7):
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
    # 🔥 UPDATED: now calls memory.retrieve_memory()
    results = memory.retrieve_memory(query)
    return [format_event(r) for r in results]


@tool
def list_recent_events(days: int = 7) -> list:
    """List events from the last X days."""
    events = memory.list_recent(days)
    return [format_event(e) for e in events]


@tool
def list_all_memories() -> list:
    """List all episodic memories."""
    events = memory.list_all()
    return [format_event(e) for e in events]


TOOLS = [
    add_or_update_memory,
    retrieve_memory,
    list_recent_events,
    list_all_memories
]


# ---------------------------------------------------------
# SYSTEM PROMPT
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


# ---------------------------------------------------------
# LLM SETUP
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyBdAni72s8pWLVcA_bBzH-uIa5aOleYxjY"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ---------------------------------------------------------
# AGENT (LANGCHAIN 0.3+)
# ---------------------------------------------------------
agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT
)


# ---------------------------------------------------------
# RUN FUNCTION
# ---------------------------------------------------------
def run_agent(text: str):
    inputs = {"messages": [{"role": "user", "content": text}]}
    result = agent.invoke(inputs)
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "No response."


# ---------------------------------------------------------
# TEST CALLS
# ---------------------------------------------------------
if __name__ == "__main__":
    print(run_agent("I met Arjun today and we ate biryani."))
    print(run_agent("Retrieve memories about Arjun"))
    print(run_agent("What happened today?"))
    print(run_agent("Show all memories"))
