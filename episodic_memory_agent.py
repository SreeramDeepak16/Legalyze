import uuid
from datetime import datetime, timedelta
import os
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------
# CHROMA SETUP (PERSISTENT VECTOR DB)
# ---------------------------------------------------------
EMBED = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CHROMA = Chroma(
    collection_name="episodic_memory",
    embedding_function=EMBED,
    persist_directory="episodic_chroma_store"
)


# ---------------------------------------------------------
# MEMORY MANAGER (PURE CHROMA + NO USER ID)
# ---------------------------------------------------------
class EpisodicMemoryManager:

    def __init__(self):
        self.db = CHROMA

    # ADD OR UPDATE MEMORY
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

        # UPDATE
        if event_id:
            self.db._collection.update(
                ids=[event_id],
                documents=[text],
                metadatas=[metadata]
            )
            self.db.persist()
            return event_id

        # NEW MEMORY
        new_id = str(uuid.uuid4())
        self.db.add(
            ids=[new_id],
            documents=[text],
            metadatas=[metadata]
        )
        self.db.persist()
        return new_id

    # SEARCH
    def search(self, query):
        results = self.db.similarity_search(query, k=10)
        return [r.metadata for r in results]

    # LIST ALL
    def list_all(self):
        data = self.db._collection.get()
        return data["metadatas"]

    # RECENT
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
    """Add or update an episodic memory."""
    mem_id = memory.add_or_update(
        event_type=event_type,
        actor=actor,
        summary=summary,
        details=details,
        event_id=event_id
    )
    return f"Memory saved with ID: {mem_id}"


@tool
def search_memory(query: str) -> list:
    """Search episodic memories."""
    results = memory.search(query)
    return [format_event(r) for r in results]


@tool
def list_recent_events(days: int = 7) -> list:
    """List recent episodic memories."""
    events = memory.list_recent(days)
    return [format_event(e) for e in events]


@tool
def list_all_memories() -> list:
    """List all episodic memories."""
    events = memory.list_all()
    return [format_event(e) for e in events]


TOOLS = [
    add_or_update_memory,
    search_memory,
    list_recent_events,
    list_all_memories
]


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are an Episodic Memory Agent.

Rules:
- If the user describes an event → call add_or_update_memory
- If user says "search", "about", "find" → call search_memory
- If user says "recent", "today", "this week" → call list_recent_events
- If user says "all events", "show all" → call list_all_memories

Always return ONLY the tool call.
"""


# ---------------------------------------------------------
# LLM + AGENT EXECUTOR
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyDg0Qrj_FVO7fRRlipMkBHBf8D0DXHQ5bQ"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    tools=TOOLS
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    ("human", "{input}")
])

agent = AgentExecutor.from_llm_and_tools(
    llm=llm,
    tools=TOOLS,
    prompt=prompt,
    verbose=True
)


# ---------------------------------------------------------
# RUNNER FUNCTION
# ---------------------------------------------------------
def run_agent(text):
    return agent.invoke({"input": text})


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
print(run_agent("I met Arjun today and we ate biryani."))
print(run_agent("Search memories about Arjun"))
print(run_agent("What happened today?"))
print(run_agent("Show all memories"))
