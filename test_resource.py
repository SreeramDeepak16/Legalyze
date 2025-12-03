from meta_memory_manager import MetaMemoryManager
from langchain_google_genai import GoogleGenerativeAI

from procedural_memory_manager import ProceduralMemoryManager
from resource_memory_manager import ResourceMemoryManager
from semantic_memory_manager import SemanticMemoryManager  


# -------------------------------------------------------
# LLM
# -------------------------------------------------------
llm = GoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    response_mime_type="application/json"
)


# -------------------------------------------------------
# AGENTS
# -------------------------------------------------------
resource_agent = ResourceMemoryManager(
    collection_name="resource_memory",
    persist_directory="./resource_db"
)

agents = {
    "semantic": SemanticMemoryManager(),
    "procedural": ProceduralMemoryManager(),
    "resource": resource_agent   # <-- FIXED (no parentheses)
}


# -------------------------------------------------------
# META MEMORY MANAGER
# -------------------------------------------------------
meta = MetaMemoryManager(llm, agents)


# -------------------------------------------------------
# TEST
# -------------------------------------------------------
print("\n meta memory manager results:")
result = meta.dispatch("what are the steps to restart linux?")
print(result)


# result = meta.dispatch({"file_path": "./tortoise_rabbit_story.pdf"})
# print(result)

# result = meta.dispatch("Give some information about tortoise and rabbit")
# print(result)
