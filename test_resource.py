# -------------------------------------------------------
# TEST SCRIPT FOR ALL MEMORY MANAGERS
# -------------------------------------------------------

from meta_memory_manager import MetaMemoryManager

# If you decide to use LLM inside memory managers, uncomment:
# from langchain_google_genai import GoogleGenerativeAI

# -------------------------------------------------------
# SETUP
# -------------------------------------------------------
meta = MetaMemoryManager()

print("\n=============================================")
print(" 🔥 Unified Memory System Test Started ")
print("=============================================\n")

# -------------------------------------------------------
# TEST INPUT SET
# -------------------------------------------------------
test_queries = [
    # CORE / SEMANTIC
    "User's name is Supradeep.",
    "What is user's name?",

    # PROCEDURAL
    "Procedure to reset password: Open settings, click 'security', tap reset.",
    "How to reset password?",

    # EPISODIC
    "Today I played cricket at 11 am.",
    "What did I do at 11 am?",

    # RESOURCE MEMORY
    {"file_path": "./tortoise_rabbit_story.pdf"},
    "Give summary of tortoise and rabbit story.",

    # KNOWLEDGE MEMORY
    "The capital of Japan is Tokyo.",
    "What is the capital of Japan?"
]

# -------------------------------------------------------
# TEST LOOP
# -------------------------------------------------------

for i, query in enumerate(test_queries, 1):
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" 🧠 TEST {i} — INPUT: {query}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        result = meta.dispatch(query)
        print(f"OUTPUT:\n{result}")
    except Exception as e:
        print(f"⚠ ERROR processing query — {e}")

print("\n==================================================")
print(" 🎯 TEST COMPLETED FOR ALL MEMORY MANAGERS")
print("==================================================\n")
