from typing import Dict, Any, List, Union
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from agents.core_memory_manager import CoreMemoryManager
from agents.episodic_memory_manager import EpisodicMemoryManager
from agents.semantic_memory_manager import SemanticMemoryManager
from agents.procedural_memory_manager import ProceduralMemoryManager
from agents.resource_memory_manager import ResourceMemoryManager
from agents.knowledge_vault_manager import KnowledgeVaultManager
import os

from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------
# Pydantic schema for LLM classification output
# --------------------------------------------------------------
class MetaDecision(BaseModel):
    task_type: str
    memory_types: List[str]


class MetaMemoryManager:

    def __init__(self):
        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("MMM_KEY"))
        self.agents = {
            "core": CoreMemoryManager(),
            "episodic": EpisodicMemoryManager(),
            "semantic": SemanticMemoryManager(),
            "procedural": ProceduralMemoryManager(),
            "resource": ResourceMemoryManager(),
            "knowledge": KnowledgeVaultManager()
            }

        self.parser = PydanticOutputParser(pydantic_object=MetaDecision)

        # ------------------------------------------------------
        # FULL PROMPT
        # ------------------------------------------------------
        self.prompt = PromptTemplate(
            input_variables=["input_text", "format_instructions"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
            template='''
You are the META MEMORY CONTROLLER of a multi-memory AI system.

Your job is to classify the user's input into:
1. task_type: "retrieve" OR "add_or_update"
2. memory_types: choose one or more from:
   ["semantic", "episodic", "procedural", "core", "knowledge", "resource"]

────────────────────────────────────────────────────────
MEMORY TYPE DEFINITIONS
────────────────────────────────────────────────────────

1. CORE MEMORY  
Stores **stable identity** of the user.
Examples:
- “My name is Arjun”
- “I live in Bangalore”
- “I am a backend developer”

2. SEMANTIC MEMORY  
Stores stable knowledge, facts, preferences, or general statements.
Examples:
- “I prefer PostgreSQL.”
- “Python is interpreted.”

3. EPISODIC MEMORY  
Stores events tied to time or context.

4. PROCEDURAL MEMORY  
Stores HOW-TO information, workflows, steps.

5. KNOWLEDGE MEMORY (Sensitive Vault)
Stores passwords, API keys, credentials.

6. RESOURCE MEMORY  
Stores external files, references, links, and documents.
Examples:
- “Here is my resume.pdf”
- “This is notes.txt”
- “My GitHub link is https://github.com/xyz”

────────────────────────────────────────────────────────
TASK RULES
────────────────────────────────────────────────────────
• If user asks a question → task_type = "retrieve"
• If user provides new data → task_type = "add_or_update"

────────────────────────────────────────────────────────
OUTPUT FORMAT (STRICT JSON)
────────────────────────────────────────────────────────

{format_instructions}

USER INPUT:
{input_text}
'''
        )

        print("MetaMemoryManager READY")


    # --------------------------------------------------------------
    # MAIN DISPATCH FUNCTION
    # --------------------------------------------------------------
    def dispatch(self, input_data: Union[str, dict]) -> Dict[str, Any]:

        # =============================================================
        # 1. DIRECT FILE HANDLING (Streamlit or backend)
        # =============================================================
        if isinstance(input_data, dict) and "file_path" in input_data:
            file_path = input_data["file_path"]

            decision = {
                "task_type": "add_or_update",
                "memory_types": ["resource"]
            }

            return {
                "decision": decision,
                "agent_results": {
                    "update": {
                        "resource": self.agents["resource"].add_or_update(file_path)
                    }
                }
            }

        # =============================================================
        # 2. If input_data is a direct file path string
        # =============================================================
        if isinstance(input_data, str) and (
            input_data.lower().endswith(".pdf") or
            input_data.lower().endswith(".txt") or
            input_data.lower().endswith(".md")
        ):
            decision = {
                "task_type": "add_or_update",
                "memory_types": ["resource"]
            }

            return {
                "decision": decision,
                "agent_results": {
                    "update": {
                        "resource": self.agents["resource"].add_or_update(input_data)
                    }
                }
            }

        # =============================================================
        # 3. Otherwise treat as natural language input
        # =============================================================
        input_text = input_data

        final_prompt = self.prompt.format(
            input_text=input_text,
            format_instructions=self.parser.get_format_instructions()
        )

        raw = self.llm.invoke(final_prompt)
        decision: MetaDecision = self.parser.parse(raw)

        # keep only memory types we actually have agents for
        memory_types = [m for m in decision.memory_types if m in self.agents]

        results = {
            "decision": decision.model_dump(),
            "agent_results": {}
        }
        print(results['decision'])

        # -------------------------------------------------------------
        # RETRIEVAL
        # -------------------------------------------------------------
        if decision.task_type == "retrieve":
            if "resource" not in memory_types:
                memory_types.append("resource")
            results["agent_results"]["retrieval"] = {}
            print(memory_types)
            for mem_type in memory_types:
                try:
                    results["agent_results"]["retrieval"][mem_type] = \
                        self.agents[mem_type].retrieve(input_text)
                except Exception as e:
                    results["agent_results"]["retrieval"][mem_type] = {"error": str(e)}

        # -------------------------------------------------------------
        # ADD / UPDATE
        # -------------------------------------------------------------
        elif decision.task_type == "add_or_update":
            results["agent_results"]["update"] = {}
            for mem_type in memory_types:
                try:
                    results["agent_results"]["update"][mem_type] = \
                        self.agents[mem_type].add_or_update(input_text)
                except Exception as e:
                    results["agent_results"]["update"][mem_type] = {"error": str(e)}
        return results
