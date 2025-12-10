# meta_memory_manager.py
import os
import importlib
from typing import Dict, Any, List, Union
from pydantic import BaseModel  # type:ignore
from langchain_core.prompts import PromptTemplate  # type:ignore
from langchain_core.output_parsers import PydanticOutputParser  # type:ignore
from langchain_google_genai import GoogleGenerativeAI  # type:ignore
from dotenv import load_dotenv

# --------------------------------------------------------------
# Pydantic schema for LLM classification output
# --------------------------------------------------------------
class MetaDecision(BaseModel):
    task_type: str
    memory_types: List[str]


# --------------------------------------------------------------
# helper to create lazy factories that import safely
# --------------------------------------------------------------
def _make_factory(module_name: str, class_name: str):
    """
    Returns a callable that will import `module_name` and instantiate `class_name`.
    If import or attribute fails, the callable will raise an informative ImportError.
    """
    def factory():
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            raise ImportError(f"Failed to import module '{module_name}': {type(e).__name__}: {e}")
        try:
            cls = getattr(mod, class_name)
        except AttributeError:
            # try to be helpful: list available attributes
            available = [a for a in dir(mod) if not a.startswith("_")]
            raise ImportError(
                f"Module '{module_name}' does not have attribute '{class_name}'. "
                f"Available attributes: {available}"
            )
        try:
            return cls()
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate {class_name} from {module_name}: {type(e).__name__}: {e}")
    return factory


# --------------------------------------------------------------
# MetaMemoryManager
# --------------------------------------------------------------
class MetaMemoryManager:
    def __init__(self):
        # Do NOT create the LLM immediately; lazy-init below
        self.llm = None
        self._llm_init_error: Union[str, None] = None

        # Agent factories: lazy-import / instantiate managers on demand
        # Make sure module names and class names exactly match your filenames & classes.
        self._agent_factories = {
            "core": _make_factory("core_memory_manager", "CoreMemoryManager"),
            "episodic": _make_factory("episodic_memory_manager", "EpisodicMemoryManager"),
            "semantic": _make_factory("semantic_memory_manager", "SemanticMemoryManager"),
            "procedural": _make_factory("procedural_memory_manager", "ProceduralMemoryManager"),
            "resource": _make_factory("resource_memory_manager", "ResourceMemoryManager"),
            "knowledge_vault": _make_factory("knowledge_vault_memory_manager", "KnowledgeVaultManager"),
        }
        self._agents: Dict[str, Any] = {}

        # Parser for meta decision from LLM
        self.parser = PydanticOutputParser(pydantic_object=MetaDecision)

        # Full prompt (strict JSON)
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

    # ------------------------------
    # Lazy LLM initializer
    # ------------------------------
    def _init_llm(self):
        """
        Initialize GoogleGenerativeAI lazily and store any init error.
        This loads .env just before creating the client to ensure GOOGLE_API_KEY is available.
        """
        if self.llm is not None or self._llm_init_error is not None:
            return

        # Load environment (harmless if already loaded)
        load_dotenv()

        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            self._llm_init_error = "GOOGLE_API_KEY not found in environment"
            return

        try:
            # create the LLM client
            self.llm = GoogleGenerativeAI(model="gemini-2.5-flash")
            self._llm_init_error = None
        except Exception as e:
            self.llm = None
            self._llm_init_error = f"LLM init failed: {type(e).__name__}: {e}"

    # ------------------------------
    # Lazy agent getter
    # ------------------------------
    def _get_agent(self, name: str):
        """
        Returns instantiated agent or raises an informative exception if the factory fails.
        """
        if name in self._agents:
            return self._agents[name]
        if name not in self._agent_factories:
            raise KeyError(f"No agent factory configured for '{name}'")
        factory = self._agent_factories[name]
        try:
            agent = factory()
        except Exception as e:
            # Raise the same exception so call-sites can see what went wrong.
            raise
        self._agents[name] = agent
        return agent

    # --------------------------------------------------------------
    # MAIN DISPATCH FUNCTION
    # --------------------------------------------------------------
    def dispatch(self, input_data: Union[str, dict]) -> Dict[str, Any]:
        # ensure LLM is initialized (and detect failures early)
        self._init_llm()
        if self.llm is None:
            # Helpful diagnostic
            has_key = bool(os.getenv("GOOGLE_API_KEY"))
            msg = "MetaMemoryManager: LLM not initialized."
            if self._llm_init_error:
                msg += " Reason: " + self._llm_init_error
            else:
                msg += " GOOGLE_API_KEY present: " + str(has_key)
            raise RuntimeError(msg)

        # =============================================================
        # 1. DIRECT FILE HANDLING (Streamlit or backend)
        # =============================================================
        if isinstance(input_data, dict) and "file_path" in input_data:
            file_path = input_data["file_path"]

            decision = {
                "task_type": "add_or_update",
                "memory_types": ["resource"]
            }

            try:
                agent = self._get_agent("resource")
                return {
                    "decision": decision,
                    "agent_results": {
                        "update": {
                            "resource": agent.add_or_update(file_path)
                        }
                    }
                }
            except Exception as e:
                return {
                    "decision": decision,
                    "agent_results": {
                        "update": {
                            "resource": {"error": str(e)}
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

            try:
                agent = self._get_agent("resource")
                return {
                    "decision": decision,
                    "agent_results": {
                        "update": {
                            "resource": agent.add_or_update(input_data)
                        }
                    }
                }
            except Exception as e:
                return {
                    "decision": decision,
                    "agent_results": {
                        "update": {
                            "resource": {"error": str(e)}
                        }
                    }
                }

        # =============================================================
        # 3. Otherwise treat as natural language input
        # =============================================================
        input_text = input_data

        # Format prompt & call meta LLM
        final_prompt = self.prompt.format(
            input_text=input_text,
            format_instructions=self.parser.get_format_instructions()
        )

        raw = self.llm.invoke(final_prompt)
        # parse into MetaDecision
        decision: MetaDecision = self.parser.parse(raw)

        # normalize memory type names (map "knowledge" -> "knowledge_vault")
        normalized = []
        for m in decision.memory_types:
            if m == "knowledge":
                normalized.append("knowledge_vault")
            else:
                normalized.append(m)
        # keep only memory types we actually have factories for
        memory_types = [m for m in normalized if m in self._agent_factories]

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
                    agent = self._get_agent(mem_type)
                except Exception as e:
                    results["agent_results"]["retrieval"][mem_type] = {"error": str(e)}
                    continue

                try:
                    results["agent_results"]["retrieval"][mem_type] = agent.retrieve(input_text)
                except Exception as e:
                    results["agent_results"]["retrieval"][mem_type] = {"error": str(e)}

        # -------------------------------------------------------------
        # ADD / UPDATE
        # -------------------------------------------------------------
        elif decision.task_type == "add_or_update":
            results["agent_results"]["update"] = {}
            for mem_type in memory_types:
                try:
                    agent = self._get_agent(mem_type)
                except Exception as e:
                    results["agent_results"]["update"][mem_type] = {"error": str(e)}
                    continue

                try:
                    results["agent_results"]["update"][mem_type] = agent.add_or_update(input_text)
                except Exception as e:
                    results["agent_results"]["update"][mem_type] = {"error": str(e)}
        print(results)
        return results
