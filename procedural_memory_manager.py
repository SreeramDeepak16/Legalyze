from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from typing import List
import uuid

load_dotenv()  # harmless; real key should come from app entrypoint

# ---------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------
class proceduralSchema(BaseModel):
    entry_type: str = Field(description='The entry type like workflow, guide, etc.')
    description: str = Field(description='A brief summary of the procedure')
    steps: List[str] = Field(description='List where each string is a step of the procedure')


# ---------------------------------------------------------
# Procedural Memory Manager (WORKS WITH LC 0.2.x)
# ---------------------------------------------------------
class ProceduralMemoryManager:

    def __init__(self):
        self.llm = None
        self.embed = None
        self.db = None
        self.parser = JsonOutputParser(pydantic_object=proceduralSchema)
        self._clients_initialized = False

        self.prompt = PromptTemplate(
            template='''
        Convert the user input into the following JSON schema:

        {
    "entry_type": "<type>",
    "description": "<short brief of the process>",
    "steps": [
        "1. <First step>",
        "2. <Second step>",
        "3. <and so on>"
    ]
    }

    Rules:
    1. entry_type should be inferred based on the input (workflow, recipe, instruction guide, etc.) 
    2. description must summarize the whole procedure in one line.
    3. Make steps numbered (“1.”, “2.”, “3.” …)
    4. Each step should be an actionable instruction.
    5. If input is not explicitly procedural, infer the likely steps.
    6. Output ONLY a JSON object — no extra text.

    INPUT:
    {input}

    OUTPUT:
    JSON only.
''',  
    # keep your original prompt; shortened here for brevity
            input_variables=['input']
        )

        self.chain = None

    def _init_clients(self):
        if self._clients_initialized:
            return
        load_dotenv()

        try:
            self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
            self.embed = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
            self.db = Chroma(
                collection_name='procedural_memory',
                persist_directory="procedural_db",
                embedding_function=self.embed
            )
            self.chain = LLMChain(prompt=self.prompt, llm=self.llm, output_parser=self.parser)
        except Exception:
            # leave as None and let callers handle
            self.llm = None
            self.embed = None
            self.db = None
            self.chain = None
        self._clients_initialized = True

    # -----------------------------------------------------
    # Convert free text into procedural JSON
    # -----------------------------------------------------
    def get_formatted_data(self, text: str) -> proceduralSchema:
        self._init_clients()
        if not self.chain:
            raise RuntimeError("LLM chain not initialized")
        result = self.chain.run({"input": text})
        if isinstance(result, dict):
            return proceduralSchema(**result)
        return proceduralSchema(**result)

    # -----------------------------------------------------
    # Add data to vector DB
    # -----------------------------------------------------
    def add_or_update(self, query: str):
        self._init_clients()
        if self.db is None:
            raise RuntimeError("Vector DB not initialized")
        formatted = self.get_formatted_data(query)
        json_str = formatted.model_dump_json()

        text = (
            formatted.description + "\n" +
            "\n".join(formatted.steps) + "\n" +
            json_str
        )

        self.db.add_texts(
            texts=[text],
            metadatas=[{"json": json_str}],
            ids=[str(uuid.uuid4())]
        )

        return formatted.model_dump()

    # -----------------------------------------------------
    # Retrieve most relevant stored procedure
    # -----------------------------------------------------
    def retrieve(self, query: str, threshold: float = 0.6):
        self._init_clients()
        if not self.db:
            return None
        results = self.db.similarity_search_with_relevance_scores(query, k=1)
        if not results:
            return None

        doc, score = results[0]
        if score < threshold:
            return None

        stored_json = doc.metadata["json"]
        return proceduralSchema.model_validate_json(stored_json).model_dump()
