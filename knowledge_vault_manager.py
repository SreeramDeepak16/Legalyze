import uuid
from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma


# ---------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------
class knowledgeVaultSchema(BaseModel):
    entry_type: str = Field(description='Type of data')
    source: str = Field(description='Origin (github, aws, gcp, etc)')
    sensitivity: Literal['low', 'medium', 'high'] = Field(description='Sensitivity level')
    secret_value: str = Field(description='Actual sensitive data')
    caption: str = Field(description='One-line human-friendly description')


# ---------------------------------------------------------
# KNOWLEDGE VAULT MANAGER
# ---------------------------------------------------------
class KnowledgeVaultManager:

    def __init__(self):
        self.llm = None
        self.embed = None
        self.db = None
        self.parser = JsonOutputParser(pydantic_object=knowledgeVaultSchema)
        self.chain = None
        self._clients_initialized = False

        self.prompt = PromptTemplate(
            template="""
Convert the user input into the following JSON schema:

{
  "entry_type": "<type of data>",
  "source": "<origin of the credential>",
  "sensitivity": "<low | medium | high>",
  "secret_value": "<the extracted secret string>",
  "caption": "<a short description>"
}

Rules:
- Output ONLY the JSON object.
- Do not add backticks.
- Infer missing attributes if not mentioned.

INPUT:
{input}
""",
            input_variables=["input"]
        )

    def _init_clients(self):
        if self._clients_initialized:
            return
        load_dotenv()
        try:
            self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
            self.embed = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
            self.db = Chroma(
                collection_name="knowledge_memory",
                persist_directory="knowledge_db",
                embedding_function=self.embed
            )
            self.chain = LLMChain(prompt=self.prompt, llm=self.llm, output_parser=self.parser)
        except Exception:
            self.llm = None
            self.embed = None
            self.db = None
            self.chain = None
        self._clients_initialized = True

    # -----------------------------------------------------
    # Format user input into JSON object (schema)
    # -----------------------------------------------------
    def get_formatted_data(self, text: str) -> knowledgeVaultSchema:
        self._init_clients()
        if not self.chain:
            raise RuntimeError("LLM chain not initialized")
        result = self.chain.run({"input": text})
        if isinstance(result, dict):
            return knowledgeVaultSchema(**result)
        return knowledgeVaultSchema(**result)

    # -----------------------------------------------------
    # Add new memory
    # -----------------------------------------------------
    def add_or_update(self, query: str):
        self._init_clients()
        if not self.db:
            raise RuntimeError("Vector DB not initialized")

        formatted = self.get_formatted_data(query)
        json_str = formatted.model_dump_json()

        store_text = (
            formatted.entry_type + "\n" +
            formatted.source + "\n" +
            formatted.caption + "\n" +
            json_str
        )

        self.db.add_texts(
            texts=[store_text],
            metadatas=[{"json": json_str}],
            ids=[str(uuid.uuid4())]
        )
        return formatted.model_dump()

    # -----------------------------------------------------
    # Retrieve closest memory
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

        json_data = doc.metadata["json"]
        return knowledgeVaultSchema.model_validate_json(json_data).model_dump()
