import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from typing import Dict

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain


# ---------------------------------------------------------
# Semantic memory schema
# ---------------------------------------------------------
class Semantic(BaseModel):
    name: str = Field(description="The name/key of the semantic memory")
    summary: str = Field(description="A short description of the concept")
    details: str = Field(description="More detailed information")
    source: str = "user"


# ---------------------------------------------------------
# Semantic Memory Manager (Working for LC 0.2.x)
# ---------------------------------------------------------
class SemanticMemoryManager:

    def __init__(self, db_path="./semanticdb"):
        self._clients_initialized = False
        self.llm = None
        self.embedding_model = None
        self.vectorstore = None
        self.parser = PydanticOutputParser(pydantic_object=Semantic)
        self.chain = None
        self.db_path = db_path

        # Prepare prompt template (parser instructions injected later)
        prompt_template = """
Extract ONE semantic fact from the user's message
and return ONLY valid JSON in the schema below:

{name, summary, details, source}

User message:
{message}

{format_instructions}
"""
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["message", "format_instructions"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def _init_clients(self):
        if self._clients_initialized:
            return
        load_dotenv()
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
            )

            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )

            self.vectorstore = Chroma(
                collection_name="semantic_memory",
                embedding_function=self.embedding_model,
                persist_directory=self.db_path
            )

            self.chain = LLMChain(llm=self.llm, prompt=self.prompt, output_parser=self.parser)
        except Exception:
            self.llm = None
            self.embedding_model = None
            self.vectorstore = None
            self.chain = None
        self._clients_initialized = True

    # -----------------------------------------------------
    # Add new semantic memory
    # -----------------------------------------------------
    def add_or_update(self, user_message: str) -> Dict:
        self._init_clients()
        if not self.chain:
            raise RuntimeError("LLM chain not initialized")

        parsed = self.chain.run({"message": user_message})

        # Parse into pydantic object
        if isinstance(parsed, dict):
            semantic = Semantic(**parsed)
        else:
            semantic = Semantic(**parsed)

        store_text = (
            f"Name: {semantic.name}\n"
            f"Summary: {semantic.summary}\n"
            f"Details: {semantic.details}"
        )

        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized")

        self.vectorstore.add_texts(
            texts=[store_text],
            metadatas=[semantic.model_dump()]
        )

        return semantic.model_dump()

    # -----------------------------------------------------
    # Retrieve closest match
    # -----------------------------------------------------
    def retrieve(self, question: str, threshold: float = 0.6):
        self._init_clients()
        if not self.vectorstore:
            return None

        results = self.vectorstore.similarity_search_with_relevance_scores(question, k=1)

        if not results:
            return None

        doc, score = results[0]

        if score < threshold:
            return None

        return doc.metadata
