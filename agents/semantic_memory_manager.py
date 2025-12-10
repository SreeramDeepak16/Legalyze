from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()
# semantic memory is being stored in databse like the below structure
class Semantic(BaseModel):
    name: str
    summary: str
    details: str
    source: str = "user"

# class of SemanticMemoryManager
class SemanticMemoryManager:

    def __init__(self, db_path="./semanticdb"):

        self.llm = GoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=os.getenv("SEMANTIC_KEY")
        )
        # Embedding model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("SEMANTIC_KEY")
        )
        # Vector database
        self.vectorstore = Chroma(
            collection_name="semantic_memory",
            embedding_function=self.embedding_model,
            persist_directory=db_path
        )
        # Parser for LLM to Semantic object
        self.parser = PydanticOutputParser(pydantic_object=Semantic)
        # Prompt to extract exactly 1 semantic fact
        template = """
Extract one semantic fact from the user's message.
Return ONLY valid JSON.
{format_instructions}
User message:
{message}
"""

        self.prompt = PromptTemplate(
            input_variables=["message", "format_instructions"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template=template
        )

        print("SemanticMemoryManager READY.")
        

    # add or update semantic memory
    def add_or_update(self, user_message: str) -> dict:
        """
        Extracts a semantic fact from user_message.
        If a fact with the same name exists then update it.
        leda insert a new memory.
        """

        final_prompt = self.prompt.format(
            message=user_message,
            format_instructions=self.parser.get_format_instructions()
        )

        response = self.llm.invoke(final_prompt)
        new_semantic = self.parser.parse(response)
        print("New name:",new_semantic.name)
        # # 2. Check if memory with same name exists 
        # existing = self._find_by_name(new_semantic.name)

        # if existing:
        #     # delete old record
        #     self.vectorstore.delete(ids=[existing["id"]])

        # 3. Store semantic fact in vector database
        embed_text = (
            f"Name: {new_semantic.name}\n"
            f"Summary: {new_semantic.summary}\n"
            f"Details: {new_semantic.details}"
        )
       
        self.vectorstore.add_documents([
            Document(
                page_content=embed_text,
                metadata=new_semantic.model_dump()
            )
        ])

        return new_semantic.model_dump()

    # retrieve
    def retrieve(self, question: str, threshold: float = 0.6):
        """
        Retrieves the closest semantic memory to the query.
        Returns None if the threshhold value is greater then 1.2.
        """

        results = self.vectorstore.similarity_search_with_relevance_scores(question, k=1)
        if not results:
            return None

        doc, score = results[0]

        if score < threshold:
            return None
        return doc.metadata

    # #we are here finding memory by name
    # def _find_by_name(self, name: str):
    #     """
    #     Search for a semantic memory by its name.
    #     Correctly reads Chroma get() response format.
    #     """
    #     data = self.vectorstore.get()  # returns dictonary with ids, metadatas, documents

    #     ids = data.get("ids", [])
    #     metadatas = data.get("metadatas", [])

    #     for idx, metadata in zip(ids, metadatas):
    #         if metadata.get("name") == name:
    #             return {"id": idx, "metadata": metadata}

    #     return None
