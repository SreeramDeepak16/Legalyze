from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import Literal
import uuid
from dotenv import load_dotenv
load_dotenv()

{
  "entry_type": "credential",
  "source": "github",
  "sensitivity": "high",
  "secret_value": "ghp_xxxxxxxxxxxxxxxxxxxx",
  "caption": "GitHub Personal Access Token for API access"
}

class knowledgeVaultSchema(BaseModel):
    entry_type : str = Field(description='Type of data')
    source : str = Field(description='')
    sensitivity : Literal['low','medium','high'] = Field(description='Sensitivity of the given data')
    secret_value : str = Field(description='The actual data that the user mentioned')
    caption : str = Field(description='A single line description of what the data is')

class KnowledgeVaultManager():
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
        self.llm_embed = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
        self.db = Chroma(collection_name='knowledge_memory',
                    persist_directory="knowledge_db",
                    embedding_function=self.llm_embed)
        self.parser = JsonOutputParser(pydantic_object=knowledgeVaultSchema)

    def get_formatted_data(self, input : str):
        prompt = PromptTemplate(
        template='''Your task is to convert any given input information into the following JSON schema:
{{
    "entry_type": <type of data>,
    "source": "<origin of the credential, e.g., github, aws, gcp>",
    "sensitivity": "<low | medium | high>",
    "secret_value": "<the extracted or transformed sensitive value>",
    "caption": "<brief human-readable description of the credential>"
}}

Rules:
1. entry_type must be inferred from the type of data like credential, password, etc.
2. source must be inferred from the input, such as "github", "aws", or another provider.
3. sensitivity must be inferred based on the nature of the information; tokens, passwords, and keys are "high".
4. secret_value must contain the core token/key/password or the primary confidential string extracted from the input.
5. caption should summarize the purpose or meaning of the secret in one short line.
6. Output must be valid JSON.
7. Do not add extra keys or formatting.
8. Infer missing attributes when not explicitly provided.

Input:
{input}

Output:
Return ONLY the JSON object in the exact schema above.
''',
            input_variables=['input']
        )
        chain = prompt | self.llm | self.parser
        result = chain.invoke({'input':input})
        if isinstance(result, dict):
            result = knowledgeVaultSchema(**result)
        return result
    

    def add_or_update(self, query : str):
        formatted_data = self.get_formatted_data(query)
        if isinstance(formatted_data, dict):
             formatted_data = knowledgeVaultSchema(**formatted_data)
        text = (
                formatted_data.entry_type + "\n" +
                "\n".join(formatted_data.source) + "\n" +
                "\n".join(formatted_data.caption) + "\n" +
                formatted_data.model_dump_json()
        )

        self.db.add_texts(
            texts=[text],
            metadatas=[{'json':formatted_data.model_dump_json()}],
            ids=[str(uuid.uuid4())]
        )
        return formatted_data.model_dump()


    def retrieve(self, query: str, threshold: float = 0.6):
        results = self.db.similarity_search_with_relevance_scores(query, k=1)
        if not results:
            return None
        doc, score = results[0]
        if score > threshold:
            return doc.metadata["json"]
        return None