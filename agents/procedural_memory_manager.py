from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from typing import List
import uuid
import os

from dotenv import load_dotenv
load_dotenv()

# {
#   "entry_type": "workflow",
#   "description": "Deploy application to production",
#   "steps": [
#     "1. Run test suite to ensure all tests pass",
#     "2. Create production build with 'npm run build'",
#     "3. Review build artifacts for any issues",
#     "4. Deploy to staging environment first",
#     "5. Perform smoke tests on staging",
#     "6. Deploy to production using CI/CD pipeline",
#     "7. Monitor application metrics post-deployment"
#   ]
# }

# input = ['''To bake a chocolate cake, start by mixing flour, eggs, and cocoa powder. Preheat the oven to 180 degrees Celsius. Pour the batter into a baking pan. Bake for 30 minutes. Let it cool and frost it.
# ''',
# '''To configure a Linux server, first update the package lists. Then install necessary dependencies. After that, configure firewall settings. Finally, test the server to ensure everything works.
# ''',
# '''First research different car models. Then visit a dealership and book a test drive. Compare multiple offers and negotiate with the seller. Finally, complete the paperwork and buy the car.
# ''',
# '''Begin by collecting all employee feedback. Then categorize feedback into themes. After that, create an improvement plan. Share the plan with the team and implement it.
# ''',
# '''To register for the event, visit the website. Fill the registration form. Make the payment. Receive confirmation on email.
# ''']


class proceduralSchema(BaseModel):
    entry_type : str = Field(description='The entry type of given procedure like workflow, guide, etc.')
    description : str = Field(description='A brief summary of the procedure')
    steps : List[str] = Field(description='List of string where each string is a step of the procedure')

class ProceduralMemoryManager():
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key=os.getenv("PROCEDURAL_KEY"))
        self.llm_embed = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001',google_api_key=os.getenv("PROCEDURAL_KEY"))
        self.db = Chroma(collection_name='procedural_memory',
                    persist_directory="procedural_db",
                    embedding_function=self.llm_embed)
        self.parser = JsonOutputParser(pydantic_object=proceduralSchema)
        

    def get_formatted_data(self, input : str):
        """Convert a procedural text into structured JSON."""
        prompt = PromptTemplate(
        template='''Your task is to convert any given input information into the following JSON schema:
            {{
                "entry_type": "<type>",
                "description": "<short brief of the process>",
                "steps": [
                    "1. <First sequential step>",
                    "2. <Second sequential step>",
                    "3. <and so on>"
                ]
            }}

            Rules:
            1. entry_type must be decided based on the type of input, for example, 'workflow', 'guide', etc.
            2. description should summarize the full process in one line.
            3. steps must be a numbered ordered list.
            4. Each step must begin with the step number and a period.
            5. Steps should be actionable instructions.
            6. Output must be valid JSON.
            7. Do not add extra keys or formatting.
            8. Infer steps if input information is not explicitly procedural.

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
            result = proceduralSchema(**result)
        return result
    

    def add_or_update(self, query : str):
        """Store the formatted procedural JSON into vector DB."""
        formatted_data = self.get_formatted_data(query)
        if isinstance(formatted_data, dict):
             formatted_data = proceduralSchema(**formatted_data)
        text = (
    formatted_data.description + "\n" +
    "\n".join(formatted_data.steps) + "\n" +
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


    






