from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os
load_dotenv()


class Feedback(BaseModel):
    complete : Literal['Enough','Not Enough'] = Field(description='Determine whether query is enough for accurate legal information retrieval or not.')

class FollowUpQuestions(BaseModel):
    ques_list : list = Field(description='All questions in string format in the form of a list')

class GeneratedQuery(BaseModel):
    new_query : str = Field(description='The generated legal information retrieval query')


class QueryAgent():
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key=os.getenv("QUERY_KEY"))

    def analyze_query(self, input_query : str) :
        parser = PydanticOutputParser(pydantic_object=Feedback)
        prompt = PromptTemplate(
            template='Analyze the given legal information retrieval query and determine if accurate information can be retrieved using this query alone. \n Query : {input_query} \n {format_instructions}',
            input_variables=['input_query'],
            partial_variables={'format_instructions':parser.get_format_instructions()}
        )
        analyze_chain = prompt | self.llm | parser
        result = analyze_chain.invoke(input_query)
        return result.complete
    
    def generate_follow_up_questions(self, incomplete_query : str) :
        parser = PydanticOutputParser(pydantic_object=FollowUpQuestions)
        prompt = PromptTemplate(
            template='Generate as less as possible follow-up questions on the given query in order to make it complete and enough for accurate legal information retrieval. \n Query : {query} \n {format_instructions}',
            input_variables=['query'],
            partial_variables={'format_instructions':parser.get_format_instructions()}
        )
        follow_up_chain = prompt | self.llm | parser
        result = follow_up_chain.invoke(incomplete_query)
        return result.ques_list
    
    def generate_new_query(self, prev_query : str, ques_list : list, ans_list : list) :
        parser = PydanticOutputParser(pydantic_object=GeneratedQuery)
        prompt = PromptTemplate(
            template='Generate a legal information retrieval query based on the previous query which was incomplete and the generated follow-up questions along with their answers. \n Previous query : {prev_query} \n Follow-up questions : {ques_list} \n Corresponding answers : {ans_list} \n {format_instructions}',
            input_variables=['prev_query','ques_list','ans_list'],
            partial_variables={'format_instructions':parser.get_format_instructions()}
        )
        generate_new_query_chain = prompt | self.llm | parser
        result = generate_new_query_chain.invoke({'prev_query':prev_query,'ques_list':ques_list,'ans_list':ans_list})
        return result.new_query
    
    def get_complete_query(self, query : str):
        sufficiency = self.analyze_query(query)
        if (sufficiency == 'Not Enough'):
            ques_list = self.generate_follow_up_questions(query)
            return {"sufficiency" : sufficiency, "follow_up_questions":ques_list}
        return {"sufficiency" : sufficiency}
            # ans_list = []
            # for ques in ques_list :
            #     ans = input(ques)
            #     ans_list.append(ans)
            # query = self.generate_new_query(query,ques_list,ans_list)











