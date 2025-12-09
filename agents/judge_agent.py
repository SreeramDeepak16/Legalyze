from langchain_core.messages import HumanMessage  # type:ignore
from typing import List  # type:ignore
from pydantic import BaseModel  # type:ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type:ignore
from run_search_agent import run_all
import json


class Judge_Memory:
    def __init__(self):
        self.memory = []

    def save_evaluation(self, evaluation):
        self.memory.append(evaluation)

    def get_last_3_evaluations(self):
        return self.memory[-3:]


class FakeSearchResult:
    def __init__(self, data):
        self.query = data.get("query")
        self.backend = data.get("backend")
        self.id = data.get("id")
        self.source = data.get("source")
        self.title = data.get("title")
        self.citation = data.get("citation")
        self.content = data.get("content")


class WrapResult:
    def __init__(self, data):
        self.query = data.get("query")
        self.backend = data.get("backend")
        self.id = data.get("id")
        self.source = data.get("source")
        self.title = data.get("title")
        self.citation = data.get("citation")
        self.content = data.get("content")


class JudgeResult(BaseModel):
    is_sufficient: bool
    reasoning: str
    source_quality: str
    date_check: str
    jurisdiction_check: str
    contradiction_check: str
    missing_information: List[str]
    suggested_refinements: List[str]


class JudgeAgent:
    def __init__(
        self,
        llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0),
        memory=Judge_Memory(),
    ):
        self.llm = llm
        self.memory = memory

    def build_judge_prompt(
        self,
        user_query,
        search_results,
        past_evaluations,
        conversation_history,
        iteration,
    ):
        search_text = "\n\n".join(
            [
                f"Source: {r.source}\nTitle: {r.title}\nContent: {r.content}"
                for r in search_results
            ]
        )

        prev_text = "\n".join(
            [
                f"- Iteration {i+1}: sufficient={e.is_sufficient}, missing={e.missing_information}"
                for i, e in enumerate(past_evaluations[-3:])
            ]
        )

        return f"""
        You are the JUDGE agent in a multi-agent retrieval system.

        ITERATION: {iteration}

        User Query:
        {user_query}

        Search Results:
        {search_text}

        Past Evaluations:
        {prev_text}

        Conversation History:
        {conversation_history}

        Return ONLY this JSON:

        {{
        "is_sufficient": bool,
        "reasoning": "",
        "source_quality": "",
        "date_check": "",
        "jurisdiction_check": "",
        "contradiction_check": "",
        "missing_information": [],
        "suggested_refinements": []
        }}
        """

    def evaluate(self, user_query, search_results, conversation_history, iteration):
        prompt = self.build_judge_prompt(
            user_query=user_query,
            search_results=search_results,
            past_evaluations=self.memory.get_last_3_evaluations(),
            conversation_history=conversation_history,
            iteration=iteration,
        )

        structured_llm = self.llm.with_structured_output(JudgeResult)

        llm_result = structured_llm.invoke([HumanMessage(content=prompt)])

        # If missing information is EMPTY → mark sufficient
        if len(llm_result.missing_information) == 0:
            llm_result.is_sufficient = True

        # Save to memory
        self.memory.save_evaluation(llm_result)

        return llm_result


class EvaluateAgent:
    def __init__(self):
        # use consistent model name / casing
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.memory = Judge_Memory()
        self.judge_agent = JudgeAgent(llm=self.llm, memory=self.memory)

        # Get initial query and run all backends
        query = input("Enter your query: ")
        self.results = run_all(query)

        # Convert JSON → objects
        self.sample_results = [
            FakeSearchResult(item) for batch in self.results for item in batch
        ]
        self.current_query = self.extract_query_from_results(self.sample_results)

    def extract_query_from_results(self, res):
        for r in res:
            if hasattr(r, "query") and r.query:
                return r.query
        return None

    def run(self):
        context = ""
        iteration = 1
        max_iterations = 3
        final_result = None

        while iteration <= max_iterations:
            print(f"\n===== ITERATION {iteration} =====")
            result = self.judge_agent.evaluate(
                user_query=self.current_query,
                search_results=self.sample_results,
                conversation_history=context,
                iteration=iteration,
            )

            print("Is Sufficient:", result.is_sufficient)
            print("Reasoning:", result.reasoning)
            print("Missing Info:", result.missing_information)
            print("Refinements:", result.suggested_refinements)

            final_result = result

            if result.is_sufficient:
                print("\nJudge verified! Stopping iteration.")
                break

            # -------------------------------------------
            # UPDATED QUERY LOGIC (only modify query)
            # -------------------------------------------

            # Priority 1 → suggested refinements
            if result.suggested_refinements:
                next_query = result.suggested_refinements[0].strip()

            # Priority 2 → missing information
            elif result.missing_information:
                next_query = (
                    f"{self.current_query} {result.missing_information[0].strip()}"
                )

            # Priority 3 → fallback
            else:
                next_query = self.current_query

            print("Next refined query:", next_query)

            # update current query
            self.current_query = next_query

            # re-run search with refined query
            self.results = run_all(self.current_query)
            self.sample_results = [
                FakeSearchResult(item) for batch in self.results for item in batch
            ]
            iteration += 1

        # Use the last result we have (final_result)
        return json.dumps(
            {
                "is_sufficient": final_result.is_sufficient,
                "reasoning": final_result.reasoning,
                "date_check": final_result.date_check,
                "jurisdiction": final_result.jurisdiction_check,
                "contradiction": final_result.contradiction_check,
                "results": self.results,
            }
        )
