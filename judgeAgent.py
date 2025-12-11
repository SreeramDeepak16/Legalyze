import os
from bs4 import BeautifulSoup #type:ignore
import re
from typing import Dict, Any
from pydantic import BaseModel #type:ignore
from typing import List
from langchain_core.messages import HumanMessage #type:ignore
from langchain_google_genai import ChatGoogleGenerativeAI #type:ignore
from agents.run_search_agent import run_all
import json



class WrapResult:
    def __init__(self, d: Dict[str, Any]):
        self.query = d.get("query")
        self.backend = d.get("backend")
        self.source = d.get("source")
        self.title = d.get("title")
        self.citation = d.get("citation")

        # Full text exactly as retrieved
        self.full_content = d.get("content") or ""

        # Summary for Judge (compact)
        self.summary_for_judge = self.build_summary(self.full_content)

        # Metadata for correctness validation
        self.metadata = self.build_metadata()

    def build_summary(self, text: str, limit: int = 1500) -> str:
        clean = " ".join(text.split())
        return clean[:limit] + ("..." if len(clean) > limit else "")

    def build_metadata(self):
        text = self.full_content.lower()
        return {
            "length": len(self.full_content),
            "has_year_1954": ("1954" in text),
            "has_year_1973": ("1973" in text),
            "has_supreme": ("supreme" in text),
            "has_united_states": ("united states" in text),
            "title_in_text": self.title.lower() in text if self.title else False,
            "word_count": len(self.full_content.split())
        }



class JudgeResult(BaseModel):
    is_sufficient: bool
    reasoning: str
    source_quality: str
    date_check: str
    jurisdiction_check: str
    contradiction_check: str
    missing_information: List[str]
    suggested_refinements: List[str]

class JudgeMemory:
    def __init__(self):
        self.memory = []

    def save(self, evaluation):
        self.memory.append(evaluation)

    def last3(self):
        return self.memory[-3:]




class JudgeAgent:
    def __init__(self, llm, memory: JudgeMemory):
        self.llm = llm
        self.memory = memory

    def build_prompt(self, query, results, past, history, iteration):
        result_blocks = "\n\n".join(
            f"[{r.backend.upper()} SOURCE]\n"
            f"URL: {r.source}\n"
            f"TITLE: {r.title}\n"
            f"SUMMARY: {r.summary_for_judge}\n"
            f"METADATA: {r.metadata}"
            for r in results
        )

        past_eval_text = "\n".join(
            f"- Prev #{i+1}: sufficient={e.is_sufficient}, missing={e.missing_information}"
            for i, e in enumerate(past)
        )

        return f"""
You are the JUDGE agent in a retrieval-validation pipeline.
You DO NOT summarize. You ONLY evaluate correctness.

Iteration: {iteration}

User Query:
{query}

Retrieved Search Results (summaries + metadata only):
{result_blocks}

Past Evaluations (last 3):
{past_eval_text}

Conversation History:
{history}

Return ONLY JSON in this format:

{{
 "is_sufficient": true/false,
 "reasoning": "string",
 "source_quality": "string",
 "date_check": "string",
 "jurisdiction_check": "string",
 "contradiction_check": "string",
 "missing_information": [],
 "suggested_refinements": []
}}
"""

    def evaluate(self, query, results, history, iteration):
        prompt = self.build_prompt(
            query=query,
            results=results,
            past=self.memory.last3(),
            history=history,
            iteration=iteration,
        )

        structured = self.llm.with_structured_output(JudgeResult)

        llm_output = structured.invoke([HumanMessage(content=prompt)])

        # auto-mark sufficient
        if len(llm_output.missing_information) == 0:
            llm_output.is_sufficient = True

        self.memory.save(llm_output)
        return llm_output




class EvaluateAgent:
    def __init__(self, query: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("JUDGE_KEY"),
            temperature=0
        )

        self.memory = JudgeMemory()
        self.judge = JudgeAgent(llm=self.llm, memory=self.memory)

        # initial search
        raw = run_all(query)
        self.results = [WrapResult(item) for batch in raw for item in batch]
        self.current_query = self.extract_query(self.results)

    def extract_query(self, results):
        for r in results:
            if r.query:
                return r.query
        return None

    def run(self):
        iteration = 1
        final = None

        while iteration <= 3:
            print(f"\n===== ITERATION {iteration} =====")

            result = self.judge.evaluate(
                query=self.current_query,
                results=self.results,
                history="",
                iteration=iteration,
            )

            print("Sufficient?", result.is_sufficient)
            print("Missing:", result.missing_information)
            print("Refinements:", result.suggested_refinements)

            final = result

            if result.is_sufficient:
                print("Judge validated final set.")
                break

            # Pick next query
            if result.suggested_refinements:
                new_query = result.suggested_refinements[0]
            elif result.missing_information:
                new_query = f"{self.current_query} {result.missing_information[0]}"
            else:
                new_query = self.current_query

            print("Refined Query →", new_query)
            self.current_query = new_query

            # Re-run search
            raw = run_all(new_query)
            self.results = [WrapResult(item) for batch in raw for item in batch]

            iteration += 1

        # Return everything, including FULL content for downstream SummaryAgent
        return {
            "is_sufficient": final.is_sufficient,
            "reasoning": final.reasoning,
            "results_full": [r.full_content for r in self.results],
            "results_objects": self.results
        }
