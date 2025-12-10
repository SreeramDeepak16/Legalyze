# simple_summary_agent_option_b.py
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()


def _flatten_results(results_field: Any) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    if not results_field:
        return flat
    if isinstance(results_field, list):
        for item in results_field:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        flat.append(sub)
            elif isinstance(item, dict):
                flat.append(item)
    return flat


def _normalize_source(src: Dict[str, Any], idx: int) -> Dict[str, str]:
    s = {}
    s["id"] = str(src.get("id") or src.get("source") or f"source_{idx}")
    s["backend"] = str(src.get("backend") or "")
    s["source"] = str(src.get("source") or "")
    s["title"] = str(src.get("title") or src.get("name") or "")
    s["citation"] = str(src.get("citation") or "")
    content = src.get("content") or src.get("text") or ""
    s["content_full"] = content
    # truncated for prompt safety
    max_prompt_chars = 3000
    s["content"] = content[:max_prompt_chars] + ("..." if len(content) > max_prompt_chars else content)
    return s


class SummaryAgent:
    """
    Input: judge_output (raw dict from Judge Agent) and question (str).
    Output: plain text answer (LLM output). No duplicate sources appended by code.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("SUMMARY_KEY"))

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a concise, careful legal assistant. Use ONLY the VERIFIED SOURCES provided to answer the question. "
                        "Do not invent facts. Speak in the voice of the system (e.g., 'We found...'). "
                        "You MUST follow the judge flag 'is_sufficient' (passed separately). Do NOT re-evaluate or overrule it."
                    ),
                ),
                (
                    "human",
                    (
                        "QUESTION:\n{question}\n\n"
                        "VERIFIED SOURCES (numbered):\n{sources_block}\n\n"
                        "JUDGE FLAG: is_sufficient = {is_sufficient}\n\n"
                        "Instructions:\n"
                        "- If is_sufficient is true: answer the QUESTION using ONLY the VERIFIED SOURCES. Do not hallucinate. "
                        "Provide an evidence-backed answer to the query. Make sure to cover everything that is important from the given sources.\n\n"
                        "- If is_sufficient is false: do NOT attempt to answer the question. Instead reply exactly and briefly:\n"
                        "'Sufficient relevant information in the available sources to answer the question could not be found.'\n\n"
                        
                    ),
                ),
            ]
        )


    def summarize_from_judge(self, judge_output: Dict[str, Any], question: str = None, max_sources: int = 8) -> str:
        if question is None : 
            question = judge_output.get("results")[0][0].get("query") 
        raw_results = judge_output.get("results", [])
        flat = _flatten_results(raw_results)
        normalized = [_normalize_source(r, i + 1) for i, r in enumerate(flat)]

        contentful = [s for s in normalized if (s.get("content_full") and str(s.get("content_full")).strip())]
        contentful = contentful[:max_sources]

        if not contentful:
            return "We could not find sufficient relevant information in the available sources to answer that question."

        lines = []
        for i, s in enumerate(contentful, start=1):
            title = s.get("title") or s.get("source") or "(no title)"
            url = s.get("source") or ""
            excerpt = (s.get("content") or "").strip().replace("\n", " ")[:800]
            lines.append(f"[{i}] title: {title}")
            lines.append(f"    backend: {s.get('backend')}")
            lines.append(f"    url: {url}")
            lines.append(f"    excerpt: {excerpt}")
            lines.append("")
        sources_block = "\n".join(lines)


        messages = self.prompt.format_messages(question=question, sources_block=sources_block, is_sufficient = judge_output.get("is_sufficient"))
        resp = self.llm.invoke(messages)
        answer = resp.content.strip()

        return answer