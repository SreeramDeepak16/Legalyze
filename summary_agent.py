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
    s: Dict[str, str] = {}
    s["id"] = str(src.get("id") or src.get("source") or f"source_{idx}")
    s["backend"] = str(src.get("backend") or "")
    s["source"] = str(src.get("source") or "")
    s["title"] = str(src.get("title") or src.get("name") or "")
    s["citation"] = str(src.get("citation") or "")
    content = src.get("content") or src.get("text") or ""
    s["content_full"] = content
    # truncated for prompt safety
    max_prompt_chars = 3000
    s["content"] = content[:max_prompt_chars] + (
        "..." if len(content) > max_prompt_chars else ""
    )
    return s


class SummaryAgent:
    """
    Input: judge_output (raw dict from Judge Agent) and question (str).
    Output: plain text answer (LLM output) with sources appended at the end.

    - Sufficiency is determined *strictly* by judge_output["is_sufficient"].
      The agent does not re-evaluate sufficiency on its own.
    - If is_sufficient is True:
        * Answer the question using ONLY the verified sources.
        * Medium length: not too short, not extremely long.
        * Try to cover all important points needed to answer the question.
    - If is_sufficient is False:
        * Clearly say that there is not enough information to fully answer.
        * Then summarize whatever information is available from the sources.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("SUMMARY_KEY")
        )

        # Prompt used when the judge says information is sufficient
        self.prompt_sufficient = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a concise, careful legal assistant. "
                        "Use ONLY the VERIFIED SOURCES provided to answer the question. "
                        "Talk in the tone that you have looked for the sources and not like they were externally provided to you."
                        "Do not invent facts or speculate beyond the sources. "
                        "Provide a long answer mentioning everything required to answer the question."
                        "(for example, aim for a few well-structured paragraphs that cover all "
                        "important points needed to answer the question)."
                    ),
                ),
                (
                    "human",
                    (
                        "QUESTION:\n{question}\n\n"
                        "VERIFIED SOURCES (numbered):\n{sources_block}\n\n"
                        "Instructions:\n"
                        "- Fully answer the QUESTION using ONLY the VERIFIED SOURCES above.\n"
                        "- Do NOT hallucinate or introduce information not supported by the sources.\n"
                        "- Organize the answer clearly, addressing all key aspects necessary "
                        "to properly answer the QUESTION.\n"
                    ),
                ),
            ]
        )

        # Prompt used when the judge says information is NOT sufficient
        self.prompt_insufficient = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a careful legal assistant. The available sources are not "
                        "sufficient to fully and reliably answer the question.\n"
                        "Talk in the tone that you have looked for the sources and not like they were externally provided to you."
                        "Your job is to:\n"
                        "1) Honestly state that the available information is not sufficient "
                        "to fully answer the question, and\n"
                        "2) Briefly summarize whatever relevant information IS present in the "
                        "sources, without speculating beyond them.\n"
                        "Use ONLY the VERIFIED SOURCES provided. Do not invent facts."
                    ),
                ),
                (
                    "human",
                    (
                        "QUESTION:\n{question}\n\n"
                        "VERIFIED SOURCES (numbered):\n{sources_block}\n\n"
                        "Instructions:\n"
                        "- Start by clearly stating that there is not enough information in these "
                        "sources to fully answer the QUESTION.\n"
                        "- Then, ONLY IF the sources contain RELEVANT information to the question, "
                        "provide a concise summary of the key points and facts that "
                        "the sources DO contain which are relevant to the QUESTION.\n"
                    ),
                ),
            ]
        )

    def summarize_from_judge(
        self,
        judge_output: Dict[str, Any],
        question: str = None,
        max_sources: int = 8,
    ) -> str:
        print("*****************output from judge****************")
        #print(judge_output.get("results")[0][0][200])
        # If question is not explicitly provided, try to recover it from judge_output
        if question is None:
            try:
                question = judge_output.get("results")[0][0].get("query")
            except Exception:
                question = ""

        raw_results = judge_output.get("results", [])
        flat = _flatten_results(raw_results)
        normalized = [_normalize_source(r, i + 1) for i, r in enumerate(flat)]

        # Filter down to contentful sources and limit to max_sources
        contentful = [
            s
            for s in normalized
            if (s.get("content_full") and str(s.get("content_full")).strip())
        ]
        contentful = contentful[:max_sources]

        # If literally nothing contentful is available, we cannot even summarize
        if not contentful:
            return (
                "We could not find sufficient relevant information in the available "
                "sources to answer that question."
            )

        # Build the sources block for the prompt
        lines: List[str] = []
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

        # Determine sufficiency strictly from the judge output
        is_sufficient = bool(judge_output.get("is_sufficient"))

        # Choose the appropriate prompt based on sufficiency
        if is_sufficient:
            messages = self.prompt_sufficient.format_messages(
                question=question,
                sources_block=sources_block,
            )
        else:
            messages = self.prompt_insufficient.format_messages(
                question=question,
                sources_block=sources_block,
            )

        # Call the LLM
        resp = self.llm.invoke(messages)
        answer = (resp.content or "").strip()

        # Append sources (title + URL) at the end of the answer
        sources_lines: List[str] = ["", "", "Sources:"]
        for i, s in enumerate(contentful, start=1):
            title = s.get("title") or "(no title)"
            url = s.get("source") or ""
            # basic title + url line
            sources_lines.append(f"[{i}] {title} - {url}")

        answer_with_sources = answer + "\n" + "\n".join(sources_lines)

        return answer_with_sources