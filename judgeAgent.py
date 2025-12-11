# judge_agent.py
import os
import re
import json
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set

from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError

# langchain / LLM wrappers (type-ignore if necessary)
from langchain_core.messages import HumanMessage  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

# your search agent entrypoint
from run_search_agent import run_all


# ----------------------------
# Data models
# ----------------------------
class JudgeResult(BaseModel):
    is_sufficient: bool = Field(default=False)
    reasoning: str
    source_quality: str = ""
    date_check: str = ""
    jurisdiction_check: str = ""
    contradiction_check: str = ""
    missing_information: List[str] = Field(default_factory=list)
    suggested_refinements: List[str] = Field(default_factory=list)


# ----------------------------
# Metadata helper functions
# ----------------------------
_JURISDICTION_KEYWORDS = {
    "us": ["united states", "u.s.", "u.s", "us federal", "federal"],
    "india": ["india", "indian", "supreme court of india", "sc of india"],
    "uk": ["united kingdom", "england", "scotland", "uk"],
    # expand as needed
}


def get_domain_type(url: Optional[str]) -> str:
    """Heuristic domain classification: 'gov', 'edu', 'court_decision', 'news', 'user_generated', 'other', 'unknown'."""
    if not url:
        return "unknown"
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
    except Exception:
        return "other"

    if hostname.endswith(".gov") or ".gov." in hostname:
        return "gov"
    if hostname.endswith(".edu") or ".edu." in hostname:
        return "edu"
    if any(k in hostname for k in ("court", "law", "uscourts", "justia", "courtlistener", "opinion")):
        return "court_decision"
    if any(k in hostname for k in ("news", "times", "post", "guardian", "bbc", "cnn")):
        return "news"
    if any(k in hostname for k in ("reddit", "wordpress", "medium", "blogspot", "quora")):
        return "user_generated"
    return "other"


def extract_years_and_dates(text: str) -> Dict[str, Optional[int]]:
    """Return earliest and latest 4-digit years found; detect ISO-like dates presence."""
    years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", text)]
    earliest = min(years) if years else None
    latest = max(years) if years else None
    iso_dates = bool(re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", text))
    return {"earliest_year": earliest, "latest_year": latest, "has_iso_date": iso_dates}


def detect_jurisdictions(text: str, url: Optional[str]) -> List[str]:
    """Keyword and TLD heuristics to detect jurisdictions mentioned or implied."""
    t = (text or "").lower()
    found: Set[str] = set()
    for key, kw_list in _JURISDICTION_KEYWORDS.items():
        for kw in kw_list:
            if kw in t:
                found.add(key)
                break

    # TLD hints
    try:
        hostname = urlparse(url).hostname or "" if url else ""
    except Exception:
        hostname = ""
    hostname = hostname.lower()
    if hostname.endswith(".uk"):
        found.add("uk")
    if hostname.endswith(".in"):
        found.add("india")
    if hostname.endswith(".gov") and "usa" in hostname:
        found.add("us")

    return sorted(found)


def contradiction_scan(text: str) -> Dict[str, Any]:
    """
    Conservative contradiction heuristic:
      - Extract sentences with assertion-like verbs.
      - Pairwise check for shared tokens and opposing polarity (negation present in one, not in the other).
    Returns a short report (boolean + examples).
    """
    sents = [s.strip() for s in re.split(r'[.\n]+', text) if s.strip()]
    assertions = []
    for s in sents:
        if re.search(r"\b(is|are|shall|must|prohib|allow|permits|forbid|not|no|never|except|unless)\b", s, re.I):
            assertions.append(s)

    contradictions = []
    STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "for", "by", "that"}
    lowered = [re.sub(r'[^a-z0-9\s]', '', a.lower()) for a in assertions]
    tokenized = [set([w for w in a.split() if w and w not in STOP]) for a in lowered]

    for i in range(len(assertions)):
        for j in range(i + 1, len(assertions)):
            common = tokenized[i].intersection(tokenized[j])
            if len(common) >= 2:
                neg_i = bool(re.search(r"\b(not|no|never|except|unless)\b", assertions[i], re.I))
                neg_j = bool(re.search(r"\b(not|no|never|except|unless)\b", assertions[j], re.I))
                if neg_i != neg_j:
                    contradictions.append({
                        "sent1": assertions[i][:400],
                        "sent2": assertions[j][:400],
                        "common_terms": sorted(list(common)),
                        "polarity_diff": (neg_i, neg_j),
                    })

    return {"has_contradiction": bool(contradictions), "examples": contradictions[:3]}


# ----------------------------
# WrapResult - per-search-result wrapper
# ----------------------------
class WrapResult:
    def __init__(self, d: Dict[str, Any], summary_limit: int = 1500):
        self.query: Optional[str] = d.get("query")
        self.backend: Optional[str] = d.get("backend")
        self.source: Optional[str] = d.get("source")
        self.title: Optional[str] = d.get("title")
        self.citation = d.get("citation")
        self.full_content: str = d.get("content") or ""
        self.summary_for_judge: str = self.build_summary(self.full_content, summary_limit)
        self.metadata: Dict[str, Any] = self.build_metadata()

    def build_summary(self, text: str, limit: int = 1500) -> str:
        clean = " ".join(text.split())
        return clean[:limit] + ("..." if len(clean) > limit else "")

    def build_metadata(self) -> Dict[str, Any]:
        text = (self.full_content or "").lower()
        title = (self.title or "").lower()

        basic = {
            "length": len(self.full_content),
            "word_count": len(self.full_content.split()),
            "title_in_text": title in text if title else False,
        }

        # domain / authority
        domain_type = get_domain_type(self.source)
        basic["domain_type"] = domain_type
        basic["is_authoritative"] = domain_type in ("gov", "edu", "court_decision")

        # years / dates
        dates = extract_years_and_dates(self.full_content)
        basic.update(dates)

        # jurisdiction hints
        jurisdictions = detect_jurisdictions(self.full_content, self.source)
        basic["jurisdictions_detected"] = jurisdictions

        # contradiction scan
        contradiction = contradiction_scan(self.full_content)
        basic["contradiction_scan"] = contradiction

        # legacy quick-checks
        basic["has_supreme"] = "supreme" in text
        basic["has_united_states"] = "united states" in text

        return basic


# ----------------------------
# Memory for judge
# ----------------------------
class JudgeMemory:
    def __init__(self):
        # store evaluaton dicts for inspectability
        self.memory: List[Dict[str, Any]] = []

    def save(self, evaluation: Dict[str, Any]) -> None:
        self.memory.append(evaluation)

    def last_n(self, n: int = 3) -> List[Dict[str, Any]]:
        return self.memory[-n:]


# ----------------------------
# JudgeAgent: LLM prompt, invocation, parse
# ----------------------------
class JudgeAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI, memory: JudgeMemory):
        self.llm = llm
        self.memory = memory

    def build_prompt(self, query: str, results: Sequence[WrapResult],
                     past: Sequence[Dict[str, Any]], history: str, iteration: int) -> str:
        result_blocks = "\n\n".join(
            f"[{(r.backend or 'UNKNOWN').upper()} SOURCE]\n"
            f"URL: {r.source}\n"
            f"TITLE: {r.title}\n"
            f"SUMMARY: {r.summary_for_judge}\n"
            f"METADATA: {json.dumps(r.metadata, ensure_ascii=False, indent=2)}"
            for r in results
        )

        past_eval_text = "\n".join(
            f"- Prev #{i+1}: sufficient={p.get('is_sufficient')}, missing={p.get('missing_information')}"
            for i, p in enumerate(past)
        ) or "None"

        return f"""
You are the JUDGE agent in a retrieval-validation pipeline. You MUST NOT summarize further; evaluate.
Use the provided METADATA fields to perform these checks:
  1) SOURCE QUALITY CHECK:
     - Use 'domain_type' and 'is_authoritative' to mark authority (gov/edu/court_decision => high authority).
     - Count authoritative vs user_generated and explain which results are authoritative.
  2) DATE CHECK:
     - Use 'earliest_year', 'latest_year', and 'has_iso_date' to determine currency.
     - If important facts rely on older dates, flag as potentially outdated.
  3) JURISDICTION CHECK:
     - Use 'jurisdictions_detected' and the URL to verify jurisdiction alignment with the user query.
     - If user specified a jurisdiction in conversation history, check it.
  4) CONTRADICTION SCAN:
     - Use 'contradiction_scan' (has_contradiction + examples). If contradictions are present, explain their significance.
  5) MISSING INFORMATION:
     - Based on the query, name specific missing points (e.g., 'no primary court decision found on topic X', 'no statute cited', 'no recent confirmation (post-2020)').

Iteration: {iteration}

User Query:
{query}

Retrieved Search Results (summaries + metadata only):
{result_blocks}

Past Evaluations (last 3):
{past_eval_text}

Conversation History / user-specified jurisdiction or context:
{history if history else "None provided"}

Return ONLY JSON in this format (valid JSON):
{{
 "is_sufficient": true/false,
 "reasoning": "string (concise)",
 "source_quality": "string (briefly list authoritative sources and concerns)",
 "date_check": "string (currency assessment)",
 "jurisdiction_check": "string (match or mismatch details)",
 "contradiction_check": "string (summary, include examples if any)",
 "missing_information": ["specific", "missing", "items"],
 "suggested_refinements": ["query refinements or terms to search next"]
}}
"""

    def _call_llm_and_parse(self, prompt: str) -> JudgeResult:
        """
        Calls the LLM and returns a validated JudgeResult.
        Handles parsing errors and returns a safe fallback with reasoning included.
        """
        try:
            msg = HumanMessage(content=prompt)
            # Different wrappers expose different call signatures - try common ones
            if hasattr(self.llm, "invoke"):
                llm_response = self.llm.invoke([msg])
            elif callable(self.llm):
                llm_response = self.llm([msg])
            else:
                # last-resort: try structured output if available
                llm_response = self.llm

            # extract textual content robustly
            if hasattr(llm_response, "content"):
                content = llm_response.content
            elif isinstance(llm_response, dict) and "content" in llm_response:
                content = llm_response["content"]
            elif isinstance(llm_response, str):
                content = llm_response
            else:
                content = str(llm_response)

            content = content.strip().strip("`").strip()

            # First attempt: direct JSON parse
            json_obj = None
            try:
                json_obj = json.loads(content)
            except json.JSONDecodeError:
                # attempt to extract first {...} block in text
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        json_obj = json.loads(content[start:end+1])
                    except json.JSONDecodeError:
                        json_obj = None

            if json_obj is None:
                print("LLM returned unparsable JSON. Raw content excerpt:")
                print(content[:1000])
                return JudgeResult(
                    is_sufficient=False,
                    reasoning=f"LLM returned unparsable JSON. Raw content excerpt: {content[:1000]}",
                    source_quality="unknown",
                    date_check="unknown",
                    jurisdiction_check="unknown",
                    contradiction_check="unknown",
                    missing_information=["LLM_parsing_failed"],
                    suggested_refinements=[]
                )

            # Validate against the schema
            try:
                jr = JudgeResult.parse_obj(json_obj)
                return jr
            except ValidationError as e:
                print("LLM JSON did not match schema:", e)
                print("Raw JSON object:", json_obj)
                return JudgeResult(
                    is_sufficient=False,
                    reasoning=f"LLM JSON did not match schema: {e}. Raw JSON: {json_obj}",
                    source_quality=json_obj.get("source_quality", ""),
                    date_check=json_obj.get("date_check", ""),
                    jurisdiction_check=json_obj.get("jurisdiction_check", ""),
                    contradiction_check=json_obj.get("contradiction_check", ""),
                    missing_information=json_obj.get("missing_information", []),
                    suggested_refinements=json_obj.get("suggested_refinements", [])
                )

        except Exception as exc:
            print("LLM call failed with exception:")
            traceback.print_exc()
            return JudgeResult(
                is_sufficient=False,
                reasoning=f"LLM invocation failed: {exc}",
                source_quality="error",
                date_check="error",
                jurisdiction_check="error",
                contradiction_check="error",
                missing_information=["llm_error"],
                suggested_refinements=[]
            )

    def evaluate(self, query: str, results: Sequence[WrapResult], history: str, iteration: int) -> JudgeResult:
        prompt = self.build_prompt(query=query, results=results, past=self.memory.last_n(3), history=history, iteration=iteration)
        jr = self._call_llm_and_parse(prompt)

        # Auto-mark: if LLM found no missing info, consider sufficient unless LLM explicitly says otherwise
        if not jr.missing_information:
            jr.is_sufficient = True

        # persist serializable dict
        # include the parsed model output only
        try:
            self.memory.save(jr.dict())
        except Exception:
            # fallback if pydantic dict method changes; store model_dump if available
            try:
                self.memory.save(jr.model_dump())  # pydantic v2 compatible
            except Exception:
                print("Failed to save judge result to memory; storing minimal info.")
                self.memory.save({"is_sufficient": jr.is_sufficient, "reasoning": jr.reasoning})

        return jr


# ----------------------------
# EvaluateAgent wrapper (top-level)
# ----------------------------

class EvaluateAgent:
    def __init__(self, query: str, judge_key_env: str = "JUDGE_KEY"):
        api_key = os.getenv(judge_key_env)
        if not api_key:
            raise EnvironmentError(f"Environment variable {judge_key_env} not found")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0,
        )

        self.memory = JudgeMemory()
        self.judge = JudgeAgent(llm=self.llm, memory=self.memory)

        raw = run_all(query)
        self.results = self._normalize_raw_results(raw)
        self.current_query = self._extract_query(self.results) or query

        # Run deterministic pre-check: if query contains a canonical citation (e.g., "514 U.S. 549"),
        # ensure the results actually include it; otherwise force a targeted refinement search.
        self._pre_refine_for_citation_if_needed(original_query=query)

    @staticmethod
    def _normalize_raw_results(raw: Any) -> List[WrapResult]:
        """
        Accept list-of-dicts or list-of-lists-of-dicts and return WrapResult list.
        """
        normalized: List[Dict[str, Any]] = []
        if raw is None:
            return []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, list):
                    normalized.extend(item)
                else:
                    normalized.append(item)
        elif isinstance(raw, dict):
            normalized.append(raw)
        else:
            try:
                normalized = list(raw)
            except Exception:
                normalized = []

        return [WrapResult(d) for d in normalized if isinstance(d, dict)]

    @staticmethod
    def _extract_query(results: Sequence[WrapResult]) -> Optional[str]:
        for r in results:
            if r.query:
                return r.query
        return None

    # ----------------------------
    # NEW: deterministic citation detection
    # ----------------------------
    @staticmethod
    def _extract_canonical_citation_from_query(q: str) -> Optional[str]:
        """
        Very small parser: look for patterns like '514 U.S. 549' or '514 U.S. 549 (1995)'.
        Returns normalized lowercase string '514 u.s. 549' if found, else None.
        """
        if not q:
            return None
        m = re.search(r"\b(\d{1,4})\s*U\.?S\.?\s*(\d{1,4})\b", q, flags=re.I)
        if m:
            return f"{m.group(1)} u.s. {m.group(2)}".lower()
        return None

    @staticmethod
    def _results_have_citation(results: Sequence[WrapResult], canonical_cite: str) -> bool:
        """
        Check citations arrays, titles, and content for the canonical citation string.
        canonical_cite should be normalized (lowercase) like '514 u.s. 549'.
        """
        if not canonical_cite:
            return False
        for r in results:
            # check citation list
            c = r.citation
            if c:
                # normalize citation entries and check for both tokens  (e.g., '514' and '549')
                try:
                    cite_strs = [str(x).lower() for x in (c if isinstance(c, (list, tuple)) else [c])]
                except Exception:
                    cite_strs = [str(c).lower()]
                for s in cite_strs:
                    if canonical_cite in s:
                        return True
            # check title
            if r.title and canonical_cite in r.title.lower():
                return True
            # check full_content for canonical cite
            if canonical_cite in (r.full_content or "").lower():
                return True
        return False

    def _pre_refine_for_citation_if_needed(self, original_query: str) -> None:
        """
        If the user query contained a canonical reporter citation (e.g., '514 U.S. 549') but none of
        the current results include that citation, issue a targeted re-query that prefers authoritative
        domains (courtlistener, scotus.gov, oyez.org, justia) and replace self.results.
        This increases the chance Judge will get direct evidence and reduces LLM hallucination.
        """
        canonical = self._extract_canonical_citation_from_query(original_query)
        if not canonical:
            return  # nothing to do

        if self._results_have_citation(self.results, canonical):
            print(f"Deterministic check: canonical citation present in initial results ({canonical}).")
            return

        # Build a targeted refinement query that strongly biases authoritative hosts
        authoritative_sites = ["courtlistener.com", "scotus.gov", "oyez.org", "justia.com"]
        site_clause = " OR ".join([f"site:{s}" for s in authoritative_sites])
        refined_query = f"{original_query} {canonical} {site_clause}"
        print(f"Deterministic check: canonical citation {canonical} missing; running targeted refinement search: {refined_query}")

        # Re-run search and update results
        try:
            raw = run_all(refined_query)
            new_results = self._normalize_raw_results(raw)
            # If new results have the citation, adopt them; else keep original results but print warning
            if self._results_have_citation(new_results, canonical):
                print("Targeted refinement found canonical citation; updating results.")
                self.results = new_results
            else:
                print("Targeted refinement did not find canonical citation. Keeping original results.")
        except Exception as e:
            print("Targeted refinement search failed with exception:")
            traceback.print_exc()
            # keep original results on failure

    # ----------------------------
    # run loop (unchanged except uses self.results from pre-refinement)
    # ----------------------------
    def run(self, max_iterations: int = 3) -> Dict[str, Any]:
        iteration = 1
        final: Optional[JudgeResult] = None
        seen_queries: Set[str] = {self.current_query} if self.current_query else set()

        while iteration <= max_iterations:
            print(f"\n===== ITERATION {iteration} =====")
            jr = self.judge.evaluate(query=self.current_query, results=self.results, history="", iteration=iteration)

            print("Sufficient?", jr.is_sufficient)
            print("Missing:", jr.missing_information)
            print("Refinements:", jr.suggested_refinements)

            final = jr

            if jr.is_sufficient:
                print("Judge validated final set.")
                break

            # Choose next query
            if jr.suggested_refinements:
                new_query = jr.suggested_refinements[0]
            elif jr.missing_information:
                new_query = f"{self.current_query} {jr.missing_information[0]}"
            else:
                new_query = self.current_query

            # Prevent cycles
            if new_query in seen_queries:
                print("Refinement cycle detected; stopping further iterations.")
                break
            if new_query:
                seen_queries.add(new_query)

            print("Refined Query →", new_query)
            self.current_query = new_query

            raw = run_all(new_query)
            self.results = self._normalize_raw_results(raw)

            iteration += 1

        return {
            "is_sufficient": bool(final.is_sufficient) if final else False,
            "reasoning": final.reasoning if final else "no-evaluation",
            "results_full": [r.full_content for r in self.results],
            "results_objects": self.results,
            "memory": self.memory.memory,
        }


# Entrypoint test when run as script (optional)
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run Judge evaluation for a query")
    parser.add_argument("query", nargs="?", default=None, help="Query text to search and judge")
    parser.add_argument("--key-env", default="JUDGE_KEY", help="Environment variable containing the Judge API key")
    args = parser.parse_args()

    # if not provided on CLI, prompt
    query = args.query
    if not query:
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
        else:
            try:
                query = input("Enter query (or paste the search term): ").strip()
            except EOFError:
                print("No query provided and unable to read from input.")
                raise SystemExit(2)

    if not query:
        print("No query provided.")
        raise SystemExit(2)

    api_key = os.getenv(args.key_env)
    if not api_key:
        print(f"Environment variable {args.key_env} not found. Please set it before running.")
        raise SystemExit(2)

    try:
        agent = EvaluateAgent(query, judge_key_env=args.key_env)
        out = agent.run()
        print(json.dumps({
            "is_sufficient": out["is_sufficient"],
            "reasoning": out["reasoning"],
            "memory_len": len(out["memory"]),
        }, indent=2))
    except Exception:
        print("Failed to run EvaluateAgent, exception trace:")
        traceback.print_exc()
