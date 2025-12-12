# judge_agent.py
import os
import re
import json
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set

from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError

# type-ignore LLM wrappers if necessary
from langchain_core.messages import HumanMessage  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

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
# Helpers: domain, dates, jurisdictions, contradictions
# ----------------------------
_JURISDICTION_KEYWORDS = {
    "us": ["united states", "u.s.", "u.s", "us federal", "federal"],
    "india": ["india", "indian", "supreme court of india", "sc of india"],
    "uk": ["united kingdom", "england", "scotland", "uk"],
}


def get_domain_type(url: Optional[str]) -> str:
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
    # fixed: use non-capturing group so full year strings are returned (avoid truncated years)
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)]
    earliest = min(years) if years else None
    latest = max(years) if years else None
    iso_dates = bool(re.search(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", text))
    return {"earliest_year": earliest, "latest_year": latest, "has_iso_date": iso_dates}


def detect_jurisdictions(text: str, url: Optional[str]) -> List[str]:
    t = (text or "").lower()
    found: Set[str] = set()
    for key, kw_list in _JURISDICTION_KEYWORDS.items():
        for kw in kw_list:
            if kw in t:
                found.add(key)
                break
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
    Less-sensitive contradiction scan:
      - skip sentences that are explicitly historical/descriptive (amend/adopt/withdraw etc.)
      - require at least 3 meaningful shared tokens and opposing polarity to flag contradiction
    """
    sents = [s.strip() for s in re.split(r'[.\n]+', text) if s.strip()]
    assertions = []
    HISTORICAL_KEYWORDS = {
        "revise", "revised", "revision", "amend", "amended", "amendments",
        "adopt", "adopted", "withdraw", "withdrawn", "not adopted", "not adopted"
    }
    for s in sents:
        if re.search(r"\b(is|are|shall|must|prohib|allow|permits|forbid|not|no|never|except|unless)\b", s, re.I):
            low = s.lower()
            # skip sentences that are primarily historical/descriptive
            if any(k in low for k in HISTORICAL_KEYWORDS):
                continue
            assertions.append(s)
    contradictions = []
    STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "for", "by", "that", "this", "these"}
    lowered = [re.sub(r'[^a-z0-9\s]', '', a.lower()) for a in assertions]
    tokenized = [set([w for w in a.split() if w and w not in STOP]) for a in lowered]
    for i in range(len(assertions)):
        for j in range(i + 1, len(assertions)):
            common = tokenized[i].intersection(tokenized[j])
            # require >=3 shared meaningful tokens to reduce false positives
            if len(common) >= 3:
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
# WrapResult
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
        domain_type = get_domain_type(self.source)
        basic["domain_type"] = domain_type
        basic["is_authoritative"] = domain_type in ("gov", "edu", "court_decision")
        dates = extract_years_and_dates(self.full_content)
        basic.update(dates)
        jurisdictions = detect_jurisdictions(self.full_content, self.source)
        basic["jurisdictions_detected"] = jurisdictions
        contradiction = contradiction_scan(self.full_content)
        basic["contradiction_scan"] = contradiction
        basic["has_supreme"] = "supreme" in text
        basic["has_united_states"] = "united states" in text
        return basic


# ----------------------------
# Memory
# ----------------------------
class JudgeMemory:
    def __init__(self):
        self.memory: List[Dict[str, Any]] = []

    def save(self, evaluation: Dict[str, Any]) -> None:
        self.memory.append(evaluation)

    def last_n(self, n: int = 3) -> List[Dict[str, Any]]:
        return self.memory[-n:]


# ----------------------------
# JudgeAgent
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
     - List the most authoritative sources you see.
  2) DATE CHECK:
     - Use 'earliest_year', 'latest_year', and 'has_iso_date' to determine currency.
  3) JURISDICTION CHECK:
     - Use 'jurisdictions_detected' and the URL to verify jurisdiction alignment.
  4) CONTRADICTION SCAN:
     - Use 'contradiction_scan' (has_contradiction + examples).
  5) MISSING INFORMATION:
     - Based on the query, explicitly name specific missing points (e.g., 'no primary court decision found', 'no statute cited').

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
 "source_quality": "string",
 "date_check": "string",
 "jurisdiction_check": "string",
 "contradiction_check": "string",
 "missing_information": ["specific", "missing", "items"],
 "suggested_refinements": ["query refinements or terms to search next"]
}}
"""

    def _call_llm_and_parse(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM robustly and return dict with keys:
          - parsed (JudgeResult or fallback dict)
          - raw_text (string)
        """
        raw_text = ""
        try:
            msg = HumanMessage(content=prompt)
            if hasattr(self.llm, "invoke"):
                llm_response = self.llm.invoke([msg])
            elif callable(self.llm):
                llm_response = self.llm([msg])
            else:
                llm_response = self.llm

            if hasattr(llm_response, "content"):
                raw_text = llm_response.content
            elif isinstance(llm_response, dict) and "content" in llm_response:
                raw_text = llm_response["content"]
            elif isinstance(llm_response, str):
                raw_text = llm_response
            else:
                raw_text = str(llm_response)

            raw_text = raw_text.strip().strip("`").strip()

            json_obj = None
            try:
                json_obj = json.loads(raw_text)
            except json.JSONDecodeError:
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        json_obj = json.loads(raw_text[start:end+1])
                    except json.JSONDecodeError:
                        json_obj = None

            if json_obj is None:
                # return a fallback structure so calling code can include raw_text
                return {"parsed": None, "raw_text": raw_text}

            try:
                jr = JudgeResult.parse_obj(json_obj)
                return {"parsed": jr, "raw_text": raw_text}
            except ValidationError:
                return {"parsed": None, "raw_text": raw_text, "json_obj": json_obj}

        except Exception:
            print("LLM call failed:")
            traceback.print_exc()
            return {"parsed": None, "raw_text": raw_text}

    # Deterministic check helpers used for Option A
    @staticmethod
    def _extract_canonical_citation_from_query(q: str) -> Optional[str]:
        if not q:
            return None
        m = re.search(r"\b(\d{1,4})\s*U\.?S\.?\s*(\d{1,4})\b", q, flags=re.I)
        if m:
            return f"{m.group(1)} u.s. {m.group(2)}".lower()
        return None

    @staticmethod
    def _results_have_citation(results: Sequence[WrapResult], canonical_cite: str) -> bool:
        if not canonical_cite:
            return False
        for r in results:
            c = r.citation
            if c:
                try:
                    cite_strs = [str(x).lower() for x in (c if isinstance(c, (list, tuple)) else [c])]
                except Exception:
                    cite_strs = [str(c).lower()]
                for s in cite_strs:
                    if canonical_cite in s:
                        return True
            if r.title and canonical_cite in r.title.lower():
                return True
            if canonical_cite in (r.full_content or "").lower():
                return True
        return False

    @staticmethod
    def _title_or_name_matches_query(results: Sequence[WrapResult], query: str) -> bool:
        q = (query or "").lower()
        for r in results:
            if not r.title:
                continue
            if r.title.lower() in q or q in r.title.lower():
                return True
            # approximate: require at least two tokens from case name be shared
            q_tokens = set(re.findall(r"[a-z0-9]+", q))
            title_tokens = set(re.findall(r"[a-z0-9]+", (r.title or "").lower()))
            if len(q_tokens.intersection(title_tokens)) >= 2:
                return True
        return False

    @staticmethod
    def _has_authoritative_result(results: Sequence[WrapResult]) -> bool:
        return any(r.metadata.get("is_authoritative") for r in results)

    def evaluate(self, query: str, results: Sequence[WrapResult], history: str, iteration: int) -> Dict[str, Any]:
        prompt = self.build_prompt(query=query, results=results, past=self.memory.last_n(3), history=history, iteration=iteration)
        llm_resp = self._call_llm_and_parse(prompt)

        parsed = llm_resp.get("parsed")
        raw_text = llm_resp.get("raw_text", "")

        # Save raw LLM output and parsed json for audit
        audit_entry: Dict[str, Any] = {"raw_llm": raw_text, "parsed": None}

        # If parsing failed but json_obj present, include it
        if parsed is None and "json_obj" in llm_resp:
            audit_entry["parsed"] = llm_resp["json_obj"]
        elif parsed is not None:
            audit_entry["parsed"] = parsed.dict()

        # Now apply Option A deterministic safety checks to determine final is_sufficient
        canonical = self._extract_canonical_citation_from_query(query)
        citation_present = self._results_have_citation(results, canonical) if canonical else False
        authoritative_exists = self._has_authoritative_result(results)
        title_match = self._title_or_name_matches_query(results, query)

        # default fallback JudgeResult if LLM parsing failed
        if parsed is None:
            # LLM failed to return valid JSON -> mark insufficient and explain why
            final_jr = JudgeResult(
                is_sufficient=False,
                reasoning="LLM did not return valid structured JSON evaluation; cannot trust sufficiency.",
                source_quality="unknown",
                date_check="unknown",
                jurisdiction_check="unknown",
                contradiction_check="unknown",
                missing_information=["LLM_parsing_failed"],
                suggested_refinements=[]
            )
            # save audit and final (include gating flag)
            try:
                self.memory.save({"audit": audit_entry, "final": final_jr.dict(), "deterministic_require_citation": bool(canonical)})
            except Exception:
                self.memory.save({"audit": audit_entry, "final": {"is_sufficient": final_jr.is_sufficient, "reasoning": final_jr.reasoning}, "deterministic_require_citation": bool(canonical)})
            return {"parsed_judge": final_jr, "raw_llm": raw_text}

        # At this point we have a parsed JudgeResult from LLM
        jr: JudgeResult = parsed  # type: ignore

        # Build features for a scored decision (less brittle than a single LLM gate)
        require_citation = bool(canonical)

        # Per-result contradiction check (aggregate across returned results)
        no_contradiction = True
        for r in results:
            try:
                cs = r.metadata.get("contradiction_scan", {})
                if cs and cs.get("has_contradiction"):
                    no_contradiction = False
                    break
            except Exception:
                # if metadata malformed, be conservative and assume potential contradiction
                no_contradiction = False
                break

        # determine whether missing items are "minor"
        MINOR_MISSINGS = ["definition", "define", "overview", "purpose", "how is", "adopt", "adoption", "one-sentence", "single-sentence"]
        missing_items = [str(m).lower() for m in jr.missing_information or []]
        minor_only_missing = len(missing_items) > 0 and all(any(k in m for k in MINOR_MISSINGS) for m in missing_items)

        # treat missing-info as partial credit rather than hard fail
        no_missing_info = len(jr.missing_information) == 0

        # Scoring weights (adjusted)
        score = 0
        if authoritative_exists:
            score += 45   # bigger weight for authority
        if title_match:
            score += 20
        if citation_present:
            score += 20
        if no_missing_info:
            score += 15
        elif minor_only_missing:
            score += 8   # some credit if only minor formatting/definition items missing
        if no_contradiction:
            score += 22

        max_possible = 45 + 20 + 20 + 15 + 22  # =122
        score_pct = int((score / max_possible) * 100) if max_possible > 0 else 0

        # Decision thresholds:
        # - If the query asked for a canonical reporter citation, be strict:
        #     require citation_present OR score_pct >= 85
        # - Otherwise (topical queries), accept if score_pct >= 55 (loosened)
        passed = False
        deterministic_reasons: List[str] = []

        if require_citation:
            if not citation_present:
                deterministic_reasons.append(f"Canonical citation '{canonical}' from query not found in any result.")
            if score_pct >= 85 or (citation_present and authoritative_exists and no_contradiction):
                passed = True
        else:
            # topical query path
            # if the LLM says sufficient and there are no contradictions and authoritative result exists -> accept
            if jr.is_sufficient and authoritative_exists and no_contradiction:
                passed = True
            else:
                # fallback to score threshold
                if score_pct >= 55:
                    passed = True
                else:
                    deterministic_reasons.append(f"Aggregate score too low: {score_pct}% (needs >=55 for topical queries).")

        # Always include the LLM's judgments as part of the reasoning details
        if not jr.is_sufficient:
            deterministic_reasons.append("LLM judged results as not sufficient.")
        if jr.missing_information:
            deterministic_reasons.append(f"LLM reported missing information: {jr.missing_information}")
        if not authoritative_exists:
            deterministic_reasons.append("No authoritative result found (gov/edu/court_decision).")
        if not no_contradiction:
            deterministic_reasons.append("Contradictions detected in retrieved content.")

        # Build final_jr consistent with previous code shape
        if passed:
            final_jr = jr
            final_jr.is_sufficient = True
            final_jr.reasoning = (final_jr.reasoning or "") + (
                f" (Accepted by scored deterministic checks: score_pct={score_pct}, "
                f"authoritative={authoritative_exists}, title_match={title_match}, citation_present={citation_present}.)"
            )
        else:
            final_jr = jr
            final_jr.is_sufficient = False
            augment = " | Deterministic checks failed: " + " ; ".join(deterministic_reasons)
            final_jr.reasoning = (final_jr.reasoning or "") + augment
            # ensure missing_information includes deterministic reasons
            for r in deterministic_reasons:
                if r not in final_jr.missing_information:
                    final_jr.missing_information.append(r)

        # Save both raw and final_jr to memory for auditing, include which gating path and score
        try:
            self.memory.save({
                "audit": audit_entry,
                "final": final_jr.dict(),
                "deterministic_require_citation": require_citation,
                "deterministic_score_pct": score_pct,
                "deterministic_reasons": deterministic_reasons,
            })
        except Exception:
            try:
                self.memory.save({
                    "audit": audit_entry,
                    "final": final_jr.model_dump(),
                    "deterministic_require_citation": require_citation,
                    "deterministic_score_pct": score_pct,
                    "deterministic_reasons": deterministic_reasons,
                })
            except Exception:
                self.memory.save({
                    "audit": audit_entry,
                    "final": {"is_sufficient": final_jr.is_sufficient, "reasoning": final_jr.reasoning},
                    "deterministic_require_citation": require_citation,
                    "deterministic_score_pct": score_pct,
                    "deterministic_reasons": deterministic_reasons,
                })

        return {"parsed_judge": final_jr, "raw_llm": raw_text}


# ----------------------------
# EvaluateAgent
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

        # deterministic pre-check: if query has canonical citation and results don't, try targeted refinement
        self._pre_refine_for_citation_if_needed(original_query=query)

    @staticmethod
    def _normalize_raw_results(raw: Any) -> List[WrapResult]:
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

    @staticmethod
    def _extract_canonical_citation_from_query(q: str) -> Optional[str]:
        if not q:
            return None
        m = re.search(r"\b(\d{1,4})\s*U\.?S\.?\s*(\d{1,4})\b", q, flags=re.I)
        if m:
            return f"{m.group(1)} u.s. {m.group(2)}".lower()
        return None

    @staticmethod
    def _results_have_citation(results: Sequence[WrapResult], canonical_cite: str) -> bool:
        if not canonical_cite:
            return False
        for r in results:
            c = r.citation
            if c:
                try:
                    cite_strs = [str(x).lower() for x in (c if isinstance(c, (list, tuple)) else [c])]
                except Exception:
                    cite_strs = [str(c).lower()]
                for s in cite_strs:
                    if canonical_cite in s:
                        return True
            if r.title and canonical_cite in r.title.lower():
                return True
            if canonical_cite in (r.full_content or "").lower():
                return True
        return False

    def _pre_refine_for_citation_if_needed(self, original_query: str) -> None:
        canonical = self._extract_canonical_citation_from_query(original_query)
        if not canonical:
            return
        if self._results_have_citation(self.results, canonical):
            print(f"Deterministic check: canonical citation present in initial results ({canonical}).")
            return
        authoritative_sites = ["courtlistener.com", "scotus.gov", "oyez.org", "justia.com"]
        site_clause = " OR ".join([f"site:{s}" for s in authoritative_sites])
        refined_query = f"{original_query} {canonical} {site_clause}"
        print(f"Deterministic check: canonical citation {canonical} missing; running targeted refinement search: {refined_query}")
        try:
            raw = run_all(refined_query)
            new_results = self._normalize_raw_results(raw)
            if self._results_have_citation(new_results, canonical):
                print("Targeted refinement found canonical citation; updating results.")
                self.results = new_results
            else:
                print("Targeted refinement did not find canonical citation. Keeping original results.")
        except Exception:
            print("Targeted refinement search failed:")
            traceback.print_exc()

    def run(self, max_iterations: int = 3) -> Dict[str, Any]:
        iteration = 1
        final = None
        seen_queries: Set[str] = {self.current_query} if self.current_query else set()

        while iteration <= max_iterations:
            print(f"\n===== ITERATION {iteration} =====")
            eval_out = self.judge.evaluate(query=self.current_query, results=self.results, history="", iteration=iteration)
            jr: JudgeResult = eval_out["parsed_judge"]
            raw_llm = eval_out.get("raw_llm", "")

            print("Sufficient?", jr.is_sufficient)
            print("Missing:", jr.missing_information)
            print("Refinements:", jr.suggested_refinements)
            # For debugging: show whether deterministic gates passed
            canonical = JudgeAgent._extract_canonical_citation_from_query(self.current_query)
            citation_present = JudgeAgent._results_have_citation(self.results, canonical) if canonical else False
            authoritative_exists = any(r.metadata.get("is_authoritative") for r in self.results)
            title_match = JudgeAgent._title_or_name_matches_query(self.results, self.current_query)
            print("Deterministic checks -> canonical:", canonical, "citation_present:", citation_present,
                  "authoritative_exists:", authoritative_exists, "title_match:", title_match)

            final = jr

            if jr.is_sufficient:
                print("Judge validated final set.")
                break

            if jr.suggested_refinements:
                new_query = jr.suggested_refinements[0]
            elif jr.missing_information:
                new_query = f"{self.current_query} {jr.missing_information[0]}"
            else:
                new_query = self.current_query

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
