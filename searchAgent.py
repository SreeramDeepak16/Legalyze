# SearchAgent.py (updated to use candidate_selector)
from urllib.parse import urlparse
from bs4 import BeautifulSoup #type:ignore
from dotenv import load_dotenv #type:ignore
load_dotenv()

import os
import re
import time
from typing import List, Dict, Any, Optional
import math

import requests #type:ignore

from connectors import CourtListenerClient 
from langchain_google_genai import ChatGoogleGenerativeAI #type:ignore
# from agents.meta_memory_manager import MetaMemoryManager

# import your selector (make sure candidate_selector.py is next to this file)
from candidate_selector import select_best_candidate  # type: ignore

# ---------- helpers ----------
from typing import List, Dict, Any, Optional

def pick_best_case(
    results: List[Dict[str, Any]],
    prefer_court_substr: str = "Supreme",
    prefer_cite_substr: str = "U.S.",
    prefer_name_substr: str = "Topeka",
) -> Optional[Dict[str, Any]]:
    """
    Score and pick the best case from CourtListener results.

    Scoring heuristics:
      +3 if `court` contains prefer_court_substr (e.g., "Supreme")
      +2 if any citation contains prefer_cite_substr (e.g., "U.S.")
      +1 if case name contains prefer_name_substr (e.g., "Topeka")
      small preference for earlier results

    Returns the best result dict or None if no results.
    """
    if not results:
        return None

    def _field_text(val):
        if val is None:
            return ""
        if isinstance(val, list):
            return " ".join(str(x) for x in val).lower()
        return str(val).lower()

    best = None
    best_score = float("-inf")
    for idx, r in enumerate(results):
        score = 0.0
        court = _field_text(r.get("court") or r.get("case_court") or r.get("source_court"))
        case_name = _field_text(r.get("case_name") or r.get("title") or "")
        citation = _field_text(r.get("citation") or "")

        if prefer_court_substr.lower() in court:
            score += 3.0
        if prefer_cite_substr.lower() in citation:
            score += 2.0
        if prefer_name_substr.lower() in case_name:
            score += 1.0

        # small boost for items earlier in the list
        score += max(0.0, 1.0 - idx * 0.05)

        # a tiny penalty for empty content (if content available)
        content_len = len(str(r.get("content") or ""))
        if content_len < 300:
            score -= 0.5

        if score > best_score:
            best_score = score
            best = r

    return best

import re
from urllib.parse import urlparse

CITATION_RE = re.compile(r'(\d{1,4})\s+U\.?S\.?\s+(\d{1,4})', re.IGNORECASE)

def normalize_text(s):
    if isinstance(s, list):
        s = " ".join([str(x) for x in s])
    return (s or "").lower().strip()

def pick_best_case(
    results: List[Dict[str, Any]],
    query: str = "",
    prefer_court_substr: str = "",
    prefer_cite_substr: str = "",
    prefer_name_substr: str = "",
) -> Optional[Dict[str, Any]]:
    if not results:
        return None

    query_norm = normalize_text(query)
    # if query contains an explicit "U.S." citation, prefer items that include that exact citation substring
    cite_match = CITATION_RE.search(query)
    explicit_cite = None
    if cite_match:
        explicit_cite = f"{cite_match.group(1)} u.s. {cite_match.group(2)}"

    def _field_text(val):
        if val is None:
            return ""
        if isinstance(val, list):
            return " ".join(str(x) for x in val).lower()
        return str(val).lower()

    best = None
    best_score = float("-inf")
    for idx, r in enumerate(results):
        score = 0.0
        court = _field_text(r.get("court") or r.get("case_court") or r.get("source_court") or r.get("court_name"))
        case_name = _field_text(r.get("case_name") or r.get("title") or r.get("name"))
        citation = _field_text(r.get("citation") or r.get("citations") or "")

        # exact citation match is highest priority
        if explicit_cite and explicit_cite in citation:
            score += 8.0

        # prefer court substring if present
        if prefer_court_substr and prefer_court_substr.lower() in court:
            score += 3.0

        # prefer citation hint (loose)
        if prefer_cite_substr and prefer_cite_substr.lower() in citation:
            score += 2.0

        # prefer name hint only if query contained that name
        if prefer_name_substr and prefer_name_substr.lower() in query_norm and prefer_name_substr.lower() in case_name:
            score += 1.0

        # small boost for earlier results
        score += max(0.0, 1.0 - idx * 0.05)

        # penalize extremely short or missing content
        content_len = len(str(r.get("content") or r.get("plain_text") or ""))
        if content_len < 200:
            score -= 1.0

        # mild bonus if query tokens appear in title or citation or content
        qtokens = [t for t in re.split(r'\s+', query_norm) if t]
        matches = 0
        haystack = " ".join([case_name, citation, str(r.get("content") or "")])
        for t in qtokens[:6]:  # check a few tokens
            if t and t in haystack:
                matches += 1
        score += 0.3 * matches

        if score > best_score:
            best_score = score
            best = r

    return best


# ---------- SearchAgent ----------
class SearchAgent:
    def __init__(self):
        self.courtlistener = CourtListenerClient()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        # self.meta_manager = MetaMemoryManager()

        
    # ----- Mirix (assumed to return content/snippet) -----
    # def search_mirix_paragraph(self, q: str, top_k: int = 3) -> List[Dict[str, Any]]:
    #     results = self.meta_manager.dispatch(q)
    #     agent_results = results["agent_results"]["retrieval"]
    #     memories = list(agent_results.keys())
    #     out:List[Dict[str,Any]] = []
    #     for memory in memories:
    #         if agent_results[memory]!=None:
    #             out.append({
    #                 "query":q,
    #                 "backend": "mirix",
    #                 "source": memory,
    #                 "title": None,
    #                 "content": agent_results[memory]})
                
    #     return out
    # ----- CourtListener -----
    CITATION_RE = re.compile(r'(\d{1,4})\s+U\.?S\.?\s+(\d{1,4})', re.IGNORECASE)

    def search_courtlistener_paragraph(self, q: str, top_k: int = 1, prefer_scotus: bool = True) -> List[Dict[str, Any]]:
        """
        CourtListener search that *returns an empty list* whenever no reliable candidate is found.

        Behavior summary:
        - Requests up to `max_candidates` from CourtListener.
        - Uses select_best_candidate(...) to pick a candidate.
        - If select_best_candidate returns None -> return [] (no CourtListener answer).
        - Performs two safety checks (short content, citation match). If either fails -> return [].
        - If a candidate is selected, ensure content is filled, attach `_source_confidence`, and return up to top_k items.
        """
        max_candidates = max(top_k, 12)

        # 1) Request candidates
        try:
            results = self.courtlistener.search_with_content(q, max_results=max_candidates)
        except Exception as e:
            print("[courtlistener] search_with_content failed:", repr(e))
            return []

        if not results:
            return []

        # 2) Extract citation hint and conservative case-name hint
        cite_match = CITATION_RE.search(q)
        prefer_cite = ""
        if cite_match:
            prefer_cite = f"{cite_match.group(1)} U.S."

        case_name_hint = None
        m_case = re.search(r'([A-Za-z0-9\.\'"\-\s,&]+?\bv\.?\b[\sA-Za-z0-9\.\'"\-\s,&]+)', q, flags=re.I)
        if m_case:
            case_name_hint = m_case.group(1).strip()

        # 3) Use selector (this respects citation short-circuit internally)
        try:
            selected = select_best_candidate(results, query_case=case_name_hint or "", query_citation=prefer_cite or "")
        except Exception as e:
            print("[selector] error calling select_best_candidate:", repr(e))
            selected = None

        # 4) If selector says "no reliable candidate", return empty list (do not fallback to weak heuristics)
        if selected is None:
            print("[courtlistener] selector returned no reliable candidate -> returning empty list")
            return []

        # 5) Ensure selected has content (selector returns a shallow copy sometimes)
        best = dict(selected)  # work on a shallow copy
        if not best.get("content"):
            sel_id = best.get("id") or best.get("doc_id") or best.get("source")
            for r in results:
                if str(r.get("id")) == str(sel_id) or str(r.get("doc_id")) == str(sel_id) or str(r.get("source")) == str(sel_id):
                    best["content"] = r.get("content") or r.get("plain_text") or r.get("casebody_text") or ""
                    # try to inherit citation/title if missing
                    if not best.get("citation"):
                        best["citation"] = r.get("citation") or r.get("citations")
                    if not best.get("title"):
                        best["title"] = r.get("case_name") or r.get("title")
                    break

        # 6) Safety check A: content length must be reasonably large
        first_content = (best.get("content") or "")
        if len(first_content) < 200:
            print("[courtlistener] selected candidate content too short -> returning empty list")
            return []

        # 7) Safety check B: if query included an explicit U.S. citation, ensure selected contains the expected reporter number
        if cite_match:
            expected_num = str(cite_match.group(1))
            cit_field = normalize_text(best.get("citation") or "")
            content_field = normalize_text(first_content)
            if expected_num not in cit_field and expected_num not in content_field:
                # try to find any result among original results that contains the citation digits
                found = None
                for r in results:
                    rcit = normalize_text(r.get("citation") or "")
                    rcontent = normalize_text(r.get("content") or r.get("plain_text") or "")
                    if expected_num in rcit or expected_num in rcontent:
                        found = r
                        break
                if found is None:
                    print("[courtlistener] citation mismatch and no alternative matched -> returning empty list")
                    return []
                else:
                    # adopt the found result as the best candidate
                    best = dict(found)
                    if not best.get("content"):
                        best["content"] = found.get("content") or found.get("plain_text") or ""

                    if len(best.get("content", "")) < 200:
                        print("[courtlistener] fallback-matched candidate content too short -> returning empty list")
                        return []

        # 8) Attach a simple _source_confidence metric based on selection score (if present)
        sel_score = None
        if isinstance(best.get("_selection_score"), (int, float)):
            sel_score = float(best["_selection_score"])
        elif isinstance(selected.get("_selection_details"), dict) and isinstance(selected["_selection_details"].get("score"), (int, float)):
            sel_score = float(selected["_selection_details"]["score"])

        # Map sel_score -> [0.0, 1.0] conservatively
        confidence = 0.0
        if sel_score is not None:
            if sel_score >= 1_000_000:  # citation short-circuit
                confidence = 1.0
            else:
                # normalize: treat 60 as low threshold, 300+ as very strong; clamp to 1.0
                confidence = min(1.0, max(0.0, (sel_score / 300.0)))
        else:
            # fallback conservative confidence for non-scored selections
            confidence = 0.5

        best["_source_confidence"] = confidence

        # 9) Produce ordered list: prefer best, then original order (but do not return junk)
        ordered = []
        if top_k == 1:
            ordered = [best]
        else:
            seen_ids = set()
            ordered.append(best); seen_ids.add(str(best.get("id") or best.get("doc_id") or best.get("source")))
            for r in results:
                rid = str(r.get("id") or r.get("doc_id") or r.get("source"))
                if rid in seen_ids:
                    continue
                ordered.append(r)
                seen_ids.add(rid)
                if len(ordered) >= top_k:
                    break

        # 10) Normalize output shape and return
        out = []
        for r in ordered[:top_k]:
            content = r.get("content") or r.get("plain_text") or r.get("casebody_text") or ""
            out.append({
                "query": q,
                "backend": "courtlistener",
                "id": r.get("id"),
                "source": r.get("url") or r.get("resource") or r.get("source"),
                "title": r.get("case_name") or r.get("title") or r.get("name"),
                "citation": r.get("citation") or r.get("citations"),
                "content": content,
                "retrieved_at": r.get("retrieved_at"),
                "_selection_score": r.get("_selection_score"),
                "_selection_details": r.get("_selection_details"),
                "_source_confidence": r.get("_source_confidence", confidence if r is best else 0.0),
            })

        return out


    # ----- Web search (SerpAPI) -----
    def search_web_paragraph(self, q: str, top_k: int = 1, preferred_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Uses SerpAPI (serpapi.com). Requires SERPAPI_API_KEY in .env.

        Returns a list containing AT MOST ONE hit (the first valid source found).
        - If preferred_domain is set (e.g. 'wikipedia.org'), only considers results from that domain.
        - Requests a few SerpAPI results (num=5) so we can skip blocked/empty results and still return one.
        """
        if not self.serpapi_key:
            return []

        serp_url = "https://serpapi.com/search.json"
        # ask for a few results so we can skip bad ones and still return one
        params = {"engine": "google", "q": q, "api_key": self.serpapi_key, "num": max(5, top_k)}
        try:
            r = requests.get(serp_url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("[web] SerpAPI error:", repr(e))
            return []

        organic = data.get("organic_results") or data.get("organic", []) or []
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def domain_of(url: Optional[str]) -> Optional[str]:
            if not url:
                return None
            try:
                from urllib.parse import urlparse
                return urlparse(url).netloc.lower().lstrip("www.")
            except Exception:
                return None

        # iterate through organic results and return the first good one
        for item in organic:
            link = item.get("link") or item.get("url")
            title = item.get("title") or ""
            serp_snippet = item.get("snippet") or item.get("snippet_highlighted") or ""

            # filter by preferred domain if requested
            if preferred_domain:
                d = domain_of(link)
                if not d or preferred_domain.lower().lstrip("www.") not in d:
                    continue

            content = ""
            if not link:
                content = serp_snippet
            else:
                try:
                    rr = requests.get(link, headers={"User-Agent": "mirix-search-agent/1.0"}, timeout=15)
                    rr.raise_for_status()
                    ctype = (rr.headers.get("Content-Type") or "").lower()

                    # PDF handling
                    if "application/pdf" in ctype or (link and link.lower().endswith(".pdf")):
                        try:
                            import io
                            from pypdf import PdfReader
                            pdf_bytes = rr.content
                            reader = PdfReader(io.BytesIO(pdf_bytes))
                            pages = []
                            for i, page in enumerate(reader.pages):
                                if i >= 20:
                                    break
                                try:
                                    text = page.extract_text() or ""
                                except Exception:
                                    text = ""
                                pages.append(text)
                            content = "\n\n".join(pages).strip()
                        except Exception as e:
                            print("PDF parsing failed (pypdf):", repr(e))
                            content = ""
                    else:
                        # HTML handling
                        try:
                            soup = BeautifulSoup(rr.text, "html.parser")
                            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                            content = "\n\n".join(paragraphs).strip()
                            if len(content) < 200:
                                content = soup.get_text(" ", strip=True)
                        except Exception as e:
                            print("HTML parsing failed:", repr(e))
                            content = ""
                except Exception as e:
                    print("Fetching link failed:", repr(e))
                    content = ""

            final_text = content or serp_snippet or ""
            if not final_text:
                # nothing usable from this result -> try next
                continue

            # para = choose_best_paragraph(final_text, q, title)
            hit = {
                "query":q,
                "backend": "web",
                "source": link,
                "title": title,
                "content": final_text,
            }
            # return a single-item list (first good source)
            return [hit]

        # if we reach here: no usable hits
        return []
