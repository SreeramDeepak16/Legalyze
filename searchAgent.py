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
    def search_courtlistener_paragraph(self, q: str, top_k: int = 1, prefer_scotus: bool = True) -> List[Dict[str, Any]]:
        max_candidates = max(top_k, 12)  # request more candidates
        results = self.courtlistener.search_with_content(q, max_results=max_candidates)

        if not results:
            return []

        # extract citation and a heuristic case-name from query for selector
        cite_match = CITATION_RE.search(q)
        prefer_cite = ""
        if cite_match:
            prefer_cite = f"{cite_match.group(1)} U.S."

        # heuristic case name extraction (very conservative)
        case_name_hint = None
        # look for patterns like "X v. Y" or "X vs Y"
        m_case = re.search(r'([A-Za-z0-9\.\'"\-\s,&]+?\bv\.?\b[\sA-Za-z0-9\.\'"\-\s,&]+)', q, flags=re.I)
        if m_case:
            case_name_hint = m_case.group(1).strip()

        # Try the candidate_selector first (this will honor short-circuit citation automatch if implemented)
        try:
            selected = select_best_candidate(results, query_case=case_name_hint or "", query_citation=prefer_cite or "")
        except Exception as e:
            # selector may raise if unexpected shape; log and fallback to existing heuristics
            print("[selector] error calling select_best_candidate:", repr(e))
            selected = None

        best = None
        # If selector returned a valid candidate, use that (it may be auto-shortcircuit)
        if selected:
            # selected is a shallow copy from select_best_candidate; it should contain id/title/citation/content or similar
            best = selected
            # but ensure we have a 'content' field; if missing try to find same id in original results and copy content
            if not best.get("content"):
                sel_id = best.get("id") or best.get("doc_id")
                for r in results:
                    if str(r.get("id")) == str(sel_id) or str(r.get("doc_id")) == str(sel_id):
                        best["content"] = r.get("content") or r.get("plain_text") or r.get("casebody_text") or ""
                        break
        else:
            # fallback to your original heuristic
            prefer_name = ""
            if "topeka" in q.lower():
                prefer_name = "Topeka"

            if prefer_scotus:
                best = pick_best_case(results,
                                    query=q,
                                    prefer_court_substr="Supreme",
                                    prefer_cite_substr=prefer_cite or "U.S.",
                                    prefer_name_substr=prefer_name)
            else:
                best = pick_best_case(results, query=q, prefer_court_substr="", prefer_cite_substr=prefer_cite, prefer_name_substr=prefer_name)

        # produce ordered list (keep original ordering after the chosen best)
        ordered = []
        if top_k == 1:
            chosen = best or results[0]
            ordered = [chosen]
        else:
            seen_ids = set()
            if best:
                ordered.append(best); seen_ids.add(best.get("id"))
            for r in results:
                if r.get("id") in seen_ids:
                    continue
                ordered.append(r); seen_ids.add(r.get("id"))
                if len(ordered) >= top_k:
                    break

        out = []
        for r in ordered[:top_k]:
            # normalize fields / prefer longer content fields if available
            content = r.get("content") or r.get("plain_text") or r.get("casebody_text") or ""
            # if the selector returned a shallow-copied selected object, it may already have "content"; prefer it
            if isinstance(r, dict) and r.get("_selection_score") and not content:
                # try to locate original result content
                orig_id = r.get("id")
                for rr in results:
                    if str(rr.get("id")) == str(orig_id):
                        content = rr.get("content") or rr.get("plain_text") or ""
                        break

            # debug logging
            print("[courtlistener] chosen id:", r.get("id"), "title:", r.get("case_name") or r.get("title"), "citation:", r.get("citation"), "content_len:", len(content))
            out.append({
                "query": q,
                "backend": "courtlistener",
                "id": r.get("id"),
                "source": r.get("url") or r.get("resource") or r.get("source"),
                "title": r.get("case_name") or r.get("title"),
                "citation": r.get("citation"),
                "content": content,
                "retrieved_at": r.get("retrieved_at"),
                # preserve selection metadata if present
                "_selection_score": r.get("_selection_score"),
                "_selection_details": r.get("_selection_details"),
            })

        # extra safety: if content is very short or clearly not matching the explicit citation in query, fallback to web search
        first = out[0] if out else None
        if first:
            if len(first["content"]) < 200:
                print("[courtlistener] low content length for chosen result; attempting web fallback")
                web_hits = self.search_web_paragraph(q, top_k=1)
                if web_hits:
                    return web_hits
            # if query had explicit U.S. citation, require that citation appears in chosen result
            if cite_match:
                # prefer exact numeric U.S. reporter match in citation list/text
                expected_num = str(cite_match.group(1))
                cit_field = normalize_text(first.get("citation") or "")
                if expected_num not in cit_field:
                    print("[courtlistener] chosen result lacks expected citation; trying to find a result with matching citation")
                    # try to find one in results
                    for r in results:
                        cit = normalize_text(r.get("citation") or "")
                        if f"{cite_match.group(1)} u.s." in cit or str(cite_match.group(1)) in cit:
                            content = r.get("content") or r.get("plain_text") or ""
                            return [ {
                                "query": q,
                                "backend": "courtlistener",
                                "id": r.get("id"),
                                "source": r.get("url"),
                                "title": r.get("case_name") or r.get("title"),
                                "citation": r.get("citation"),
                                "content": content,
                                "retrieved_at": r.get("retrieved_at"),
                            } ]
                    # fallback to web if no matching citation in results
                    web_hits = self.search_web_paragraph(q, top_k=1)
                    if web_hits:
                        return web_hits

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
