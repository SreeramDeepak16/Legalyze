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
 
from agents.connectors import CourtListenerClient 
from langchain_google_genai import ChatGoogleGenerativeAI #type:ignore
from agents.meta_memory_manager import MetaMemoryManager

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


# ---------- SearchAgent ----------
class SearchAgent:
    def __init__(self):
        self.courtlistener = CourtListenerClient()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.meta_manager = MetaMemoryManager()

        
    # ----- Mirix (assumed to return content/snippet) -----
    def search_mirix_paragraph(self, q: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.meta_manager.dispatch(q)
        agent_results = results["agent_results"]["retrieval"]
        memories = list(agent_results.keys())
        out:List[Dict[str,Any]] = []
        for memory in memories:
            if agent_results[memory]!=None:
                out.append({
                    "query":q,
                    "backend": "mirix",
                    "source": memory,
                    "title": None,
                    "content": agent_results[memory]})
                
        return out
    # ----- CourtListener -----
    def search_courtlistener_paragraph(self, q: str, top_k: int = 1, prefer_scotus: bool = True) -> List[Dict[str, Any]]:
        """
        Fetches up to `max_candidates` search results from CourtListener, picks the best one
        (using pick_best_case) and returns JSON-shaped hits limited by top_k.

        - prefer_scotus: when True, pick_best_case will prefer 'Supreme' court / 'U.S.' citations / 'Topeka' name.
        """
        max_candidates = max(top_k, 8)  # request several so we can rank
        results = self.courtlistener.search_with_content(q, max_results=max_candidates)

        if not results:
            return []

        # If prefer_scotus, bias the ranking toward SCOTUS/U.S. reports/Topeka
        if prefer_scotus:
            best = pick_best_case(results,
                                  prefer_court_substr="Supreme",
                                  prefer_cite_substr="U.S.",
                                  prefer_name_substr="Topeka")
        else:
            best = pick_best_case(results,
                                  prefer_court_substr="",  # weaker bias
                                  prefer_cite_substr="",
                                  prefer_name_substr="")

        # If user asked for multiple hits (top_k > 1), produce ordered list:
        ordered: List[Dict[str, Any]] = []
        if top_k == 1:
            chosen = best or results[0]
            ordered = [chosen]
        else:
            # put best first, then the rest (unique)
            seen_ids = set()
            if best:
                ordered.append(best); seen_ids.add(best.get("id"))
            for r in results:
                if r.get("id") in seen_ids:
                    continue
                ordered.append(r); seen_ids.add(r.get("id"))
                if len(ordered) >= top_k:
                    break

        out: List[Dict[str, Any]] = []
        for r in ordered[:top_k]:
            out.append({
                "query": q,
                "backend": "courtlistener",
                "id": r.get("id"),
                "source": r.get("url"),
                "title": r.get("case_name"),
                "citation": r.get("citation"),
                "content": r.get("content") or "",
                "retrieved_at": r.get("retrieved_at"),
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
