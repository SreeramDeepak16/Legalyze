# search_agent/agent.py
from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

#from mirix_interface import MirixMemoryClientProtocol, MirixMemoryStub
from connectors import CourtListenerClient
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- helpers ----------
def _norm_score(x, lo, hi):
    if hi - lo == 0:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

# def choose_best_paragraph(content: str, query: str, title: Optional[str] = None, max_paras: int = 600) -> str:
#     """
#     Given full text `content` and `query`, return the most relevant paragraph.
#     Heuristics: paragraph length, query-term matches, presence of legal signpost words,
#     penalize headings/short captions.
#     """
#     if not content:
#         return ""

#     q_tokens = [t.lower() for t in re.split(r"\W+", query) if len(t) > 2][:12]
#     signpost = ["held", "holding", "held that", "therefore", "conclude", "concluded",
#                 "accordingly", "judgment", "opinion", "reason", "reasoning", "analysis",
#                 "dissent", "majority", "court finds", "liable", "reversed", "affirmed", "remanded"]

#     paras = [p.strip() for p in re.split(r'\n{2,}', content) if p.strip()]
#     if not paras:
#         paras = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]

#     best_para = ""
#     best_score = -999.0

#     for idx, para in enumerate(paras[:max_paras]):
#         low = para.lower()
#         if len(low) < 60:
#             continue
#         letters = "".join(ch for ch in para if ch.isalpha())
#         if len(letters) >= 6:
#             upp = sum(1 for ch in letters if ch.isupper())
#             if upp / max(1, len(letters)) > 0.75:
#                 continue

#         score = min(len(low) / 500.0, 2.0)
#         for t in q_tokens:
#             if t in low:
#                 score += 1.2
#         for s in signpost:
#             if s in low:
#                 score += 2.0
#         if title:
#             tlow = title.lower()
#             if tlow and tlow in low:
#                 score += 2.5
#         if idx <= 2:
#             score -= 0.45
#         if idx > 1:
#             score += 0.1
#         if " v. " in low or " u.s. " in low or "§" in para or "sec." in low:
#             score += 0.5

#         if score > best_score:
#             best_score = score
#             best_para = para

#     if not best_para:
#         for p in paras:
#             if len(p) > 150:
#                 best_para = p
#                 break
#     if not best_para:
#         best_para = paras[0] if paras else content[:800]

#     return best_para.strip()[:1600]


# ---------- SearchAgent ----------
class SearchAgent:
    #def __init__(self, mirix_client: Optional[MirixMemoryClientProtocol] = None, gemini_model: str = "gemini-1.5-pro"):
    def __init__(self, gemini_model: str = "gemini-2.5-pro"):
        #self.mirix = mirix_client or MirixMemoryStub()
        self.courtlistener = CourtListenerClient()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.llm = (
            ChatGoogleGenerativeAI(model=gemini_model, temperature=0)
            if os.getenv("GOOGLE_API_KEY")
            else None
        )

    # ----- Mirix (assumed to return content/snippet) -----
    def search_mirix_paragraph(self, q: str, top_k: int = 3) -> List[Dict[str, Any]]:
        hits = self.mirix.search(q, top_k=top_k)
        out: List[Dict[str, Any]] = []
        for h in hits:
            content = h.get("content") or h.get("snippet") or ""
            title = h.get("source") or h.get("title") or None
            # para = choose_best_paragraph(content, q, title)
            out.append({
                "query":q,
                "backend": "mirix",
                "id": h.get("id"),
                "source": h.get("source"),
                "title": h.get("title"),
                "content": content,
            })
        return out

    # ----- CourtListener -----
    def search_courtlistener_paragraph(self, q: str, top_k: int = 1) -> List[Dict[str, Any]]:
        results = self.courtlistener.search(q, page_size=max(top_k, 10))
        q_tokens = [t.lower() for t in re.split(r"\W+", q) if len(t) > 2]
        def name_match(r):
            cn = (r.get("case_name") or "").lower()
            cit = (r.get("citation") or "").lower()
            if any(tok in cn for tok in q_tokens if len(tok) > 3):
                return True
            if any(tok in cit for tok in q_tokens if len(tok) > 1):
                return True
            return False

        matching = [r for r in results if name_match(r)]
        ordered = (matching + [r for r in results if r not in matching])[:top_k]

        out: List[Dict[str, Any]] = []
        for r in ordered:
            api_url = r.get("url")
            full = self.courtlistener.fetch(api_url)
            content = full.get("content", "") or ""
            title = r.get("case_name")
            # para = choose_best_paragraph(content, q, title)
            out.append({
                "query":q, 
                "backend": "courtlistener",
                "source": api_url,
                "title": title,
                "citation": r.get("citation"),
                "content": content,
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
