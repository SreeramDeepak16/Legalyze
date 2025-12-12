# SearchAgent.py (fixed)
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup # type:ignore
from dotenv import load_dotenv # type:ignore
load_dotenv()

import os
import re
import time
from typing import List, Dict, Any, Optional

import requests # type:ignore

from agents.connectors import CourtListenerClient
from agents.meta_memory_manager import MetaMemoryManager
from langchain_google_genai import ChatGoogleGenerativeAI #type:ignore

from agents.candidate_selector import select_best_candidate  # type: ignore

CITATION_RE = re.compile(r'(\d{1,4})\s+U\.?S\.?\s+(\d{1,4})', re.IGNORECASE)


def normalize_text(s: Optional[Any]) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    return str(s).lower().strip()


class SearchAgent:
    def __init__(self):
        self.courtlistener = CourtListenerClient()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        # self.meta_manager = MetaMemoryManager()

    # def search_mirix_paragraph(self, q: str, top_k: int = 3) -> List[Dict[str, Any]]:
    #     results = self.meta_manager.dispatch(q)
    #     agent_results = results.get("agent_results", {}).get("retrieval", {})
    #     out: List[Dict[str, Any]] = []
    #     for memory, val in agent_results.items():
    #         if val is not None:
    #             out.append({
    #                 "query": q,
    #                 "backend": "mirix",
    #                 "source": memory,
    #                 "title": None,
    #                 "content": val
    #             })
    #     return out

    def search_courtlistener_paragraph(self, q: str, top_k: int = 1, prefer_scotus: bool = True) -> List[Dict[str, Any]]:
        max_candidates = max(top_k, 12)
        try:
            results = self.courtlistener.search_with_content(q, max_results=max_candidates)
        except Exception as e:
            print("[courtlistener] search_with_content failed:", repr(e))
            return []

        if not results:
            return []

        cite_match = CITATION_RE.search(q)
        prefer_cite = ""
        if cite_match:
            prefer_cite = f"{cite_match.group(1)} U.S."

        case_name_hint = None
        m_case = re.search(r'([A-Za-z0-9\.\'"\-\s,&]+?\bv\.?\b[\sA-Za-z0-9\.\'"\-\s,&]+)', q, flags=re.I)
        if m_case:
            case_name_hint = m_case.group(1).strip()

        try:
            selected = select_best_candidate(results, query_case=case_name_hint or "", query_citation=prefer_cite or "")
        except Exception as e:
            print("[selector] error calling select_best_candidate:", repr(e))
            selected = None

        if selected is None:
            print("[courtlistener] selector returned no reliable candidate -> returning empty list")
            return []

        best = dict(selected)
        if not best.get("content"):
            sel_id = best.get("id") or best.get("doc_id") or best.get("source")
            for r in results:
                if str(r.get("id")) == str(sel_id) or str(r.get("doc_id")) == str(sel_id) or str(r.get("source")) == str(sel_id):
                    best["content"] = r.get("content") or r.get("plain_text") or r.get("casebody_text") or ""
                    if not best.get("citation"):
                        best["citation"] = r.get("citation") or r.get("citations")
                    if not best.get("title"):
                        best["title"] = r.get("case_name") or r.get("title")
                    break

        first_content = (best.get("content") or "")
        if len(first_content) < 200:
            print("[courtlistener] selected candidate content too short -> returning empty list")
            return []

        if cite_match:
            expected_num = str(cite_match.group(1))
            cit_field = normalize_text(best.get("citation") or "")
            content_field = normalize_text(first_content)
            if expected_num not in cit_field and expected_num not in content_field:
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
                    best = dict(found)
                    if not best.get("content"):
                        best["content"] = found.get("content") or found.get("plain_text") or ""
                    if len(best.get("content", "")) < 200:
                        print("[courtlistener] fallback-matched candidate content too short -> returning empty list")
                        return []

        sel_score = None
        if isinstance(best.get("_selection_score"), (int, float)):
            sel_score = float(best.get("_selection_score"))
        elif isinstance(selected.get("_selection_details"), dict) and isinstance(selected["_selection_details"].get("score"), (int, float)):
            sel_score = float(selected["_selection_details"]["score"])

        confidence = 0.0
        if sel_score is not None:
            if sel_score >= 1_000_000:
                confidence = 1.0
            else:
                confidence = min(1.0, max(0.0, (sel_score / 300.0)))
        else:
            confidence = 0.5

        best["_source_confidence"] = confidence

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

    def search_web_paragraph(self, q: str, top_k: int = 1, preferred_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        SerpAPI -> attempt best extraction from each organic result.
        Fallbacks:
        1) HTML extraction using BeautifulSoup with browser-like headers.
        2) Direct PDF parsing if Content-Type indicates PDF or linked PDF found on page.
        3) Jina.ai text-extraction proxy (https://r.jina.ai/http/<url>) if site blocks fetches (403) or extraction is poor.
        Returns up to `top_k` hits.
        """
        if not self.serpapi_key:
            return []

        serp_url = "https://serpapi.com/search.json"
        params = {"engine": "google", "q": q, "api_key": self.serpapi_key, "num": max(5, top_k)}
        try:
            r = requests.get(serp_url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("[web] SerpAPI error:", repr(e))
            return []

        organic = data.get("organic_results") or data.get("organic") or []
        hits: List[Dict[str, Any]] = []

        # simple pool of modern user agents to reduce bot blocks
        USER_AGENTS = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]

        def domain_of(url: Optional[str]) -> Optional[str]:
            if not url:
                return None
            try:
                return urlparse(url).netloc.lower().lstrip("www.")
            except Exception:
                return None

        def try_fetch_with_headers(url: str, timeout: int = 18):
            """Return (response_or_none, status, used_headers)"""
            import random
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
                "Connection": "keep-alive",
            }
            try:
                rr = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                rr.raise_for_status()
                return rr, rr.status_code, headers
            except requests.exceptions.HTTPError as he:
                return he.response if isinstance(he, requests.exceptions.HTTPError) else None, getattr(he.response, "status_code", None), headers
            except Exception:
                return None, None, headers

        for item in organic:
            if len(hits) >= top_k:
                break

            link = item.get("link") or item.get("url")
            title = item.get("title") or ""
            serp_snippet = item.get("snippet") or item.get("snippet_highlighted") or ""
            if preferred_domain:
                d = domain_of(link)
                if not d or preferred_domain.lower().lstrip("www.") not in d:
                    continue

            final_text = ""
            # Try primary fetch + extraction, with a few retries
            tried_jina = False
            for attempt in range(3):
                if not link:
                    final_text = serp_snippet or ""
                    break
                rr, status, used_headers = try_fetch_with_headers(link)
                # If we got a proper response object and status is OK
                if rr is not None and getattr(rr, "status_code", 0) == 200:
                    try:
                        ctype = (rr.headers.get("Content-Type") or "").lower()
                    except Exception:
                        ctype = ""
                    # PDF direct
                    if "application/pdf" in ctype or (link and link.lower().endswith(".pdf")):
                        try:
                            import io
                            from pypdf import PdfReader
                            pdf_bytes = rr.content
                            reader = PdfReader(io.BytesIO(pdf_bytes))
                            pages = []
                            for i, page in enumerate(reader.pages):
                                if i >= 120:
                                    break
                                try:
                                    text = page.extract_text() or ""
                                except Exception:
                                    text = ""
                                if text:
                                    pages.append(text)
                            final_text = "\n\n".join(pages).strip()
                        except Exception as e:
                            print("[web] PDF parsing failed (direct):", repr(e))
                            final_text = ""
                    else:
                        # HTML extraction
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(rr.text, "html.parser")
                            # quick cleaning
                            for bad in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
                                bad.decompose()
                            for br in soup.find_all("br"):
                                br.replace_with("\n")

                            # look for linked PDFs first
                            pdf_url = None
                            try:
                                for a in soup.find_all("a", href=True):
                                    href = a["href"].strip()
                                    if href.lower().endswith(".pdf"):
                                        pdf_url = href
                                        break
                                    if "pdf" in href.lower() and (href.lower().startswith("http") or href.startswith("/")):
                                        pdf_url = href
                                        break
                                if pdf_url:
                                    from urllib.parse import urljoin
                                    pdf_url = urljoin(rr.url, pdf_url)
                                    rr_pdf, status_pdf, _ = try_fetch_with_headers(pdf_url, timeout=20)
                                    if rr_pdf is not None and getattr(rr_pdf, "status_code", 0) == 200:
                                        try:
                                            import io
                                            from pypdf import PdfReader
                                            reader = PdfReader(io.BytesIO(rr_pdf.content))
                                            pages = []
                                            for i, page in enumerate(reader.pages):
                                                if i >= 120:
                                                    break
                                                try:
                                                    text = page.extract_text() or ""
                                                except Exception:
                                                    text = ""
                                                if text:
                                                    pages.append(text)
                                            if pages:
                                                final_text = "\n\n".join(pages).strip()
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                            # if no PDF text or too short, try JSON-LD/meta/og/article selectors
                            if not final_text or len(final_text) < 400:
                                try:
                                    og = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
                                    if og and og.get("content"):
                                        mtext = og.get("content").strip()
                                        if len(mtext) > len(final_text):
                                            final_text = mtext

                                    for script in soup.find_all("script", {"type": "application/ld+json"}):
                                        try:
                                            import json
                                            payload = json.loads(script.string or "{}")
                                            candidates = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
                                            for c in candidates:
                                                if isinstance(c, dict):
                                                    ab = c.get("articleBody") or c.get("description") or c.get("text")
                                                    if isinstance(ab, str) and len(ab) > len(final_text):
                                                        final_text = ab.strip()
                                                        break
                                        except Exception:
                                            continue
                                except Exception:
                                    pass

                            # opinion / content containers and large blocks
                            if not final_text or len(final_text) < 400:
                                try:
                                    selectors = [
                                        "div.opinion", "div#opinion", "div.content", "div.casetext",
                                        "div.casebody", "div.opinionText", "article", "div#main", "div#content",
                                        "section.opinion", "div#opinionBody"
                                    ]
                                    found_blocks = []
                                    for sel in selectors:
                                        parts = soup.select(sel)
                                        if parts:
                                            for p in parts:
                                                text = p.get_text(" ", strip=True)
                                                if len(text) > 200:
                                                    found_blocks.append(text)
                                    if found_blocks:
                                        final_text = "\n\n".join(found_blocks)
                                    else:
                                        blocks = []
                                        for tag in soup.find_all(["p", "div", "article", "section"]):
                                            txt = tag.get_text(" ", strip=True)
                                            if len(txt) > 250:
                                                blocks.append(txt)
                                        if blocks:
                                            final_text = "\n\n".join(blocks)
                                        else:
                                            text = soup.get_text(" ", strip=True)
                                            if len(text) > 400:
                                                final_text = text
                                except Exception as e:
                                    print("[web] HTML parsing failed:", repr(e))
                                    final_text = ""
                        except Exception as e:
                            print("[web] BeautifulSoup parse error:", repr(e))
                            final_text = ""
                else:
                    # 403 or other non-200 -> try fallback proxies or jina.ai
                    if getattr(rr, "status_code", None) == 403:
                        # attempt Jina.ai text extraction fallback once
                        if not tried_jina:
                            tried_jina = True
                            try:
                                # jina expects the raw target url after /http:// or /https://
                                cleaned = link.replace("https://", "").replace("http://", "")
                                jina_url = f"https://r.jina.ai/http://{cleaned}"
                                jr = requests.get(jina_url, headers={"User-Agent": USER_AGENTS[0]}, timeout=18)
                                jr.raise_for_status()
                                text = jr.text or ""
                                if text and len(text) > 200:
                                    final_text = text.strip()
                                    break
                            except Exception as e:
                                # jina fallback failed, continue attempts
                                print("[web] jina.ai fallback failed:", repr(e))
                    # small sleep/backoff
                    time.sleep(0.8 * (attempt + 1))
                    continue

                # if we got to here and final_text is long enough, break
                if final_text and len(final_text) >= 400:
                    break

                # if too short, try a short delay and retry (sometimes servers give better content after redirect/resolving)
                time.sleep(0.6)

            # final fallback to serp snippet
            if not final_text or len(final_text) < 200:
                final_text = serp_snippet or final_text or ""

            if not final_text or len(final_text.strip()) < 150:
                # skip poor hits
                continue

            hit = {
                "query": q,
                "backend": "web",
                "source": link,
                "title": title,
                "content": final_text,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "_serp_meta": {"position": organic.index(item) if item in organic else None}
            }
            hits.append(hit)

        return hits[:top_k]
