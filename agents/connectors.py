from dotenv import load_dotenv
load_dotenv()

import os
import time
import requests
from typing import Dict, Any, Optional, Callable, List
from bs4 import BeautifulSoup

USER_AGENT = {"User-Agent": "mirix-search-agent/1.0 (+https://example.org)"}


class CourtListenerClient:
    """
    CourtListener client that:
    - Uses /api/rest/v4/search/ to find clusters with opinions.
    - For each result, fetches the underlying opinion via /opinions/{id}/.
    - Extracts plain text (or strips HTML as a fallback).
    - On API failure/timeout, falls back to scraping the public search page.

    High-level helper:
        search_with_content(query, max_results)
        -> list[ {id, url, case_name, citation, content, retrieved_at} ]
    """

    API_ROOT = "https://www.courtlistener.com/api/rest/v4"
    SEARCH_URL = f"{API_ROOT}/search/"
    OPINIONS_URL = f"{API_ROOT}/opinions/"
    KEY = os.getenv("COURTLISTENER_API_KEY")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": USER_AGENT["User-Agent"],
            "Accept": "application/json",
        }
        if self.KEY:
            headers["Authorization"] = f"Token {self.KEY}"
        return headers

    # ---------- low-level opinion fetch ---------- #
    def _fetch_opinion_by_id(self, opinion_id: int) -> Dict[str, Any]:
        """
        Fetch one opinion by ID from /opinions/{id}/ and return dict with:
            { "case_name", "citation", "content", "retrieved_at" }
        """
        url = f"{self.OPINIONS_URL}{opinion_id}/"
        try:
            # a bit more generous timeout here
            r = requests.get(url, headers=self._headers(), timeout=25)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("[CourtListener] _fetch_opinion_by_id API error:", repr(e))
            return {
                "case_name": None,
                "citation": None,
                "content": "",
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        # Prefer plain_text, but fall back to any HTML field.
        text = (data.get("plain_text") or "").strip()
        if not text:
            html = (
                data.get("html_with_citations")
                or data.get("html")
                or data.get("html_lawbox")
                or data.get("html_columbia")
                or ""
            )
            if html:
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text("\n", strip=True)

        return {
            "case_name": data.get("case_name"),
            "citation": data.get("citation"),
            "content": text or "",
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # ---------- fallback: scrape public search page ---------- #
    def _fallback_scrape_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Scrape https://www.courtlistener.com/?q=... for /opinion/ links,
        extract opinion ID from URL, and then fetch via _fetch_opinion_by_id.
        """
        print("[CourtListener] Using HTML search fallback...")
        search_url = f"https://www.courtlistener.com/?q={requests.utils.requote_uri(query)}"
        items: List[Dict[str, Any]] = []

        try:
            page = requests.get(search_url, headers=USER_AGENT, timeout=25)
            page.raise_for_status()
        except Exception as e:
            print("[CourtListener] Fallback search page request failed:", repr(e))
            return items

        soup = BeautifulSoup(page.text, "html.parser")
        seen = set()

        for a in soup.select("a[href*='/opinion/']"):
            href = a.get("href")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.courtlistener.com" + href
            if href in seen:
                continue
            seen.add(href)

            # Extract numeric opinion id from URL
            opinion_id: Optional[int] = None
            try:
                parts = href.rstrip("/").split("/")
                for p in parts:
                    if p.isdigit():
                        opinion_id = int(p)
                        break
            except Exception:
                opinion_id = None

            if opinion_id is not None:
                opinion = self._fetch_opinion_by_id(opinion_id)
                case_name = opinion["case_name"] or a.get_text(" ", strip=True)
                items.append({
                    "id": opinion_id,
                    "url": href,
                    "case_name": case_name,
                    "citation": opinion["citation"],
                    "content": opinion["content"],
                    "retrieved_at": opinion["retrieved_at"],
                })
            else:
                # if we can't parse an id, we could still scrape raw HTML, but we'll skip for now
                continue

            if len(items) >= max_results:
                break

        return items

    # ---------- high-level: search + full content ---------- #
    def search_with_content(self, query: str, max_results: int = 1) -> List[Dict[str, Any]]:
        """
        1) Try /api/rest/v4/search/ with the query.
        2) For each hit, grab the first opinion's ID and fetch via /opinions/{id}/.
        3) On failure/timeout, fall back to scraping the HTML search page.

        Returns list of dicts:
           {
             "id": opinion_id,
             "url": "https://www.courtlistener.com/opinion/.../",
             "case_name": "Brown v. Board of Education",
             "citation": ["978 F.2d 585", ...] or None,
             "content": "<plain text of opinion>",
             "retrieved_at": "<iso8601>",
           }
        """
        results: List[Dict[str, Any]] = []

        # ---- primary: REST v4 search API ---- #
        try:
            params = {
                "q": query,
                "page_size": max_results,
                "cluster__has_opinion": "true",
            }
            # longer timeout to reduce ReadTimeout issues
            r = requests.get(self.SEARCH_URL, headers=self._headers(), params=params, timeout=25)
            r.raise_for_status()
            data = r.json()

            for item in data.get("results", [])[:max_results]:
                opinions = item.get("opinions") or []
                if not opinions:
                    continue

                op_id = opinions[0].get("id")
                if not op_id:
                    continue

                abs_url = item.get("absolute_url")
                if abs_url and not abs_url.startswith("http"):
                    abs_url = "https://www.courtlistener.com" + abs_url

                opinion = self._fetch_opinion_by_id(op_id)

                results.append({
                    "id": op_id,
                    "url": abs_url,
                    "case_name": item.get("caseName") or item.get("caseNameFull"),
                    "citation": item.get("citation"),
                    "content": opinion["content"],
                    "retrieved_at": opinion["retrieved_at"],
                })

        except Exception as e:
            print("[CourtListener] /search/ request failed:", repr(e))

        # ---- if API path failed or gave nothing, fallback to scraping ---- #
        if not results:
            results = self._fallback_scrape_search(query, max_results=max_results)

        return results

# ------------------- SerpAPI helper ------------------- #
def serpapi_search_json(query: str, num: int = 1) -> Dict[str, Any]:
    """
    Lightweight SerpAPI wrapper returning parsed JSON.
    Requires SERPAPI_API_KEY in .env.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise EnvironmentError("SERPAPI_API_KEY not set in environment")

    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "api_key": api_key, "num": num}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def make_serpapi_tool() -> Optional[Callable[[str], Any]]:
    """
    Return a callable that wraps serpapi_search_json(query) -> dict
    Useful if you want a tool-like function to plug into other code.
    """
    if not os.getenv("SERPAPI_API_KEY"):
        return None

    def run(q: str):
        try:
            return serpapi_search_json(q)
        except Exception as e:
            return {"error": str(e)}
    return run