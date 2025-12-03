# search_agent/connectors.py
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
    Minimal CourtListener v4 client with graceful fallback.
    - Tries v4 API (requires COURTLISTENER_API_KEY in .env).
    - If API fails (403/401/timeout/DNS/404), falls back to scraping the public search page.
    - fetch() accepts either the public /opinion/... URL or the API URL and converts where possible.
    """
    BASE = "https://www.courtlistener.com/api/rest/v4/opinions/"
    KEY = os.getenv("COURTLISTENER_API_KEY")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": USER_AGENT["User-Agent"],
            "Accept": "application/json",
        }
        if self.KEY:
            headers["Authorization"] = f"Token {self.KEY}"
        return headers

    def search(self, query: str, page_size: int = 1) -> List[Dict[str, Any]]:
        # Try v4 API
        try:
            api_url = f"{self.BASE}?q={requests.utils.requote_uri(query)}&page_size={page_size}"
            r = requests.get(api_url, headers=self._headers(), timeout=15)
            if r.status_code == 200:
                data = r.json()
                items = []
                for item in data.get("results", []):
                    items.append({
                        "id": item.get("id"),
                        "url": item.get("absolute_url"),
                        "case_name": item.get("case_name"),
                        "citation": item.get("citation"),
                    })
                return items
            else:
                # Non-200: log & fallback
                print(f"[CourtListener] API v4 returned {r.status_code}: {r.text[:300]}")
        except Exception as e:
            print("[CourtListener] v4 request failed; falling back to scraping. Error:", repr(e))

        # Fallback: scrape public search page
        try:
            search_url = f"https://www.courtlistener.com/?q={requests.utils.requote_uri(query)}"
            page = requests.get(search_url, headers=USER_AGENT, timeout=15)
            page.raise_for_status()

            soup = BeautifulSoup(page.text, "html.parser")
            items = []
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
                title = a.get_text(" ", strip=True)
                items.append({
                    "id": None,
                    "url": href,
                    "case_name": title,
                    "citation": None
                })
                if len(items) >= page_size:
                    break
            return items
        except Exception as e:
            print("[CourtListener] Scraping fallback failed:", repr(e))
            return []

    def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch opinion content.
        Accepts either:
        - a public URL like https://www.courtlistener.com/opinion/{id}/...
        - an API URL like https://www.courtlistener.com/api/rest/v4/opinions/{id}/
        Returns dict with 'content' (text), 'case_name', 'citation', 'source', 'retrieved_at'.
        """
        # Convert public opinion URL to API if possible
        try:
            if "/opinion/" in url and "api/rest" not in url:
                parts = url.rstrip("/").split("/")
                opinion_id = next((p for p in parts if p.isdigit()), None)
                if opinion_id:
                    url = f"https://www.courtlistener.com/api/rest/v4/opinions/{opinion_id}/"
        except Exception:
            pass

        # Try API fetch
        try:
            r = requests.get(url, headers=self._headers(), timeout=15)
            r.raise_for_status()
            data = r.json()
            text = data.get("plain_text") or data.get("html_with_citations") or ""
            return {
                "backend": "courtlistener",
                "source": data.get("absolute_url") or url,
                "case_name": data.get("case_name"),
                "citation": data.get("citation"),
                "content": text,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        except Exception as e:
            # API fetch failed — try HTML fetch
            print("[CourtListener] API fetch failed; trying HTML fallback:", repr(e))

        # HTML fallback
        try:
            page = requests.get(url, headers=USER_AGENT, timeout=15)
            page.raise_for_status()
            soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = "\n\n".join(paragraphs)
            return {
                "backend": "courtlistener",
                "source": url,
                "case_name": None,
                "citation": None,
                "content": text,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        except Exception as e:
            print("[CourtListener] HTML fallback also failed:", repr(e))
            return {"backend": "courtlistener", "source": url, "content": "", "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


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
