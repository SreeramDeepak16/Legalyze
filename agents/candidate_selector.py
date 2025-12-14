"""
candidate_selector.py

Utilities to score and select the best candidate document returned
from CourtListener (or other legal search backend).

Exports:
  - select_best_candidate(results, query_case, query_citation, min_score_threshold, top_k_debug)
  - score_candidate(candidate, query_case, query_citation)

Short-circuit behavior:
  If the query contains an explicit U.S. citation (e.g. "418 U.S. 683")
  and a candidate's structured citation list contains a matching
  normalized citation, that candidate is auto-selected.
"""

from typing import List, Dict, Optional, Tuple, Any
import logging
import re

# ---------------------------------------------------------------------
# Logger setup (library-safe)
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
# Do NOT configure handlers or levels here. Applications should configure logging globally.

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

_CITATION_US_RE = re.compile(
    r'(\d{1,4})\s*\.?\s*(u(?:nited)?\s*\.?\s*s\.? )?\.?\s*(\d{1,4})',
    re.IGNORECASE
)

def normalize_text(s: Optional[Any]) -> str:
    """Lowercase and collapse whitespace for matching; works for lists too."""
    if s is None:
        return ""
    if isinstance(s, (list, tuple)):
        s = " ".join(str(x) for x in s)
    return re.sub(r'\s+', ' ', str(s)).strip().lower()

def citation_variants(citation: Optional[str]) -> List[str]:
    """
    Produce a set of normalized citation strings from input.

    Examples:
      "418 U.S. 683" -> ["418 u.s. 683", "418 u s 683", "418 us 683", ...]
    """
    if not citation:
        return []

    cit = str(citation).strip()
    m = _CITATION_US_RE.search(cit)
    variants = set()

    if m:
        first = m.group(1)
        # use canonical mid form
        mid = "u.s."
        last = m.group(3) if m.lastindex and m.lastindex >= 3 else m.group(2)
        forms = [
            f"{first} {mid} {last}",
            f"{first} {mid}{last}",
            f"{first} u s {last}",
            f"{first} us {last}",
            f"{first} u.s {last}",
            f"{first}u.s.{last}",
            f"{first}u s{last}",
        ]
        for f in forms:
            variants.add(normalize_text(f))
    else:
        norm = normalize_text(citation)
        variants.add(norm)
        digits = "".join(re.findall(r'\d+', cit))
        if digits:
            variants.add(digits)

    return sorted(variants)

# ---------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------

def score_candidate(candidate: Dict[str, Any],
                    query_case: str,
                    query_citation: str) -> Tuple[int, Dict[str, Any]]:
    """
    Score a single candidate.

    Returns:
        (score:int, details:dict)
    """
    details: Dict[str, Any] = {"hits": []}
    base_score = 0

    # Normalized fields
    case_name = normalize_text(
        candidate.get("case_name")
        or candidate.get("title")
        or candidate.get("name")
        or ""
    )

    citations_field = (
        candidate.get("citation")
        or candidate.get("citations")
        or candidate.get("citation_list")
        or candidate.get("citation_str")
        or candidate.get("citation_text")
    )

    if isinstance(citations_field, str):
        citations_list = [citations_field]
    elif isinstance(citations_field, (list, tuple)):
        citations_list = citations_field
    else:
        citations_list = []

    citations_norm = [normalize_text(x) for x in citations_list]

    content = normalize_text(
        candidate.get("content")
        or candidate.get("plain_text")
        or candidate.get("snippet")
        or ""
    )
    court = normalize_text(
        candidate.get("court")
        or candidate.get("case_court")
        or candidate.get("source_court")
        or candidate.get("court_name")
        or ""
    )

    query_case_norm = normalize_text(query_case)
    query_citation_norm = normalize_text(query_citation)

    # Heuristic 1: exact case name match
    if query_case_norm and query_case_norm == case_name:
        base_score += 80
        details["hits"].append(("case_name_exact", query_case_norm))
    elif query_case_norm and query_case_norm in case_name:
        base_score += 35
        details["hits"].append(("case_name_contains", query_case_norm))

    # Heuristic 2: citation list contains citation
    for qc in citation_variants(query_citation_norm):
        for c in citations_norm:
            if qc == c or qc in c or c in qc:
                base_score += 120
                details["hits"].append(("citation_list", qc, c))

    # Heuristic 3: citation appears in content
    if query_citation_norm and query_citation_norm in content:
        base_score += 30
        details["hits"].append(("citation_in_content", query_citation_norm))

    query_digits = "".join(re.findall(r'\d+', query_citation_norm)) if query_citation_norm else ""
    if query_digits and query_digits in content:
        base_score += 10
        details["hits"].append(("citation_digits_in_content", query_digits))

    # Heuristic 4: Supreme Court boost
    if "supreme" in court:
        base_score += 10
        details["hits"].append(("court_supreme", court))

    # Heuristic 5: token overlap
    qtokens = [t for t in re.split(r'\W+', query_case_norm) if t]
    overlap = 0
    hay = " ".join([case_name, content, " ".join(citations_norm)])
    for t in qtokens[:8]:
        if t and t in hay:
            overlap += 1

    base_score += overlap * 4
    if overlap:
        details["hits"].append(("token_overlap", overlap, qtokens[:8]))

    # Heuristic 6: content length boost/penalty
    content_len = len(content)
    if content_len > 50_000:
        base_score += 6
    elif content_len > 10_000:
        base_score += 4
    elif content_len > 1000:
        base_score += 2
    elif content_len < 200:
        base_score -= 8
        details["hits"].append(("short_content_penalty", content_len))

    # Heuristic 7: ID similarity with citation digits
    cid = str(
        candidate.get("id")
        or candidate.get("doc_id")
        or candidate.get("source")
        or ""
    )
    if query_digits and query_digits in cid:
        base_score += 5
        details["hits"].append(("id_matches_citation_digits", cid))

    final_score = int(base_score)
    details["raw_score"] = final_score
    return final_score, details

# ---------------------------------------------------------------------
# Selection function (public)
# ---------------------------------------------------------------------

def select_best_candidate(results: List[Dict],
                          query_case: str,
                          query_citation: str,
                          min_score_threshold: int = 30,
                          top_k_debug: int = 5) -> Optional[Dict]:
    """
    Choose the best candidate from results. Returns the candidate dict or None if none pass threshold.
    """

    if not results:
        logger.info("select_best_candidate: no input results")
        return None

    # Short-circuit citation match (improved: prefer majority opinion when possible)
    query_citation_norm = normalize_text(query_citation)
    query_citation_variants = citation_variants(query_citation_norm)

    def _looks_like_dissent_or_concurrence(cand_text: str, cand_meta: Dict[str, Any]) -> bool:
        """Return True if document text/meta strongly suggests dissent/concurring (i.e., not Opinion of the Court)."""
        if not cand_text:
            return False
        txt = normalize_text(cand_text)
        # obvious markers of dissent/concurrence
        dissent_markers = ["dissent", "dissenting", "concurring", "dissent from", "concurring in part", "concurs"]
        # markers of majority/opinion of the court
        majority_markers = ["opinion of the court", "for the court", "opinion by", "rehnquist, j.", "scalia, j.", "kennedy, j."]
        for m in dissent_markers:
            if m in txt:
                return True
        # also check explicit fields (author/opinion_type) if present
        ot = normalize_text(cand_meta.get("opinion_type") or cand_meta.get("type") or cand_meta.get("opinion") or "")
        if any(m in ot for m in ["dissent", "concurring", "concurrence"]):
            return True
        # If it contains an obvious majority marker, treat as NOT dissent
        if any(m in txt for m in majority_markers):
            return False
        return False

    # first pass: try to pick a candidate with matching citation that does NOT look like a dissent
    for qc in query_citation_variants:
        for cand in results:
            try:
                citations_list = (
                    cand.get("citation")
                    or cand.get("citations")
                    or cand.get("citation_list")
                    or []
                )
                if isinstance(citations_list, str):
                    citations_list = [citations_list]

                citations_norm = [normalize_text(str(x)) for x in citations_list]

                for c in citations_norm:
                    if qc == c or qc in c or c in qc:
                        cand_text = cand.get("content") or cand.get("plain_text") or cand.get("snippet") or ""
                        if not _looks_like_dissent_or_concurrence(cand_text, cand):
                            selected = dict(cand)
                            auto_score = 10**6
                            selected["_selection_score"] = auto_score
                            selected["_selection_details"] = {
                                "hits": [("citation_list_shortcircuit", qc)],
                                "score": auto_score,
                            }
                            logger.info(
                                "Short-circuit citation match (preferred majority) found. Auto-selected id=%s (qc=%s).",
                                cand.get("id") or "<no-id>", qc
                            )
                            return selected
            except Exception as e:
                logger.debug("citation short-circuit check failed for id=%s: %s", cand.get("id") or "<no-id>", e)

    # second pass: if no majority-like candidate found, fall back to original behavior
    for qc in query_citation_variants:
        for cand in results:
            try:
                citations_list = (
                    cand.get("citation")
                    or cand.get("citations")
                    or cand.get("citation_list")
                    or []
                )
                if isinstance(citations_list, str):
                    citations_list = [citations_list]

                citations_norm = [normalize_text(str(x)) for x in citations_list]

                for c in citations_norm:
                    if qc == c or qc in c or c in qc:
                        selected = dict(cand)
                        auto_score = 10**6
                        selected["_selection_score"] = auto_score
                        selected["_selection_details"] = {
                            "hits": [("citation_list_shortcircuit", qc)],
                            "score": auto_score,
                        }
                        logger.info(
                            "Short-circuit citation match found (fallback). Auto-selected id=%s (qc=%s).",
                            cand.get("id") or "<no-id>", qc
                        )
                        return selected

            except Exception as e:
                logger.debug(
                    "citation short-circuit check failed for id=%s: %s",
                    cand.get("id") or "<no-id>", e
                )

    # Compute scores normally
    scored: List[Tuple[int, Dict, Dict]] = []

    for candidate in results:
        try:
            sc, details = score_candidate(candidate, query_case, query_citation)
        except Exception as e:
            logger.exception(
                "score_candidate raised for id=%s: %s",
                candidate.get("id") or "<no-id>", e
            )
            sc = -9999
            details = {"error": str(e)}

        scored.append((sc, candidate, details))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Debug top scores
    logger.info("Top candidates (score, id, title/snippet):")
    for i, (sc, cand, det) in enumerate(scored[:top_k_debug]):
        cid = cand.get("id") or cand.get("doc_id") or cand.get("source") or "<no-id>"
        title = (
            cand.get("title")
            or cand.get("case_name")
            or cand.get("name")
            or cand.get("snippet")
            or ""
        )
        hits = det.get("hits") if isinstance(det, dict) else []
        logger.info(
            "  %d) score=%d id=%s title=%s hits=%s",
            i + 1, sc, cid,
            (title[:200] + ("..." if len(title) > 200 else "")),
            hits
        )

    best_score, best_candidate, best_details = scored[0]
    logger.info(
        "Best candidate id=%s score=%d",
        best_candidate.get("id") or best_candidate.get("doc_id") or "<no-id>",
        best_score
    )

    if best_score < min_score_threshold:
        logger.warning(
            "No candidate passed the minimum score threshold (%d). Best score was %d. Returning None.",
            min_score_threshold, best_score
        )
        return None

    selected = dict(best_candidate)
    selected["_selection_score"] = best_score
    selected["_selection_details"] = {
        "score_candidate": best_score,
        "score_candidate_details": best_details,
    }
    return selected