"""Literature tool - 'has anyone published on this target?'

After the archive says a target is (or isn't) a catalogued planet, this asks a
different question: does the *literature* mention it? A candidate can be absent
from the confirmed-planet table yet already studied in papers - that still
lowers novelty.

Sources:
  * arXiv API  - no key required. Terms of use: <=1 request / 3 s, single
                 connection. We make one request per call and are polite.
  * NASA ADS   - richer, but needs a free token. Set the ADS_TOKEN environment
                 variable to activate; otherwise that half is skipped cleanly.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET

import requests

from exoscout import cache

ARXIV_URL = "http://export.arxiv.org/api/query"
ARXIV_TTL = 12 * 3600
ADS_URL = "https://api.adsabs.harvard.edu/v1/search/query"
TIMEOUT = 30
_ATOM = "{http://www.w3.org/2005/Atom}"

# Simple module-level throttle to honour arXiv's 1-request-per-3-seconds rule.
_last_arxiv_call = 0.0


def _query_terms(tic_id: int | None, toi: float | None, extra: str | None = None) -> list[str]:
    """Build the search strings most likely to appear in a paper."""
    terms: list[str] = []
    if toi is not None:
        base = f"{toi:g}"
        terms += [f'"TOI-{base}"', f'"TOI {base}"']
        star = f"{int(toi)}"          # star-level designation, e.g. TOI-700
        terms += [f'"TOI-{star}"']
    if tic_id is not None:
        terms += [f'"TIC {tic_id}"', f'"TIC{tic_id}"']
    if extra:
        terms.append(f'"{extra}"')
    # de-dupe, keep order
    return list(dict.fromkeys(terms))


def search_arxiv(query_terms: list[str], max_results: int = 8) -> dict:
    """Search arXiv for any of the query terms (OR'd). Returns structured hits."""
    global _last_arxiv_call
    source = "arXiv API"
    if not query_terms:
        return {"ok": True, "hits": [], "source": source, "query": ""}

    search_query = " OR ".join(f"all:{t}" for t in query_terms)
    cache_key = f"{search_query}|{max_results}"

    try:
        xml = cache.get("arxiv", cache_key, ARXIV_TTL)
        if xml is None:
            # Throttle only on a real network call (honours arXiv's 1/3 s rule).
            wait = 3.0 - (time.time() - _last_arxiv_call)
            if wait > 0:
                time.sleep(wait)
            r = requests.get(
                ARXIV_URL,
                params={
                    "search_query": search_query,
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                timeout=TIMEOUT,
                headers={"User-Agent": "ExoScout/0.1 (portfolio project)"},
            )
            _last_arxiv_call = time.time()
            r.raise_for_status()
            xml = r.text
            cache.put("arxiv", cache_key, xml)
        root = ET.fromstring(xml)

        hits = []
        for entry in root.findall(f"{_ATOM}entry"):
            title = (entry.findtext(f"{_ATOM}title") or "").strip().replace("\n", " ")
            summary = (entry.findtext(f"{_ATOM}summary") or "").strip().replace("\n", " ")
            published = (entry.findtext(f"{_ATOM}published") or "")[:10]
            url = entry.findtext(f"{_ATOM}id") or ""
            authors = [
                a.findtext(f"{_ATOM}name") for a in entry.findall(f"{_ATOM}author")
            ]
            hits.append({
                "title": title,
                "authors": ", ".join(a for a in authors[:3] if a) + ("  et al." if len(authors) > 3 else ""),
                "published": published,
                "url": url,
                "abstract": summary[:280] + ("..." if len(summary) > 280 else ""),
            })
        return {"ok": True, "hits": hits, "source": source, "query": search_query}
    except requests.RequestException as e:
        return {"ok": False, "error": f"arXiv request failed: {e}", "source": source, "hits": []}
    except ET.ParseError as e:
        return {"ok": False, "error": f"arXiv parse failed: {e}", "source": source, "hits": []}


def search_ads(query_terms: list[str], max_results: int = 8) -> dict:
    """Search NASA ADS. Skipped cleanly if ADS_TOKEN is not set."""
    source = "NASA ADS API"
    token = os.environ.get("ADS_TOKEN", "").strip()
    if not token:
        return {"ok": True, "skipped": True, "reason": "ADS_TOKEN not set", "hits": [], "source": source}
    if not query_terms:
        return {"ok": True, "hits": [], "source": source}

    q = " OR ".join(query_terms)
    try:
        r = requests.get(
            ADS_URL,
            params={"q": q, "rows": max_results, "fl": "title,author,year,bibcode,abstract",
                    "sort": "date desc"},
            headers={"Authorization": f"Bearer {token}",
                     "User-Agent": "ExoScout/0.1 (portfolio project)"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
        hits = [{
            "title": (d.get("title") or [""])[0],
            "authors": ", ".join((d.get("author") or [])[:3]) + ("  et al." if len(d.get("author") or []) > 3 else ""),
            "published": str(d.get("year", "")),
            "url": f"https://ui.adsabs.harvard.edu/abs/{d.get('bibcode','')}",
            "abstract": (d.get("abstract", "") or "")[:280],
        } for d in docs]
        return {"ok": True, "hits": hits, "source": source}
    except requests.RequestException as e:
        return {"ok": False, "error": f"ADS request failed: {e}", "source": source, "hits": []}


def check_novelty(tic_id: int | None = None, toi: float | None = None, star_name: str | None = None) -> dict:
    """Combine arXiv + ADS into a single novelty read for the target."""
    terms = _query_terms(tic_id, toi, star_name)
    arxiv = search_arxiv(terms)
    ads = search_ads(terms)

    n_arxiv = len(arxiv.get("hits", []))
    n_ads = len(ads.get("hits", []))
    total = n_arxiv + n_ads

    if not (arxiv.get("ok") or ads.get("ok")):
        headline = "Literature search unavailable"
    elif total == 0:
        headline = "No literature matches - potentially novel"
    else:
        headline = f"{total} literature match(es) - target already studied"

    return {
        "ok": arxiv.get("ok", False) or ads.get("ok", False),
        "terms": terms,
        "arxiv": arxiv,
        "ads": ads,
        "n_matches": total,
        "summary": headline,
        "source": "arXiv + ADS",
    }
