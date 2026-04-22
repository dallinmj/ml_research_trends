"""Data collection and aggregation utilities.

Fetches papers from the Semantic Scholar Graph API and produces tidy
``pandas.DataFrame`` objects suitable for downstream analysis and plotting.
"""

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = (
    "paperId,title,authors,year,citationCount,abstract,url,publicationVenue"
)


def collect_topic_data(
    keywords,
    max_results_per_keyword=100,
    min_year=None,
    max_year=None,
    save_path=None,
    api_key=None,
    request_delay=1.1,
    max_retries=5,
    fields=None,
):
    """Search Semantic Scholar for each keyword and return a combined dataframe
    of deduplicated papers.

    Parameters
    ----------
    keywords : list[str]
        Search queries to run against the Semantic Scholar API.
    max_results_per_keyword : int
        Hard cap on the number of papers fetched per keyword.
    min_year, max_year : int or None
        Optional inclusive year filter. If ``None`` (default), no filtering
        is applied on that side of the range.
    save_path : str or None
        If given, write the resulting dataframe to this CSV path.
    api_key : str or None
        Semantic Scholar API key. If ``None``, read from the ``API_KEY`` or
        ``SEMANTIC_SCHOLAR_API_KEY`` environment variable (``.env``-loaded).
    request_delay : float
        Minimum seconds between API requests (simple client-side throttle).
    max_retries : int
        Number of retries on HTTP 429 before raising.
    fields : str or None
        Comma-separated Semantic Scholar ``fields`` parameter. Defaults to
        a sensible set covering id/title/authors/year/citations/abstract/url
        /venue.
    """
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing API key. Pass api_key=... or set API_KEY / "
            "SEMANTIC_SCHOLAR_API_KEY in your environment."
        )

    headers = {"x-api-key": api_key}
    fields = fields or DEFAULT_FIELDS

    all_papers = []
    seen_ids = set()
    last_request_time = None

    total_requests = sum(
        -(-max_results_per_keyword // 100) for _ in keywords
    )

    with tqdm(total=total_requests, desc="Fetching papers", unit="req") as pbar:
        for keyword in keywords:
            offset = 0
            pbar.set_postfix(keyword=keyword[:30])

            while offset < max_results_per_keyword:
                limit = min(100, max_results_per_keyword - offset)

                if last_request_time is not None:
                    elapsed = time.time() - last_request_time
                    if elapsed < request_delay:
                        time.sleep(request_delay - elapsed)

                params = {
                    "query": keyword,
                    "limit": limit,
                    "offset": offset,
                    "fields": fields,
                }

                last_request_time = time.time()
                for attempt in range(max_retries):
                    response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
                    if response.status_code == 429:
                        wait = 2 ** attempt
                        tqdm.write(f"Rate limited (429). Waiting {wait}s before retry...")
                        time.sleep(wait)
                        last_request_time = time.time()
                        continue
                    response.raise_for_status()
                    break
                else:
                    raise RuntimeError("Exceeded retry limit due to repeated 429 errors.")
                papers = response.json().get("data", [])

                pbar.update(1)
                pbar.set_postfix(keyword=keyword[:30], collected=len(all_papers))

                if not papers:
                    break

                for paper in papers:
                    paper_id = paper.get("paperId")
                    year = paper.get("year")

                    if not paper_id or paper_id in seen_ids:
                        continue
                    if min_year is not None and (year is None or year < min_year):
                        continue
                    if max_year is not None and (year is None or year > max_year):
                        continue
                    if year is None:
                        continue

                    authors = ", ".join(
                        author.get("name", "") for author in paper.get("authors", [])
                    )

                    venue = paper.get("publicationVenue") or {}

                    all_papers.append({
                        "paper_id": paper_id,
                        "title": paper.get("title"),
                        "authors": authors,
                        "year": year,
                        "citation_count": paper.get("citationCount", 0),
                        "abstract": paper.get("abstract"),
                        "url": paper.get("url"),
                        "venue": venue.get("name"),
                        "keyword": keyword,
                    })

                    seen_ids.add(paper_id)

                offset += len(papers)

                if len(papers) < limit:
                    break

    df = pd.DataFrame(all_papers)

    if df.empty:
        raise ValueError("No papers were collected. Try different keywords.")

    df["title"] = df["title"].fillna("").str.strip()
    df["abstract"] = df["abstract"].fillna("").str.strip()
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    if save_path:
        df.to_csv(save_path, index=False)

    return df


def summarize_topic_data(df):
    """Return a dict of quick descriptive stats for a topic dataframe."""
    return {
        "total_papers": len(df),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "average_citations": round(df["citation_count"].mean(), 2),
        "median_citations": round(df["citation_count"].median(), 2),
        "missing_abstracts": int(df["abstract"].isna().sum() + (df["abstract"] == "").sum()),
        "top_venues": df["venue"].fillna("Unknown").value_counts().head(10),
    }


def analyze_topic_trends(df):
    """Aggregate ``df`` by year into paper counts and citation averages."""
    return (
        df.groupby("year")
        .agg(
            paper_count=("paper_id", "count"),
            average_citations=("citation_count", "mean"),
            median_citations=("citation_count", "median"),
        )
        .reset_index()
        .sort_values("year")
    )
