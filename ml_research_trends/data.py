"""Pull paper data from Semantic Scholar and summarize it."""

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = "paperId,title,authors,year,citationCount,abstract,url,publicationVenue"


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
    """Search Semantic Scholar for each keyword and return one combined DataFrame."""
    # Grab the API key from the environment.
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Pass api_key=... or set API_KEY in your .env.")

    headers = {"x-api-key": api_key}
    fields = fields or DEFAULT_FIELDS

    all_papers = []
    seen_ids = set()
    last_request_time = None

    # Total number of API requests we'll need (for the progress bar).
    total_requests = sum(-(-max_results_per_keyword // 100) for _ in keywords)

    with tqdm(total=total_requests, desc="Fetching papers", unit="req") as pbar:
        for keyword in keywords:
            offset = 0
            pbar.set_postfix(keyword=keyword[:30])

            while offset < max_results_per_keyword:
                limit = min(100, max_results_per_keyword - offset)

                # Wait a bit between requests so we don't get rate limited.
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

                # Retry a few times if you get rate limited.
                last_request_time = time.time()
                for attempt in range(max_retries):
                    response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
                    if response.status_code == 429:
                        wait = 2 ** attempt
                        tqdm.write(f"Rate limited. Waiting {wait}s before retry...")
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

                    # Skip duplicates and anything outside the year range.
                    if not paper_id or paper_id in seen_ids:
                        continue
                    if year is None:
                        continue
                    if min_year is not None and year < min_year:
                        continue
                    if max_year is not None and year > max_year:
                        continue

                    authors = ", ".join(a.get("name", "") for a in paper.get("authors", []))
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

    # Clean up the columns a little.
    df["title"] = df["title"].fillna("").str.strip()
    df["abstract"] = df["abstract"].fillna("").str.strip()
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    if save_path:
        df.to_csv(save_path, index=False)

    return df


def summarize_topic_data(df):
    """Return a dict of quick stats about a topic."""
    missing = int(df["abstract"].isna().sum() + (df["abstract"] == "").sum())
    return {
        "total_papers": len(df),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "average_citations": round(df["citation_count"].mean(), 2),
        "median_citations": round(df["citation_count"].median(), 2),
        "missing_abstracts": missing,
        "top_venues": df["venue"].fillna("Unknown").value_counts().head(10),
    }


def analyze_topic_trends(df):
    """Group papers by year and compute counts and average citations."""
    trends = df.groupby("year").agg(
        paper_count=("paper_id", "count"),
        average_citations=("citation_count", "mean"),
        median_citations=("citation_count", "median"),
    )
    return trends.reset_index().sort_values("year")
