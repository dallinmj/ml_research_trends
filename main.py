import os
import time
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def collect_topic_data(
    keywords,
    max_results_per_keyword=100,
    min_year=2018,
    max_year=2026,
    save_path=None
):
    load_dotenv()
    api_key = os.getenv("API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    if not api_key:
        raise ValueError("Missing API key. Put API_KEY in your .env file.")

    headers = {
        "x-api-key": api_key
    }

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
                    if elapsed < 1.1:
                        time.sleep(1.1 - elapsed)

                params = {
                    "query": keyword,
                    "limit": limit,
                    "offset": offset,
                    "fields": "paperId,title,authors,year,citationCount,abstract,url,publicationVenue"
                }

                last_request_time = time.time()
                for attempt in range(5):
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
                    if year is None or year < min_year or year > max_year:
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
                        "keyword": keyword
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
    summary = {
        "total_papers": len(df),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "average_citations": round(df["citation_count"].mean(), 2),
        "median_citations": round(df["citation_count"].median(), 2),
        "missing_abstracts": int(df["abstract"].isna().sum() + (df["abstract"] == "").sum()),
        "top_venues": df["venue"].fillna("Unknown").value_counts().head(10)
    }

    return summary


def analyze_topic_trends(df):
    trend_df = (
        df.groupby("year")
        .agg(
            paper_count=("paper_id", "count"),
            average_citations=("citation_count", "mean"),
            median_citations=("citation_count", "median")
        )
        .reset_index()
        .sort_values("year")
    )

    return trend_df


def plot_topic_counts_by_year(trend_df, topic_name="Topic"):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=trend_df, x="year", y="paper_count")

    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title(f"{topic_name} Papers by Year")
    plt.tight_layout()
    plt.show()


def main():
    keywords = [
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "RAG"
    ]

    df = collect_topic_data(
        keywords=keywords,
        max_results_per_keyword=100,
        min_year=2018,
        max_year=2026,
        save_path="rag_papers.csv"
    )

    summary = summarize_topic_data(df)
    trends = analyze_topic_trends(df)

    print("SUMMARY")
    print("Total papers:", summary["total_papers"])
    print("Year range:", summary["year_min"], "-", summary["year_max"])
    print("Average citations:", summary["average_citations"])
    print("Median citations:", summary["median_citations"])
    print("Missing abstracts:", summary["missing_abstracts"])
    print("\nTop venues:")
    print(summary["top_venues"])

    print("\nTRENDS")
    print(trends)

    plot_topic_counts_by_year(trends, topic_name="RAG")


if __name__ == "__main__":
    main()