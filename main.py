import os
import time
import numpy as np
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


def embed_and_plot_abstracts(
    df,
    model_name="Qwen/Qwen3-Embedding-4B",
    reducer="umap",
    batch_size=8,
    max_papers=None,
    cache_path=None,
    topic_name="Topic",
    random_state=42,
):
    """Embed paper abstracts with a Qwen3 embedding model and plot a 2-D
    projection colored by publication year.

    Parameters
    ----------
    df : DataFrame returned by ``collect_topic_data``.
    model_name : HuggingFace model id for the embedding model.
    reducer : "umap", "tsne", or "pca".
    batch_size : batch size used when encoding abstracts.
    max_papers : optional cap on number of abstracts (for quick iteration).
    cache_path : optional .npy path to cache embeddings keyed to the df order.
    """
    work = df.copy()
    work = work[work["abstract"].astype(str).str.len() > 0].reset_index(drop=True)
    if max_papers is not None:
        work = work.head(max_papers).reset_index(drop=True)

    if work.empty:
        raise ValueError("No abstracts available to embed.")

    embeddings = None
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(work):
            embeddings = cached
            print(f"Loaded cached embeddings from {cache_path}")

    if embeddings is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {device}...")
        model = SentenceTransformer(model_name, device=device)

        abstracts = work["abstract"].tolist()
        embeddings = model.encode(
            abstracts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if cache_path:
            np.save(cache_path, embeddings)
            print(f"Saved embeddings to {cache_path}")

    reducer = reducer.lower()
    if reducer == "umap":
        import umap

        reducer_obj = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=random_state,
        )
        coords = reducer_obj.fit_transform(embeddings)
    elif reducer == "tsne":
        from sklearn.manifold import TSNE

        coords = TSNE(
            n_components=2,
            metric="cosine",
            init="pca",
            perplexity=min(30, max(5, len(work) // 4)),
            random_state=random_state,
        ).fit_transform(embeddings)
    elif reducer == "pca":
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=random_state).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    work["x"] = coords[:, 0]
    work["y"] = coords[:, 1]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 7))
    scatter = sns.scatterplot(
        data=work,
        x="x",
        y="y",
        hue="year",
        palette="viridis",
        s=35,
        alpha=0.8,
        edgecolor="none",
    )
    plt.title(f"{topic_name} abstract embeddings ({reducer.upper()}) colored by year")
    plt.xlabel(f"{reducer.upper()}-1")
    plt.ylabel(f"{reducer.upper()}-2")
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles, labels, title="Year", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return work[["paper_id", "title", "year", "x", "y"]]


def plot_landmark_timeline(
    df,
    landmark_date,
    landmark_label="Landmark paper",
    freq="M",
    cumulative=False,
    smooth_window=3,
    topic_name="Topic",
):
    """Plot a continuous-time histogram of paper volume with a vertical marker
    at a landmark paper's publication date, so you can see how quickly
    downstream research picks up.

    Parameters
    ----------
    df : DataFrame returned by ``collect_topic_data``. Uses the ``year`` column
        (other date columns are used if present: ``publication_date``).
    landmark_date : str or Timestamp. e.g. "2020-05-28" for GPT-3.
    freq : pandas offset alias. "M" = monthly, "Q" = quarterly, "Y" = yearly.
    cumulative : if True, plot cumulative paper count instead of per-bucket.
    smooth_window : rolling mean window (in buckets) to smooth the curve.
        Set to 1 to disable smoothing.
    """
    work = df.copy()

    if "publication_date" in work.columns:
        dates = pd.to_datetime(work["publication_date"], errors="coerce")
    else:
        dates = pd.to_datetime(
            work["year"].astype("Int64").astype(str) + "-07-01",
            errors="coerce",
        )
    work["date"] = dates
    work = work.dropna(subset=["date"])

    landmark = pd.to_datetime(landmark_date)

    counts = (
        work.set_index("date")
        .assign(n=1)["n"]
        .resample(freq)
        .sum()
        .sort_index()
    )

    if counts.empty:
        raise ValueError("No dated papers available.")

    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq=freq)
    counts = counts.reindex(full_index, fill_value=0)

    if cumulative:
        series = counts.cumsum()
        y_label = "Cumulative papers"
    else:
        series = counts
        if smooth_window and smooth_window > 1:
            series = series.rolling(smooth_window, min_periods=1, center=True).mean()
        y_label = f"Papers per {freq} (rolling mean, w={smooth_window})"

    before = int(counts[counts.index < landmark].sum())
    after = int(counts[counts.index >= landmark].sum())

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(series.index, series.values, color="steelblue", linewidth=2)
    ax.fill_between(series.index, series.values, alpha=0.2, color="steelblue")

    ax.axvline(landmark, color="crimson", linestyle="--", linewidth=2)
    ax.annotate(
        f"{landmark_label}\n{landmark.date()}",
        xy=(landmark, ax.get_ylim()[1] * 0.95),
        xytext=(8, -10),
        textcoords="offset points",
        color="crimson",
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    ax.set_title(
        f"{topic_name}: research volume around {landmark_label}\n"
        f"{before} papers before  |  {after} papers after"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({"date": series.index, "value": series.values})


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

    # Uncomment to embed abstracts with Qwen3-4B and view a 2D projection
    # by year. First run will download the model (~8GB) and cache embeddings.
    # embed_and_plot_abstracts(
    #     df,
    #     model_name="Qwen/Qwen3-Embedding-4B",
    #     reducer="umap",
    #     batch_size=8,
    #     cache_path="rag_qwen3_embeddings.npy",
    #     topic_name="RAG",
    # )

    # Uncomment to see research volume around a landmark paper. Example:
    # the original RAG paper (Lewis et al., 2020).
    # plot_landmark_timeline(
    #     df,
    #     landmark_date="2020-05-22",
    #     landmark_label="Lewis et al. — RAG",
    #     freq="M",
    #     smooth_window=3,
    #     topic_name="RAG",
    # )


if __name__ == "__main__":
    main()