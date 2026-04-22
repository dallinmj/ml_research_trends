'''
ml_research_trends — package API reference
==========================================

A small toolkit for collecting papers from Semantic Scholar on a topic and
analyzing / visualizing publication trends.

Package layout
--------------
ml_research_trends/
    data.py           -> collection + summary + aggregation
    visualization.py  -> plots + embeddings

Quick start
-----------
    from ml_research_trends import (
        collect_topic_data, summarize_topic_data, analyze_topic_trends,
        plot_topic_counts_by_year, embed_and_plot_abstracts,
        plot_landmark_timeline,
    )

----------------------------------------------------------------------
DATA  (ml_research_trends.data)
----------------------------------------------------------------------
collect_topic_data(
    keywords,                         # list[str] of search queries
    max_results_per_keyword=100,      # int, cap per keyword
    min_year=None,                    # int or None, inclusive lower bound
    max_year=None,                    # int or None, inclusive upper bound
    save_path=None,                   # optional CSV output path
    api_key=None,                     # str, overrides env var if given
    request_delay=1.1,                # float seconds between requests
    max_retries=5,                    # int, retry count on HTTP 429
    fields=None,                      # str, Semantic Scholar fields override
)  ->  pandas.DataFrame
    Columns: paper_id, title, authors, year, citation_count,
             abstract, url, venue, keyword.

summarize_topic_data(df)  ->  dict
    Returns total_papers, year_min, year_max, average_citations,
    median_citations, missing_abstracts, top_venues.

analyze_topic_trends(df)  ->  pandas.DataFrame
    Per-year paper_count, average_citations, median_citations.

----------------------------------------------------------------------
VISUALIZATION  (ml_research_trends.visualization)
----------------------------------------------------------------------
plot_topic_counts_by_year(
    trend_df,                         # from analyze_topic_trends
    topic_name="Topic",               # str, used in title
    figsize=(10, 6),                  # (w, h) inches
    show=True,                        # bool, call plt.show()
    save_path=None,                   # optional image path
)  ->  matplotlib.figure.Figure

embed_and_plot_abstracts(
    df,                               # from collect_topic_data
    model_name_or_path="google/embeddinggemma-300m",  # HF id OR local path
    device=None,                      # "cuda", "cuda:0", "cpu", "mps",
                                      #   a list like ["cuda:0","cuda:1"] for
                                      #   multi-GPU sharded encoding,
                                      #   or None for auto-detect
    reducer="umap",                   # "umap" | "tsne" | "pca"
    batch_size=8,                     # int
    max_papers=None,                  # optional int cap
    cache_path=None,                  # optional .npy cache path
    topic_name="Topic",               # str, used in title
    random_state=42,                  # int
    normalize_embeddings=True,        # bool
    trust_remote_code=False,          # bool, passed to SentenceTransformer
    umap_kwargs=None,                 # dict, overrides UMAP defaults
    tsne_kwargs=None,                 # dict, overrides TSNE defaults
    pca_kwargs=None,                  # dict, overrides PCA defaults
    figsize=(11, 7),                  # (w, h) inches
    show=True,                        # bool, call plt.show()
    save_path=None,                   # optional image path
)  ->  pandas.DataFrame
    Columns: paper_id, title, year, x, y.

plot_landmark_timeline(
    df,                               # from collect_topic_data
    landmark_date,                    # str or Timestamp, e.g. "2020-05-22"
    landmark_label="Landmark paper",  # str
    cumulative=False,                 # bool
    topic_name="Topic",               # str
    figsize=(11, 6),                  # (w, h) inches
    show=True,                        # bool, call plt.show()
    save_path=None,                   # optional image path
)  ->  pandas.DataFrame
    Columns: year, value.
----------------------------------------------------------------------
'''

from .data import (
    collect_topic_data,
    summarize_topic_data,
    analyze_topic_trends,
)
from .visualization import (
    plot_topic_counts_by_year,
    embed_and_plot_abstracts,
    plot_landmark_timeline,
)

__all__ = [
    "collect_topic_data",
    "summarize_topic_data",
    "analyze_topic_trends",
    "plot_topic_counts_by_year",
    "embed_and_plot_abstracts",
    "plot_landmark_timeline",
]
