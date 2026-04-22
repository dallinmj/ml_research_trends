---
title: Documentation
---

# Documentation

[← Back to home](index.md)

Reference for every function exposed by the `ml_research_trends` package.

---

## `collect_topic_data`

```python
collect_topic_data(
    keywords,
    max_results_per_keyword=100,
    min_year=None,
    max_year=None,
    save_path=None,
    api_key=None,
    request_delay=1.1,
    max_retries=5,
    fields=None,
) -> pandas.DataFrame
```

Query the Semantic Scholar Graph API for each keyword, de-duplicate the
results, filter them by year, and return one combined DataFrame.

**Arguments**

- `keywords` (list[str]): search strings to send to the API.
- `max_results_per_keyword` (int): cap on papers collected per keyword.
- `min_year`, `max_year` (int or None): inclusive year filter.
- `save_path` (str or None): if given, the DataFrame is written to this CSV.
- `api_key` (str or None): Semantic Scholar API key. If not given, it's read
  from the `API_KEY` environment variable (a `.env` file is supported).
- `request_delay` (float): seconds to wait between requests (rate limit).
- `max_retries` (int): number of retries on HTTP 429.
- `fields` (str or None): comma-separated list of API fields to request.

**Returns**

A `pandas.DataFrame` with columns: `paper_id`, `title`, `authors`, `year`,
`citation_count`, `abstract`, `url`, `venue`, `keyword`.

**Example**

```python
df = collect_topic_data(["chain of thought"], max_results_per_keyword=100)
```

---

## `summarize_topic_data`

```python
summarize_topic_data(df) -> dict
```

Return a dictionary of quick stats about a topic DataFrame:
`total_papers`, `year_min`, `year_max`, `average_citations`,
`median_citations`, `missing_abstracts`, and `top_venues`.

**Example**

```python
summary = summarize_topic_data(df)
print(summary["total_papers"], summary["year_min"], summary["year_max"])
```

---

## `analyze_topic_trends`

```python
analyze_topic_trends(df) -> pandas.DataFrame
```

Group papers by year and compute `paper_count`, `average_citations`, and
`median_citations`. Useful input for `plot_topic_counts_by_year`.

**Example**

```python
trends = analyze_topic_trends(df)
```

---

## `plot_topic_counts_by_year`

```python
plot_topic_counts_by_year(
    trend_df,
    topic_name="Topic",
    figsize=(10, 6),
    show=True,
    save_path=None,
) -> matplotlib.figure.Figure
```

Bar chart of papers per year. Expects the DataFrame returned by
`analyze_topic_trends`.

**Example**

```python
plot_topic_counts_by_year(trends, topic_name="RAG", save_path="rag.png")
```

---

## `embed_and_plot_abstracts`

```python
embed_and_plot_abstracts(
    df,
    model_name_or_path="google/embeddinggemma-300m",
    device=None,
    reducer="umap",          # or "tsne" / "pca"
    batch_size=8,
    max_papers=None,
    cache_path=None,
    topic_name="Topic",
    random_state=42,
    normalize_embeddings=True,
    trust_remote_code=False,
    umap_kwargs=None,
    tsne_kwargs=None,
    pca_kwargs=None,
    figsize=(11, 7),
    show=True,
    save_path=None,
    html_path=None,
) -> pandas.DataFrame
```

Embed each paper's abstract with a Sentence-Transformers model, reduce to 2-D
(UMAP / t-SNE / PCA), and produce a scatter plot colored by year. Embeddings
are cached to `cache_path` so subsequent runs don't re-encode.

**Returns**

A DataFrame with columns `paper_id`, `title`, `year`, `x`, `y`.

**Example**

```python
coords = embed_and_plot_abstracts(
    df,
    model_name_or_path="Qwen/Qwen3-Embedding-4B",
    cache_path="rag_emb.npy",
    save_path="rag_umap.png",
    html_path="rag_umap.html",
)
```

---

## `plot_landmark_timeline`

```python
plot_landmark_timeline(
    df,
    landmark_date,
    landmark_label="Landmark paper",
    cumulative=False,
    topic_name="Topic",
    figsize=(11, 6),
    show=True,
    save_path=None,
) -> pandas.DataFrame
```

Line plot of papers per year with a vertical dashed line marking a landmark
paper's date. Also prints how many papers came before vs. after the landmark.

**Example**

```python
plot_landmark_timeline(
    df,
    landmark_date="2020-05-22",
    landmark_label="Lewis et al. — RAG",
    topic_name="RAG",
)
```
