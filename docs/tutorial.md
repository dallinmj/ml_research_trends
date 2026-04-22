---
title: Tutorial
---

# Tutorial

[← Back to home](index.md)

This page walks through installing the `ml_research_trends` package and using
it to collect, analyze, and visualize papers for a topic.

## 1. Install

Clone the repo and install in editable mode:

```bash
git clone https://github.com/dallinmj/ml_research_trends.git
cd ml_research_trends
pip install -e .
```

If you plan to collect new data (as opposed to just using the CSVs already in
`local_data/`), add a `.env` file at the project root:

```
API_KEY=your_semantic_scholar_key_here
```

You can get a free key from [Semantic Scholar](https://www.semanticscholar.org/product/api).

## 2. Collect papers for a topic

```python
from ml_research_trends import collect_topic_data

df = collect_topic_data(
    keywords=[
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "RAG",
    ],
    max_results_per_keyword=200,
    min_year=2019,
    max_year=2026,
    save_path="rag_papers.csv",
)

print(df.head())
print("Total papers:", len(df))
```

This queries Semantic Scholar for each keyword, de-duplicates the results,
filters by year, and returns a tidy `pandas.DataFrame`. It also writes the
DataFrame to CSV if `save_path` is given.

## 3. Summarize and analyze

```python
from ml_research_trends import summarize_topic_data, analyze_topic_trends

summary = summarize_topic_data(df)
print("Total papers:", summary["total_papers"])
print("Year range:", summary["year_min"], "-", summary["year_max"])
print("Top venues:\n", summary["top_venues"])

trends = analyze_topic_trends(df)
print(trends)
```

## 4. Make plots

```python
from ml_research_trends import (
    plot_topic_counts_by_year,
    plot_landmark_timeline,
    embed_and_plot_abstracts,
)

plot_topic_counts_by_year(trends, topic_name="RAG", save_path="rag_by_year.png")

plot_landmark_timeline(
    df,
    landmark_date="2020-05-22",
    landmark_label="Lewis et al. — RAG",
    topic_name="RAG",
    save_path="rag_timeline.png",
)

embed_and_plot_abstracts(
    df,
    model_name_or_path="Qwen/Qwen3-Embedding-4B",
    reducer="umap",
    cache_path="rag_embeddings.npy",
    topic_name="RAG",
    save_path="rag_umap.png",
    html_path="rag_umap.html",
)
```

Embedding the first time will download the model (~8 GB) and cache the
vectors to `cache_path`. After that, rerunning is fast.

## 5. Run the Streamlit app

From the project root:

```bash
streamlit run app.py
```

Then open the URL. Pick a topic in
the sidebar and explore the Overview, Trends Over Time, and Topic Embeddings
pages.
