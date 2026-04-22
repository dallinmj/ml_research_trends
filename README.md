# ML Research Trends

A small Python package + Streamlit app I built to look at how different
sub-areas of machine learning research have grown over time. For each topic
(like transformers, RAG, chain-of-thought reasoning, and vision-language
models) I pull papers from the Semantic Scholar API, embed their abstracts
with Qwen3-Embedding-4B, and make some plots.

## What's in here

- `ml_research_trends/` — the installable Python package (data collection + plots)
- `app.py` + `pages/` — a multi-page Streamlit app for poking around the data
- `local_data/` — CSVs, cached embeddings, and plots for each topic (one subfolder per topic)
- `docs/` — files for the GitHub Pages site

## Install

Clone the repo, then from the project root:

```bash
pip install -e .
```

That installs the `ml_research_trends` package along with everything the app
and notebooks need.

You'll also want a `.env` file with a Semantic Scholar API key if you plan to
collect new data:

```
API_KEY=your_semantic_scholar_key_here
```

You don't need the key just to run the app — the CSVs in `local_data/` are
already included.

## Quickstart (package)

```python
from ml_research_trends import (
    collect_topic_data,
    analyze_topic_trends,
    plot_topic_counts_by_year,
)

df = collect_topic_data(
    keywords=["retrieval augmented generation", "RAG"],
    max_results_per_keyword=200,
    min_year=2019,
)

trends = analyze_topic_trends(df)
plot_topic_counts_by_year(trends, topic_name="RAG")
```

## Run the Streamlit app

```bash
streamlit run app.py
```

Then use the sidebar to pick a topic and explore the trends / embeddings pages.

## Topics currently included

- Transformer Architecture (landmark: *Attention Is All You Need*, 2017)
- Retrieval-Augmented Generation (landmark: Lewis et al. RAG, 2020)
- Reasoning / Chain-of-Thought (landmark: Wei et al. CoT, 2022)
- Multimodal / Vision-Language (landmark: Radford et al. CLIP, 2021)

## Data source

All paper metadata comes from the [Semantic Scholar Graph API](https://api.semanticscholar.org/).
The embeddings were generated with
[Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B).

## Project links

- **Docs + Tutorial + Report:** see the [GitHub Pages site](https://dallinmj.github.io/ml_research_trends/)
- **GitHub repo:** https://github.com/dallinmj/ml_research_trends
