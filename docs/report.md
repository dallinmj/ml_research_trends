---
title: Technical Report
---

# Technical Report

[← Back to home](index.md)

## Motivating question

**How has the volume of ML research on a handful of high-impact sub-topics
grown over time, and how much of that growth can be tied to specific
landmark papers?**

The intuition is that a single influential paper (e.g. *Attention Is All You
Need*) can kick off a wave of follow-up work. We wanted to see whether that
pattern actually shows up in the data for several well-known sub-fields, and
also whether the papers in each sub-field cluster together semantically in a
way that reflects the topic's sub-structure. We also wanted to see if there was
a delay in how soon the research followed these landmark studies. 

## Data

We collected paper metadata from the
[Semantic Scholar Graph API](https://api.semanticscholar.org/) using keyword
search. Four topics were included:

| Topic | Keywords | Landmark paper |
| --- | --- | --- |
| Transformer Architecture | "self-attention neural network", "multi-head attention", ... | Vaswani et al. (2017) |
| Retrieval-Augmented Generation | "retrieval augmented generation", "RAG", ... | Lewis et al. (2020) |
| Reasoning / Chain-of-Thought | "chain of thought", "step by step reasoning LLM", ... | Wei et al. (2022) |
| Multimodal / Vision-Language | "vision language model", "CLIP", ... | Radford et al. (2021) |

For each topic we pulled up to ~500 papers per keyword, de-duplicated by paper
ID, and filtered to the year window. The cleaned CSVs live in
`local_data/<topic>/`.

## Methodology

1. **Collection.** `collect_topic_data` calls the Semantic Scholar search
   endpoint for each keyword, handling pagination, rate limits, and de-duplication.
2. **Cleaning.** Missing abstracts and titles are stripped, `year` and
   `citation_count` are changed to integers, and rows without a year are
   dropped.
3. **Trend analysis.** `analyze_topic_trends` groups by year and computes
   `paper_count`, `average_citations`, and `median_citations`.
4. **Landmark timeline.** `plot_landmark_timeline` draws the yearly counts
   with a vertical line on the landmark paper's date, and prints the number
   of papers before vs. after it.
5. **Semantic embeddings.** `embed_and_plot_abstracts` encodes each abstract
   with [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
   and reduces the resulting 4096-dim vectors to 2-D with UMAP (cosine
   metric, `n_neighbors=15`, `min_dist=0.1`). Embeddings are cached to
   `.npy` so the whole pipeline is re-runnable in a few seconds.
6. **Dashboard.** The Streamlit app (`app.py` + `pages/`) loads the cached
   CSVs and embeddings, re-runs UMAP on demand with user-chosen
   hyperparameters, and renders an interactive Plotly scatter plot.

## Key takeaways

- **The landmark-paper effect is real and visible.** For every topic, the
  per-year paper count is basically flat before the landmark paper and then
  grows sharply afterward. Transformer papers, for example, are dominated by
  post-2017 work; RAG and CoT both show near-vertical growth immediately
  after 2020 / 2022 respectively.
- **Growth is still accelerating.** Some of the topics have their highest paper
  counts in the most recent full year in the dataset. Even "older" areas
  like the original Transformer line are still growing, not plateauing.
- **Abstract embeddings recover meaningful sub-clusters.** In the UMAP
  projection, papers group into visibly separate sub-communities (e.g.
  retrieval systems vs. reasoning vs. evaluation benchmarks within
  the RAG topic). Color-by-year also shows a clear temporal drift across the
  embedding space, suggesting the language used to describe each sub-field
  has shifted noticeably over the last few years.
- **Citations are extremely skewed.** In every topic the median citation
  count is small but the mean is much larger, driven by a handful of very
  high-citation landmark papers. This is why the Streamlit app reports the
  median rather than the mean.

## Reproducibility

Anyone can reproduce every artifact in the repo by:

```bash
git clone https://github.com/dallinmj/ml_research_trends.git
cd ml_research_trends
pip install -e .
cd local_data
python run.py           # edit the TOPIC / SLUG / KEYWORDS block to pick a topic
```

The raw CSVs, cached embeddings, and plots are all committed under
`local_data/<topic>/` so you don't *have* to re-run the pipeline to explore
the results — the Streamlit app and notebooks will work out of the box.

## Limitations

- Semantic Scholar's search endpoint is keyword-based, so topic coverage is
  only as good as the keyword list. A paper that uses different terminology
  will be missed.
- The 2026 counts are partial (the year isn't over yet), so the last bar on
  every plot is an undercount.
- Citation counts are a snapshot at collection time and will drift upward.

## Potential next steps

- Add more topics (diffusion models, RLHF, interpretability, etc.).
- Cluster the UMAP space and auto-label the clusters with top keywords.
- Compare growth curves across topics on one shared plot.
