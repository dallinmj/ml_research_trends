---
title: ML Research Trends
---

# ML Research Trends

A class project that looks at how a few different areas of machine learning
research have grown over time. I pulled papers from the Semantic Scholar API,
embedded their abstracts with Qwen3-Embedding-4B, and made a small Python
package + Streamlit app to explore the results.

## Links

- [Tutorial](tutorial.md) — how to install the package and use it end-to-end
- [Documentation](documentation.md) — reference for each function in the package
- [Technical Report](report.md) — motivating question, methodology, and findings
- [Streamlit App](https://ml-research-trends.streamlit.app/) — interactive dashboard
- [GitHub Repository](https://github.com/dallinmj/ml_research_trends) — source code and data

## Topics included

- Transformer Architecture
- Retrieval-Augmented Generation (RAG)
- Reasoning / Chain-of-Thought
- Multimodal / Vision-Language Models

For each topic the repo includes a CSV of papers, cached embeddings, and a
few plots under `local_data/<topic>/`.
