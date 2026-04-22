"""Helpers shared across the Streamlit pages."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "local_data"

# Make sure we can import the `ml_research_trends` package when running the app.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# All the topics we have data for. Each topic lives in its own subfolder
# under local_data/ (e.g. local_data/transformer/transformer_papers.csv).
TOPIC_REGISTRY = {
    "transformer": {
        "name": "Transformer Architecture",
        "papers_csv": "transformer_papers.csv",
        "embeddings_npy": "transformer_qwen3_embeddings.npy",
        "umap_png": "transformer_embeddings_umap.png",
        "landmark_date": "2017-06-12",
        "landmark_label": "Vaswani et al. — Attention Is All You Need",
    },
    "rag": {
        "name": "Retrieval-Augmented Generation",
        "papers_csv": "rag_papers.csv",
        "embeddings_npy": "rag_qwen3_embeddings.npy",
        "umap_png": "rag_embeddings_umap.png",
        "landmark_date": "2020-05-22",
        "landmark_label": "Lewis et al. — RAG",
    },
    "reasoning": {
        "name": "Reasoning / Chain-of-Thought",
        "papers_csv": "reasoning_papers.csv",
        "embeddings_npy": "reasoning_qwen3_embeddings.npy",
        "umap_png": "reasoning_embeddings_umap.png",
        "landmark_date": "2022-01-28",
        "landmark_label": "Wei et al. — Chain-of-Thought",
    },
    "multimodal_vlm": {
        "name": "Multimodal / Vision-Language",
        "papers_csv": "multimodal_vlm_papers.csv",
        "embeddings_npy": "multimodal_vlm_qwen3_embeddings.npy",
        "umap_png": "multimodal_vlm_embeddings_umap.png",
        "landmark_date": "2021-02-26",
        "landmark_label": "Radford et al. — CLIP",
    },
}


def topic_dir(slug):
    """Folder where a topic's CSVs/embeddings/plots live."""
    return DATA_DIR / slug


def data_path(slug, filename):
    """Full path to a file inside a topic's folder."""
    if not filename:
        return None
    return topic_dir(slug) / filename


def available_topics():
    # Only show topics whose CSV is actually on disk.
    topics = {}
    for slug, meta in TOPIC_REGISTRY.items():
        if data_path(slug, meta["papers_csv"]).exists():
            topics[slug] = meta
    return topics


@st.cache_data(show_spinner=False)
def load_papers(slug):
    meta = TOPIC_REGISTRY[slug]
    df = pd.read_csv(data_path(slug, meta["papers_csv"]))

    # Clean up the year and citation columns.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["citation_count"] = pd.to_numeric(df.get("citation_count"), errors="coerce").fillna(0).astype(int)

    # Fill in empty strings for missing text columns.
    for col in ["title", "abstract", "authors", "venue", "url", "keyword"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_embeddings(slug):
    path = data_path(slug, TOPIC_REGISTRY[slug]["embeddings_npy"])
    if path is None or not path.exists():
        return None
    return np.load(path)


@st.cache_data(show_spinner="Projecting embeddings with UMAP...")
def compute_umap(slug, n_neighbors=15, min_dist=0.1, random_state=42):
    """Run UMAP on the saved embeddings and return a DataFrame with x/y."""
    import umap

    embeddings = load_embeddings(slug)
    if embeddings is None:
        return None

    df = load_papers(slug)
    df = df[df["abstract"].str.len() > 0].reset_index(drop=True)

    # Just in case the CSV and the .npy don't quite line up.
    n = min(len(df), len(embeddings))
    df = df.head(n).reset_index(drop=True)
    embeddings = embeddings[:n]

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    return df


def sidebar_topic_picker():
    """Dropdown in the sidebar to pick which topic to look at."""
    topics = available_topics()
    if not topics:
        st.sidebar.error("No topic CSVs found in local_data/.")
        st.stop()

    slugs = list(topics.keys())
    slug = st.sidebar.selectbox(
        "Topic",
        slugs,
        format_func=lambda s: topics[s]["name"],
        key="topic_slug",
    )
    return slug, topics[slug]


def sidebar_filters(df):
    """Year, keyword, and citation filters in the sidebar."""
    st.sidebar.markdown("### Filters")

    y_min, y_max = int(df["year"].min()), int(df["year"].max())
    if y_min == y_max:
        year_range = (y_min, y_max)
        st.sidebar.caption(f"Year: {y_min}")
    else:
        year_range = st.sidebar.slider("Year range", y_min, y_max, (y_min, y_max))

    keyword_options = sorted(k for k in df.get("keyword", pd.Series()).unique() if k)
    selected_keywords = st.sidebar.multiselect(
        "Search keywords",
        options=keyword_options,
        default=keyword_options,
    )

    min_citations = st.sidebar.number_input("Min. citations", min_value=0, value=0, step=1)

    out = df[
        (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
        & (df["citation_count"] >= min_citations)
    ]
    if selected_keywords and "keyword" in df.columns:
        out = out[out["keyword"].isin(selected_keywords)]
    return out.reset_index(drop=True)
