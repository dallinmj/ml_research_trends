"""Plots and embedding utilities for the research papers data."""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _resolve_device(device=None):
    """Pick a torch device. Uses CUDA if available, then MPS, then CPU."""
    if device is not None:
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def plot_topic_counts_by_year(
    trend_df,
    topic_name="Topic",
    figsize=(10, 6),
    show=True,
    save_path=None,
):
    """Bar chart of papers per year."""
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=figsize)
    sns.barplot(data=trend_df, x="year", y="paper_count")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title(f"{topic_name} Papers by Year")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def embed_and_plot_abstracts(
    df,
    model_name_or_path="google/embeddinggemma-300m",
    device=None,
    reducer="umap",
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
):
    """Embed paper abstracts and plot them in 2-D colored by year."""
    # Only keep papers that actually have an abstract.
    work = df.copy()
    work = work[work["abstract"].astype(str).str.len() > 0].reset_index(drop=True)
    if max_papers is not None:
        work = work.head(max_papers).reset_index(drop=True)
    if work.empty:
        raise ValueError("No abstracts available to embed.")

    # Try to load cached embeddings first to avoid re-running the model.
    embeddings = None
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(work):
            embeddings = cached
            print(f"Loaded cached embeddings from {cache_path}")

    if embeddings is None:
        from sentence_transformers import SentenceTransformer

        # If `device` is a list of GPUs we'll shard encoding across them.
        multi_device = isinstance(device, (list, tuple))
        if multi_device:
            target_devices = list(device)
            load_device = target_devices[0]
            print(f"Loading {model_name_or_path} on {load_device} (multi-GPU: {target_devices})...")
        else:
            load_device = _resolve_device(device)
            print(f"Loading {model_name_or_path} on {load_device}...")

        model = SentenceTransformer(
            model_name_or_path,
            device=load_device,
            trust_remote_code=trust_remote_code,
        )
        abstracts = work["abstract"].tolist()

        if multi_device:
            pool = model.start_multi_process_pool(target_devices=target_devices)
            try:
                embeddings = model.encode_multi_process(
                    abstracts,
                    pool=pool,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                )
            finally:
                model.stop_multi_process_pool(pool)
        else:
            embeddings = model.encode(
                abstracts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
            )

        if cache_path:
            np.save(cache_path, embeddings)
            print(f"Saved embeddings to {cache_path}")

    # Reduce to 2-D so we can plot it.
    reducer = reducer.lower()
    if reducer == "umap":
        import umap
        defaults = dict(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=random_state)
        defaults.update(umap_kwargs or {})
        coords = umap.UMAP(**defaults).fit_transform(embeddings)
    elif reducer == "tsne":
        from sklearn.manifold import TSNE
        defaults = dict(
            n_components=2,
            metric="cosine",
            init="pca",
            perplexity=min(30, max(5, len(work) // 4)),
            random_state=random_state,
        )
        defaults.update(tsne_kwargs or {})
        coords = TSNE(**defaults).fit_transform(embeddings)
    elif reducer == "pca":
        from sklearn.decomposition import PCA
        defaults = dict(n_components=2, random_state=random_state)
        defaults.update(pca_kwargs or {})
        coords = PCA(**defaults).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    work["x"] = coords[:, 0]
    work["y"] = coords[:, 1]

    # Static matplotlib scatter.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    # Optional interactive Plotly version with hover tooltips.
    if html_path:
        import plotly.express as px
        import textwrap

        hover_df = work.copy()
        for col in ("title", "authors", "venue", "url"):
            if col in hover_df.columns:
                hover_df[col] = hover_df[col].fillna("").astype(str)
        if "citation_count" not in hover_df.columns:
            hover_df["citation_count"] = 0

        # Wrap long titles so the hover box doesn't get super wide.
        hover_df["title_wrapped"] = hover_df["title"].map(
            lambda t: "<br>".join(textwrap.wrap(t, width=80)) or t
        )

        fig_plotly = px.scatter(
            hover_df,
            x="x",
            y="y",
            color="year",
            color_continuous_scale="Viridis",
            hover_data={
                "title_wrapped": True,
                "authors": True,
                "year": True,
                "venue": True,
                "citation_count": True,
                "url": True,
                "x": False,
                "y": False,
            },
            labels={
                "x": f"{reducer.upper()}-1",
                "y": f"{reducer.upper()}-2",
                "title_wrapped": "Title",
                "citation_count": "Citations",
            },
            title=f"{topic_name} abstract embeddings ({reducer.upper()}) — hover for paper details",
        )
        fig_plotly.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0)))
        fig_plotly.update_layout(
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        fig_plotly.write_html(html_path, include_plotlyjs="cdn")

    return work[["paper_id", "title", "year", "x", "y"]]


def plot_landmark_timeline(
    df,
    landmark_date,
    landmark_label="Landmark paper",
    cumulative=False,
    topic_name="Topic",
    figsize=(11, 6),
    show=True,
    save_path=None,
):
    """Line plot of papers per year with a vertical marker on a landmark date."""
    work = df.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["year"])
    if work.empty:
        raise ValueError("No dated papers available.")

    landmark = pd.to_datetime(landmark_date)

    # Count papers per year, filling in missing years with 0.
    counts = work.groupby("year").size().sort_index()
    full_years = pd.Index(range(int(counts.index.min()), int(counts.index.max()) + 1), name="year")
    counts = counts.reindex(full_years, fill_value=0)

    if cumulative:
        series = counts.cumsum()
        y_label = "Cumulative papers"
    else:
        series = counts
        y_label = "Papers per year"

    before = int(counts[counts.index < landmark.year].sum())
    after = int(counts[counts.index >= landmark.year].sum())

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        series.index,
        series.values,
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=7,
        markerfacecolor="steelblue",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )

    # Vertical dashed line + label for the landmark paper.
    ax.axvline(landmark.year, color="crimson", linestyle="--", linewidth=2)
    ax.annotate(
        f"{landmark_label}\n{landmark.date()}",
        xy=(landmark.year, ax.get_ylim()[1] * 0.95),
        xytext=(8, -10),
        textcoords="offset points",
        color="crimson",
        fontsize=10,
        fontweight="bold",
        va="top",
    )

    ax.set_xticks(list(series.index))
    ax.set_title(
        f"{topic_name}: research volume around {landmark_label}\n"
        f"{before} papers before  |  {after} papers after"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return pd.DataFrame({"year": series.index, "value": series.values})
