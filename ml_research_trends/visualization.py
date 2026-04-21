"""Visualization and embedding utilities.

Plots topic trends and projects paper abstracts into 2-D with a
sentence-transformer-compatible embedding model.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _resolve_device(device=None):
    """Resolve a torch device string. If ``device`` is None, auto-pick:
    first available CUDA GPU, else MPS, else CPU.
    """
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
    """Bar chart of paper counts per year from ``analyze_topic_trends``."""
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
    """Embed paper abstracts with a sentence-transformer-compatible model
    and plot a 2-D projection colored by publication year.

    Parameters
    ----------
    df : DataFrame returned by ``collect_topic_data``.
    model_name_or_path : HuggingFace model id OR local path to a model.
    device : torch device string (e.g. "cuda", "cuda:0", "cuda:7", "cpu",
        "mps"), OR a list of device strings (e.g.
        ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]) to shard encoding across
        multiple GPUs via sentence-transformers' multi-process pool. If
        None, auto-detects the best single device.
    reducer : "umap", "tsne", or "pca".
    batch_size : batch size used when encoding abstracts.
    max_papers : optional cap on number of abstracts (for quick iteration).
    cache_path : optional .npy path to cache embeddings keyed to the df order.
    normalize_embeddings : L2-normalize embeddings during encoding.
    trust_remote_code : pass-through to SentenceTransformer for models that
        require it (e.g. some custom architectures).
    umap_kwargs, tsne_kwargs, pca_kwargs : dicts to override the respective
        reducer's default arguments.
    figsize, show, save_path : static-plot controls (matplotlib PNG).
    html_path : optional path to also write an interactive Plotly HTML
        scatter with hover tooltips (title / authors / year / venue /
        citations / url). Requires ``plotly`` to be installed.
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

        # ``device`` may be a single device string (e.g. "cuda:7") or a list
        # of device strings (e.g. ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]).
        # When a list is passed we spin up a multi-process pool so the
        # abstracts are sharded across multiple GPUs.
        multi_device = isinstance(device, (list, tuple))

        if multi_device:
            target_devices = list(device)
            # Load the model on the first device; each worker process will
            # move its own copy onto the appropriate target device.
            load_device = target_devices[0]
            print(
                f"Loading {model_name_or_path} on {load_device} "
                f"(multi-GPU pool: {target_devices})..."
            )
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

    reducer = reducer.lower()
    if reducer == "umap":
        import umap

        defaults = dict(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=random_state,
        )
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

    if html_path:
        import plotly.express as px

        hover_df = work.copy()
        for col in ("title", "authors", "venue", "url"):
            if col in hover_df.columns:
                hover_df[col] = hover_df[col].fillna("").astype(str)
        if "citation_count" not in hover_df.columns:
            hover_df["citation_count"] = 0

        # Wrap long titles so the hover tooltip doesn't blow out horizontally.
        def _wrap(text, width=80):
            import textwrap
            return "<br>".join(textwrap.wrap(text, width=width)) or text

        hover_df["title_wrapped"] = hover_df["title"].map(_wrap)

        hover_data = {
            "title_wrapped": True,
            "authors": True,
            "year": True,
            "venue": True,
            "citation_count": True,
            "url": True,
            "x": False,
            "y": False,
        }

        fig_plotly = px.scatter(
            hover_df,
            x="x",
            y="y",
            color="year",
            color_continuous_scale="Viridis",
            hover_data=hover_data,
            labels={
                "x": f"{reducer.upper()}-1",
                "y": f"{reducer.upper()}-2",
                "title_wrapped": "Title",
                "citation_count": "Citations",
            },
            title=(
                f"{topic_name} abstract embeddings ({reducer.upper()}) "
                f"colored by year — hover for paper details"
            ),
        )
        fig_plotly.update_traces(
            marker=dict(size=7, opacity=0.8, line=dict(width=0)),
        )
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
    """Plot per-year paper counts as points connected by a line, with a
    vertical marker at a landmark paper's publication date.

    Parameters
    ----------
    df : DataFrame returned by ``collect_topic_data``. Uses the ``year`` column.
    landmark_date : str or Timestamp, e.g. "2020-05-28" for GPT-3.
    cumulative : if True, plot cumulative paper count instead of per-year.
    """
    work = df.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["year"])

    if work.empty:
        raise ValueError("No dated papers available.")

    landmark = pd.to_datetime(landmark_date)

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
