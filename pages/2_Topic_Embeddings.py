"""Interactive 2-D UMAP scatter plot of paper abstract embeddings."""

import plotly.express as px
import streamlit as st

from app_utils import (
    compute_umap,
    data_path,
    load_papers,
    sidebar_filters,
    sidebar_topic_picker,
)

st.set_page_config(page_title="Topic Embeddings", layout="wide")
st.title("Topic exploration — abstract embeddings")

slug, meta = sidebar_topic_picker()
df = load_papers(slug)
filtered = sidebar_filters(df)

st.markdown(
    "Each point is one paper, positioned by the semantic similarity of "
    "its abstract (Qwen3-Embedding-4B → UMAP → 2-D). Hover for details."
)

# If there are no embeddings saved, just show the static PNG.
embeddings_path = data_path(slug, meta["embeddings_npy"])
if embeddings_path is None or not embeddings_path.exists():
    st.info("No cached embeddings found for this topic. Showing the static UMAP image instead.")
    umap_png = data_path(slug, meta.get("umap_png"))
    if umap_png and umap_png.exists():
        st.image(str(umap_png), use_container_width=True)
    else:
        st.warning("No embeddings or image available for this topic.")
    st.stop()

# UMAP parameter controls.
col_a, col_b, col_c = st.columns(3)
n_neighbors = col_a.slider("UMAP n_neighbors", 5, 50, 15, 1)
min_dist = col_b.slider("UMAP min_dist", 0.0, 0.99, 0.1, 0.05)
random_state = col_c.number_input("Random seed", value=42, step=1)

coords_df = compute_umap(
    slug,
    n_neighbors=n_neighbors,
    min_dist=float(min_dist),
    random_state=int(random_state),
)
if coords_df is None:
    st.warning("Could not compute UMAP for this topic.")
    st.stop()

# Keep only the points that pass the sidebar filters.
keep_ids = set(filtered["paper_id"].astype(str))
view = coords_df[coords_df["paper_id"].astype(str).isin(keep_ids)].copy()

if view.empty:
    st.warning("No embedded papers match the current filters.")
    st.stop()

# Shorten the title so hover tooltips aren't huge.
view["title_short"] = view["title"].str.slice(0, 140)

fig = px.scatter(
    view,
    x="x",
    y="y",
    color="year",
    color_continuous_scale="Viridis",
    hover_data={
        "title_short": True,
        "authors": True,
        "year": True,
        "venue": True,
        "citation_count": True,
        "url": True,
        "x": False,
        "y": False,
    },
    labels={"title_short": "Title", "citation_count": "Citations"},
    title=f"{meta['name']} — {len(view):,} abstracts (UMAP n_neighbors={n_neighbors}, min_dist={min_dist})",
)
fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0)))
fig.update_layout(
    template="plotly_white",
    height=900,
    yaxis=dict(scaleanchor="x", scaleratio=1),
    hoverlabel=dict(bgcolor="white", font_size=12, font_color="black", bordercolor="black"),
)
st.plotly_chart(fig, use_container_width=True)
