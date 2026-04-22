"""Main page for the ML Research Trends Streamlit app.

Run with: streamlit run app.py
"""

import streamlit as st

from app_utils import (
    available_topics,
    load_papers,
    sidebar_filters,
    sidebar_topic_picker,
)

st.set_page_config(page_title="Machine Learning Research Trends", layout="wide")

st.title("Machine Learning Research Trends")
st.markdown(
    """
    An interactive dashboard for exploring how a few Machine Learning (ML) research
    sub-fields have grown over time. For each topic we pulled papers
    from Semantic Scholar, embedded their abstracts with
    **Qwen3-Embedding-4B**, and projected them to 2-D with UMAP.

    Use the sidebar to pick a topic and filter the data. Additional
    views (trends, embeddings) are available from the **Pages** menu
    in the sidebar.
    """
)

slug, meta = sidebar_topic_picker()
df = load_papers(slug)
filtered = sidebar_filters(df)

# Save the filtered data so other pages can use it too.
st.session_state["filtered_df"] = filtered

st.subheader(f"Overview — {meta['name']}")

# Four quick summary stats at the top.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Papers (filtered)", f"{len(filtered):,}")
col2.metric("Papers (total)", f"{len(df):,}")
if len(filtered) > 0:
    col3.metric("Year range", f"{int(filtered['year'].min())}–{int(filtered['year'].max())}")
    col4.metric("Median citations", f"{int(filtered['citation_count'].median())}")
else:
    col3.metric("Year range", "—")
    col4.metric("Median citations", "—")

st.markdown(f"**Landmark paper:** {meta['landmark_label']} ({meta['landmark_date']})")

with st.expander("Top venues (filtered)", expanded=False):
    if len(filtered) > 0:
        top_venues = (
            filtered["venue"]
            .replace("", "Unknown")
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .rename_axis("venue")
            .reset_index(name="papers")
        )
        st.dataframe(top_venues, use_container_width=True)
    else:
        st.info("No papers match the current filters.")

with st.expander("Most cited papers (filtered)", expanded=True):
    if len(filtered) > 0:
        top_cited = filtered.sort_values("citation_count", ascending=False).head(10)
        top_cited = top_cited[["title", "authors", "year", "citation_count", "venue", "url"]]
        st.dataframe(
            top_cited,
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("url"),
                "citation_count": st.column_config.NumberColumn(format="%d"),
            },
            hide_index=True,
        )
    else:
        st.info("No papers match the current filters.")

st.markdown("---")
topic_names = ", ".join(m["name"] for m in available_topics().values())
st.caption(
    f"Available topics: {topic_names}. "
    "Data source: Semantic Scholar Graph API (cached CSVs in `local_data/`)."
)
