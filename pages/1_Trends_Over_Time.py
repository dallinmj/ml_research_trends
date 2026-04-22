"""Yearly paper counts and a landmark timeline for the selected topic."""

import matplotlib.pyplot as plt
import streamlit as st

from app_utils import load_papers, sidebar_filters, sidebar_topic_picker
from ml_research_trends.data import analyze_topic_trends
from ml_research_trends.visualization import plot_landmark_timeline, plot_topic_counts_by_year

st.set_page_config(page_title="Trends Over Time", layout="wide")
st.title("Trends over time")

slug, meta = sidebar_topic_picker()
df = load_papers(slug)
filtered = sidebar_filters(df)

if filtered.empty:
    st.warning("No papers match the current filters.")
    st.stop()

# Papers per year plot.
st.subheader(f"Papers per year — {meta['name']}")
trends = analyze_topic_trends(filtered)
fig = plot_topic_counts_by_year(trends, topic_name=meta["name"], show=False)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

with st.expander("Per-year statistics", expanded=False):
    st.dataframe(trends, use_container_width=True, hide_index=True)

# Landmark timeline plot.
st.subheader("Landmark timeline")
st.caption(
    f"Vertical marker at **{meta['landmark_label']}** ({meta['landmark_date']})."
)

cumulative = st.checkbox("Show cumulative paper counts", value=False)

plot_landmark_timeline(
    filtered,
    landmark_date=meta["landmark_date"],
    landmark_label=meta["landmark_label"],
    cumulative=cumulative,
    topic_name=meta["name"],
    show=False,
)
# plot_landmark_timeline doesn't return the figure, so grab the current one.
st.pyplot(plt.gcf(), use_container_width=True)
plt.close("all")
