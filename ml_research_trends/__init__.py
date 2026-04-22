"""Small toolkit for collecting and visualizing ML research papers."""

from .data import (
    collect_topic_data,
    summarize_topic_data,
    analyze_topic_trends,
)
from .visualization import (
    plot_topic_counts_by_year,
    embed_and_plot_abstracts,
    plot_landmark_timeline,
)

__all__ = [
    "collect_topic_data",
    "summarize_topic_data",
    "analyze_topic_trends",
    "plot_topic_counts_by_year",
    "embed_and_plot_abstracts",
    "plot_landmark_timeline",
]
