"""Local experiment runner.

Run with:
    uv run python run.py
or:
    python run.py

Edit the TOPIC / KEYWORDS / SLUG block below to play with different topics
and hyperparameters. This file is NOT part of the installable package —
it's just a convenient driver for your own experiments.
"""

from ml_research_trends import (
    collect_topic_data,
    summarize_topic_data,
    analyze_topic_trends,
    plot_topic_counts_by_year,
    embed_and_plot_abstracts,
    plot_landmark_timeline,
)

# Past runs (uncomment a block to rerun):

# --- RAG ------------------------------------------------------------
# TOPIC = "RAG"
# SLUG = "rag"
# KEYWORDS = [
#     "retrieval augmented generation",
#     "retrieval-augmented generation",
#     "RAG",
# ]
# LANDMARK_DATE = "2020-05-22"
# LANDMARK_LABEL = "Lewis et al. — RAG"
# --------------------------------------------------------------------

# --- Reasoning / Chain-of-Thought -------------------------------------
# TOPIC = "Reasoning"
# SLUG = "reasoning"
# KEYWORDS = [
#     "chain of thought",
#     "chain-of-thought prompting",
#     "large language model reasoning",
#     "step by step reasoning language model",
#     "reasoning model",
# ]
# LANDMARK_DATE = "2022-01-28"  # Wei et al., CoT (arXiv v1)
# LANDMARK_LABEL = "Wei et al. — Chain-of-Thought"
# ----------------------------------------------------------------------

# --- Multimodal / Vision-Language Models ------------------------------
# TOPIC = "Multimodal / Vision-Language"
# SLUG = "multimodal_vlm"
# KEYWORDS = [
#     "vision language model",
#     "vision-language pretraining",
#     "multimodal large language model",
#     "image text contrastive learning",
#     "CLIP contrastive language image",
#     "visual instruction tuning",
# ]
# LANDMARK_DATE = "2021-02-26"  # Radford et al., CLIP (arXiv v1)
# LANDMARK_LABEL = "Radford et al. — CLIP"
# ----------------------------------------------------------------------

# --- Transformer Architecture -----------------------------------------
# "Attention Is All You Need" (Vaswani et al., 2017) predates our 2019+
# collection window, but it's the foundational architecture anchor that
# virtually all later work in this repo builds on.
TOPIC = "Transformer Architecture"
SLUG = "transformer"
KEYWORDS = [
    "self-attention neural network",
    "multi-head attention",
    "transformer language model",
    "encoder decoder transformer",
]
LANDMARK_DATE = "2017-06-12"  # Vaswani et al., Attention Is All You Need (arXiv v1)
LANDMARK_LABEL = "Vaswani et al. — Attention Is All You Need"
# ----------------------------------------------------------------------


def main():
    df = collect_topic_data(
        keywords=KEYWORDS,
        max_results_per_keyword=500,
        min_year=2016,
        max_year=2026,
        save_path=f"{SLUG}_papers.csv",
    )

    summary = summarize_topic_data(df)
    trends = analyze_topic_trends(df)

    print("SUMMARY")
    print("Total papers:", summary["total_papers"])
    print("Year range:", summary["year_min"], "-", summary["year_max"])
    print("Average citations:", summary["average_citations"])
    print("Median citations:", summary["median_citations"])
    print("Missing abstracts:", summary["missing_abstracts"])
    print("\nTop venues:")
    print(summary["top_venues"])

    print("\nTRENDS")
    print(trends)

    # NOTE: we're in a headless container (no display server), so we save
    # plots to PNG files instead of relying on plt.show() windows. Open the
    # .png files from Cursor's file explorer to view them.
    plot_topic_counts_by_year(
        trends,
        topic_name=TOPIC,
        show=False,
        save_path=f"{SLUG}_counts_by_year.png",
    )
    print(f"Saved {SLUG}_counts_by_year.png")

    # Default embedding model is google/embeddinggemma-300m (small & fast).
    # For this run we override to Qwen3-Embedding-4B as a hyperparameter.
    # First run will download the model (~8GB) and cache embeddings.
    # Set to a single device string (e.g. "cuda:7") for single-GPU, or a
    # list of device strings to shard encoding across multiple GPUs.
    GPUS = ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]

    embed_and_plot_abstracts(
        df,
        model_name_or_path="Qwen/Qwen3-Embedding-4B",
        device=GPUS,
        reducer="umap",
        batch_size=8,
        cache_path=f"{SLUG}_qwen3_embeddings.npy",
        topic_name=TOPIC,
        show=False,
        save_path=f"{SLUG}_embeddings_umap.png",
        html_path=f"{SLUG}_embeddings_umap.html",
    )
    print(f"Saved {SLUG}_embeddings_umap.png")
    print(f"Saved {SLUG}_embeddings_umap.html (interactive — open in a browser)")

    plot_landmark_timeline(
        df,
        landmark_date=LANDMARK_DATE,
        landmark_label=LANDMARK_LABEL,
        topic_name=TOPIC,
        show=False,
        save_path=f"{SLUG}_landmark_timeline.png",
    )
    print(f"Saved {SLUG}_landmark_timeline.png")


if __name__ == "__main__":
    main()
