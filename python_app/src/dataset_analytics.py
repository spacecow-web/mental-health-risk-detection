"""Exploratory data analysis (EDA) utilities for labeled social media text corpora."""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TOKEN_WORD_RE = re.compile(r"[a-zA-Z]{3,}")
BASIC_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "are", "was", "have",
    "not", "but", "from", "they", "his", "her", "she", "him", "just", "all",
    "like", "what", "when", "out", "can", "about", "been", "had", "has", "were",
    "would", "there", "their", "into", "will", "your", "more", "very", "because",
}


@dataclass
class DatasetSummary:
    """Real descriptive statistics computed strictly from a supplied CSV corpus."""

    total_rows: int
    valid_rows: int
    missing_text_rows: int
    missing_label_rows: int
    duplicate_texts: int
    num_classes: int
    class_counts: pd.DataFrame
    length_stats: Dict[str, float]
    top_terms_by_class: Dict[str, List[Dict[str, int]]]


def analyze_corpus(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> DatasetSummary:
    """Compute real descriptive statistics and data quality checks on a labeled corpus."""
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset must include '{text_col}' and '{label_col}' columns.")

    total_rows = len(df)
    missing_text = int(df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == "").sum())
    missing_label = int(df[label_col].isna().sum())

    valid_df = df.dropna(subset=[text_col, label_col]).copy()
    valid_df[text_col] = valid_df[text_col].astype(str).str.strip()
    valid_df[label_col] = valid_df[label_col].astype(str).str.strip()
    valid_df = valid_df[valid_df[text_col] != ""]

    duplicate_texts = int(valid_df[text_col].duplicated().sum())
    word_counts = valid_df[text_col].apply(lambda s: len(s.split())).to_numpy(dtype=float)
    char_counts = valid_df[text_col].apply(len).to_numpy(dtype=float)

    class_counts = (
        valid_df[label_col]
        .value_counts()
        .reset_index()
        .rename(columns={label_col: "Label", "count": "Count"})
    )
    if "index" in class_counts.columns and "Label" not in class_counts.columns:
        class_counts = class_counts.rename(columns={"index": "Label"})
    class_counts["Share (%)"] = (class_counts["Count"] / max(len(valid_df), 1) * 100.0).round(2)

    length_stats = {
        "mean_words": float(np.mean(word_counts)) if word_counts.size else 0.0,
        "median_words": float(np.median(word_counts)) if word_counts.size else 0.0,
        "p95_words": float(np.percentile(word_counts, 95)) if word_counts.size else 0.0,
        "max_words": float(np.max(word_counts)) if word_counts.size else 0.0,
        "mean_chars": float(np.mean(char_counts)) if char_counts.size else 0.0,
    }

    top_terms_by_class: Dict[str, List[Dict[str, int]]] = {}
    for lbl, group in valid_df.groupby(label_col):
        counter: Counter[str] = Counter()
        for doc in group[text_col]:
            words = [w.lower() for w in TOKEN_WORD_RE.findall(doc)]
            counter.update(w for w in words if w not in BASIC_STOPWORDS)
        top_terms_by_class[str(lbl)] = [
            {"term": term, "frequency": count} for term, count in counter.most_common(12)
        ]

    return DatasetSummary(
        total_rows=total_rows,
        valid_rows=len(valid_df),
        missing_text_rows=missing_text,
        missing_label_rows=missing_label,
        duplicate_texts=duplicate_texts,
        num_classes=len(class_counts),
        class_counts=class_counts,
        length_stats=length_stats,
        top_terms_by_class=top_terms_by_class,
    )


def plot_class_distribution(summary: DatasetSummary) -> go.Figure:
    """Bar chart showing exact class balance."""
    fig = px.bar(
        summary.class_counts,
        x="Label",
        y="Count",
        text="Count",
        color_discrete_sequence=["#0F2942"],
        title="Label Distribution in Supplied Corpus",
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        xaxis_title="Category",
        yaxis_title="Post Count",
    )
    return fig


def plot_word_length_histogram(df: pd.DataFrame, text_col: str, label_col: str) -> go.Figure:
    """Histogram of word count per post stratified by label."""
    plot_df = df.dropna(subset=[text_col, label_col]).copy()
    plot_df["word_count"] = plot_df[text_col].astype(str).apply(lambda s: len(s.split()))
    fig = px.histogram(
        plot_df,
        x="word_count",
        color=label_col,
        nbins=30,
        barmode="overlay",
        opacity=0.65,
        title="Word Count Distribution by Class",
        color_discrete_sequence=["#0F2942", "#0D7377", "#2563EB", "#B45309", "#64748B"],
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        xaxis_title="Words per Post",
        yaxis_title="Frequency",
    )
    return fig
