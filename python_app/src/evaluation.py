"""Held-out model evaluation module using Scikit-learn, Plotly, and Matplotlib/Seaborn."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


@dataclass
class EvaluationSummary:
    """Exact held-out metrics computed strictly from real predictions."""

    sample_count: int
    labels: List[str]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    roc_auc_macro: Optional[float]
    confusion_matrix: np.ndarray
    per_class_df: pd.DataFrame
    roc_curves: Dict[str, Dict[str, np.ndarray]]


def evaluate_held_out_predictions(
    df: pd.DataFrame,
    true_col: str = "true_label",
    pred_col: str = "predicted_label",
) -> EvaluationSummary:
    """Compute held-out classification metrics from an evaluation DataFrame."""
    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Evaluation CSV must contain '{true_col}' and '{pred_col}' columns.")

    clean_df = df.dropna(subset=[true_col, pred_col]).copy()
    if clean_df.empty:
        raise ValueError("Evaluation CSV contains zero valid labeled rows.")

    y_true = clean_df[true_col].astype(str).tolist()
    y_pred = clean_df[pred_col].astype(str).tolist()
    labels = sorted(set(y_true) | set(y_pred))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class_rows = [
        {
            "Label": lbl,
            "Precision": float(report_dict[lbl]["precision"]),
            "Recall": float(report_dict[lbl]["recall"]),
            "F1-Score": float(report_dict[lbl]["f1-score"]),
            "Support": int(report_dict[lbl]["support"]),
        }
        for lbl in labels
    ]
    per_class_df = pd.DataFrame(per_class_rows)

    roc_curves: Dict[str, Dict[str, np.ndarray]] = {}
    roc_auc_macro: Optional[float] = None
    prob_cols = {lbl: f"prob_{lbl}" for lbl in labels if f"prob_{lbl}" in clean_df.columns}

    if len(prob_cols) == len(labels) and len(labels) >= 2:
        prob_matrix = clean_df[[prob_cols[lbl] for lbl in labels]].astype(float).to_numpy()
        if len(labels) == 2:
            pos_label = labels[1]
            y_bin = np.array([1 if y == pos_label else 0 for y in y_true])
            if len(np.unique(y_bin)) == 2:
                fpr, tpr, _ = roc_curve(y_bin, prob_matrix[:, 1])
                auc_val = float(roc_auc_score(y_bin, prob_matrix[:, 1]))
                roc_auc_macro = auc_val
                roc_curves[pos_label] = {"fpr": fpr, "tpr": tpr, "auc": np.array([auc_val])}
        else:
            y_bin = label_binarize(y_true, classes=labels)
            aucs: List[float] = []
            for idx, lbl in enumerate(labels):
                if len(np.unique(y_bin[:, idx])) == 2:
                    fpr, tpr, _ = roc_curve(y_bin[:, idx], prob_matrix[:, idx])
                    auc_val = float(roc_auc_score(y_bin[:, idx], prob_matrix[:, idx]))
                    aucs.append(auc_val)
                    roc_curves[lbl] = {"fpr": fpr, "tpr": tpr, "auc": np.array([auc_val])}
            if aucs:
                roc_auc_macro = float(np.mean(aucs))

    return EvaluationSummary(
        sample_count=len(clean_df),
        labels=labels,
        accuracy=acc,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        roc_auc_macro=roc_auc_macro,
        confusion_matrix=cm,
        per_class_df=per_class_df,
        roc_curves=roc_curves,
    )


def plot_confusion_matrix_plotly(summary: EvaluationSummary) -> go.Figure:
    """Interactive Plotly confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=summary.confusion_matrix,
            x=summary.labels,
            y=summary.labels,
            colorscale=[[0, "#F8FAFC"], [0.5, "#93C5FD"], [1, "#0F2942"]],
            text=summary.confusion_matrix,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Held-Out Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis_autorange="reversed",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
    )
    return fig


def plot_confusion_matrix_seaborn(summary: EvaluationSummary) -> plt.Figure:
    """Publication-ready Matplotlib/Seaborn confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        summary.confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=summary.labels,
        yticklabels=summary.labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix (N={summary.sample_count})")
    fig.tight_layout()
    return fig


def plot_roc_curves_plotly(summary: EvaluationSummary) -> Optional[go.Figure]:
    """Interactive One-vs-Rest ROC-AUC curves when predicted probabilities are present."""
    if not summary.roc_curves:
        return None
    fig = go.Figure()
    palette = ["#0F2942", "#0D7377", "#2563EB", "#B45309", "#64748B"]
    for idx, (lbl, data) in enumerate(summary.roc_curves.items()):
        auc_val = float(data["auc"][0])
        fig.add_trace(
            go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=f"{lbl} (AUC = {auc_val:.3f})",
                line=dict(color=palette[idx % len(palette)], width=2.2),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline (0.500)",
            line=dict(color="#94A3B8", dash="dash", width=1.2),
        )
    )
    fig.update_layout(
        title="Receiver Operating Characteristic (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
    )
    return fig
