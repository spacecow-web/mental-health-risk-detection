"""Held-out model evaluation module using Scikit-learn, Plotly, and Matplotlib/Seaborn."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    clean_df[true_col] = clean_df[true_col].astype(str).str.strip()
    clean_df[pred_col] = clean_df[pred_col].astype(str).str.strip()
    clean_df = clean_df[(clean_df[true_col] != "") & (clean_df[pred_col] != "")]
    if clean_df.empty:
        raise ValueError("Evaluation CSV contains zero valid labeled rows.")

    y_true = clean_df[true_col].tolist()
    y_pred = clean_df[pred_col].tolist()
    labels = sorted(set(y_true) | set(y_pred))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class_rows = [
        {
            "Label": lbl,
            "Precision": round(float(report_dict[lbl]["precision"]), 4),
            "Recall": round(float(report_dict[lbl]["recall"]), 4),
            "F1-Score": round(float(report_dict[lbl]["f1-score"]), 4),
            "Support": int(report_dict[lbl]["support"]),
        }
        for lbl in labels
    ]
    per_class_rows.append(
        {
            "Label": "Macro Average",
            "Precision": round(float(macro_p), 4),
            "Recall": round(float(macro_r), 4),
            "F1-Score": round(float(macro_f1), 4),
            "Support": len(clean_df),
        }
    )
    per_class_rows.append(
        {
            "Label": "Weighted Average",
            "Precision": round(float(weighted_p), 4),
            "Recall": round(float(weighted_r), 4),
            "F1-Score": round(float(weighted_f1), 4),
            "Support": len(clean_df),
        }
    )
    per_class_df = pd.DataFrame(per_class_rows)

    roc_curves: Dict[str, Dict[str, np.ndarray]] = {}
    roc_auc_macro: Optional[float] = None
    prob_cols = {lbl: f"prob_{lbl}" for lbl in labels if f"prob_{lbl}" in clean_df.columns}

    if len(prob_cols) == len(labels) and len(labels) >= 2:
        prob_df = clean_df[[prob_cols[lbl] for lbl in labels]].apply(pd.to_numeric, errors="coerce")
        if prob_df.notna().all().all():
            prob_matrix = prob_df.to_numpy(dtype=float)
            if np.isfinite(prob_matrix).all():
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


def evaluate_classifier_on_dataframe(
    classifier: Any,
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    split_col: Optional[str] = "split",
    max_samples: int = 200,
) -> pd.DataFrame:
    """Run real classifier inference on held-out test records (no file upload required)."""
    work_df = df.dropna(subset=[text_col, label_col]).copy()
    if split_col and split_col in work_df.columns:
        test_subset = work_df[work_df[split_col].astype(str).str.lower().str.contains("test", na=False)]
        if not test_subset.empty:
            work_df = test_subset
    if len(work_df) > max_samples:
        work_df = work_df.sample(n=max_samples, random_state=42)

    rows: List[Dict[str, Any]] = []
    for _, row in work_df.iterrows():
        text_val = str(row[text_col]).strip()
        true_val = str(row[label_col]).strip()
        if not text_val or not true_val:
            continue
        pred = classifier.predict(text_val)
        record: Dict[str, Any] = {
            "text": text_val,
            "true_label": true_val,
            "predicted_label": pred.predicted_label,
        }
        for lbl, prob in pred.probabilities.items():
            record[f"prob_{lbl}"] = float(prob)
        rows.append(record)

    if not rows:
        raise ValueError("No valid records could be evaluated from the active dataset.")
    return pd.DataFrame(rows)


def save_evaluation_summary(
    summary: EvaluationSummary,
    model_name: str,
    source_file: str,
    output_path: Path,
) -> None:
    """Persist evaluation metrics to disk as JSON for automatic reloading."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "source_file": source_file,
        "sample_count": summary.sample_count,
        "labels": summary.labels,
        "accuracy": summary.accuracy,
        "macro_precision": summary.macro_precision,
        "macro_recall": summary.macro_recall,
        "macro_f1": summary.macro_f1,
        "weighted_f1": summary.weighted_f1,
        "roc_auc_macro": summary.roc_auc_macro,
        "confusion_matrix": summary.confusion_matrix.tolist(),
        "per_class": summary.per_class_df.to_dict(orient="records"),
        "roc_curves": {
            lbl: {
                "fpr": curves["fpr"].tolist(),
                "tpr": curves["tpr"].tolist(),
                "auc": curves["auc"].tolist(),
            }
            for lbl, curves in summary.roc_curves.items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_saved_evaluation_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Reload previously persisted evaluation metrics JSON if available."""
    if not summary_path.exists() or not summary_path.is_file():
        return None
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = EvaluationSummary(
            sample_count=int(raw["sample_count"]),
            labels=list(raw["labels"]),
            accuracy=float(raw["accuracy"]),
            macro_precision=float(raw["macro_precision"]),
            macro_recall=float(raw["macro_recall"]),
            macro_f1=float(raw["macro_f1"]),
            weighted_f1=float(raw["weighted_f1"]),
            roc_auc_macro=float(raw["roc_auc_macro"]) if raw.get("roc_auc_macro") is not None else None,
            confusion_matrix=np.array(raw["confusion_matrix"], dtype=int),
            per_class_df=pd.DataFrame(raw.get("per_class", [])),
            roc_curves={
                lbl: {
                    "fpr": np.array(c["fpr"], dtype=float),
                    "tpr": np.array(c["tpr"], dtype=float),
                    "auc": np.array(c["auc"], dtype=float),
                }
                for lbl, c in raw.get("roc_curves", {}).items()
            },
        )
        return {
            "model_name": raw.get("model_name", "Transformer Checkpoint"),
            "source_file": raw.get("source_file", str(summary_path)),
            "summary": summary,
        }
    except Exception:
        return None


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
