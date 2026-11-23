"""MindTrace AI — Multi-page Streamlit Academic Research Dashboard."""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import AppConfig, DEFAULT_DATA_DIR, DEFAULT_EVAL_CSV, PreprocessingConfig, SUPPORTED_BACKBONES
from src.data_loader import REQUIRED_DATASET_GUIDE, inspect_dataset
from src.dataset_analytics import analyze_corpus, plot_class_distribution, plot_word_length_histogram
from src.ethics import (
    CONFIDENCE_NOT_CLINICAL_PROBABILITY,
    CRISIS_RESOURCES,
    DATASET_LIMITATIONS,
    PRIVACY_STATEMENT,
    RESEARCH_DISCLAIMER,
    XAI_EVIDENCE_BOUNDARY,
    should_show_supportive_guidance,
)
from src.evaluation import (
    EvaluationSummary,
    evaluate_held_out_predictions,
    plot_confusion_matrix_plotly,
    plot_roc_curves_plotly,
)
from src.explainability import ExplainabilityEngine, plot_token_importance
from src.model_inference import MentalHealthTransformerClassifier, ModelNotReadyError
from src.preprocessing import inspect_tokens, preprocess_text, validate_text


st.set_page_config(
    page_title="MindTrace AI — Explainable Mental Health NLP",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #F8FAFC; color: #0F172A; }
      section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
      .mindtrace-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
      }
      .mindtrace-disclaimer {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.9rem;
        color: #1E293B;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_classifier(checkpoint_path: str) -> Optional[MentalHealthTransformerClassifier]:
    cfg = AppConfig().model
    updated = type(cfg)(checkpoint_dir=Path(checkpoint_path))
    try:
        return MentalHealthTransformerClassifier(updated)
    except ModelNotReadyError:
        return None


def render_disclaimer() -> None:
    st.markdown(
        f'<div class="mindtrace-disclaimer"><strong>Research Disclaimer:</strong> {RESEARCH_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def render_supportive_resources() -> None:
    st.warning(
        "Distress-associated language or risk categories were observed in this session. "
        "MindTrace AI cannot provide clinical care. If you or someone you know is struggling, "
        "confidential support is available:"
    )
    for item in CRISIS_RESOURCES:
        st.markdown(f"- **{item['region']} — {item['service']}**: {item['contact']}")


def plot_model_comparison(runs: Dict[str, EvaluationSummary]) -> go.Figure:
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    fig = go.Figure()
    palette = ["#0F2942", "#0D7377", "#2563EB", "#64748B"]
    for idx, (run_name, summary) in enumerate(runs.items()):
        fig.add_trace(
            go.Bar(
                name=f"{run_name} (N={summary.sample_count})",
                x=metrics,
                y=[summary.accuracy, summary.macro_precision, summary.macro_recall, summary.macro_f1],
                marker_color=palette[idx % len(palette)],
            )
        )
    fig.update_layout(
        barmode="group",
        title="Held-Out Model Comparison",
        yaxis_title="Score (0.0 – 1.0)",
        yaxis_range=[0, 1.05],
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
    )
    return fig


def plot_training_history(history_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "train_loss" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=history_df["epoch"],
                y=history_df["train_loss"],
                mode="lines+markers",
                name="Training Loss",
                line=dict(color="#0F2942", width=2.2),
            )
        )
    if "eval_loss" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=history_df["epoch"],
                y=history_df["eval_loss"],
                mode="lines+markers",
                name="Validation Loss",
                line=dict(color="#0D7377", width=2.2),
            )
        )
    fig.update_layout(
        title="Training and Validation Loss by Epoch",
        xaxis_title="Epoch",
        yaxis_title="Cross-Entropy Loss",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
    )
    return fig


def main() -> None:
    cfg = AppConfig()
    with st.sidebar:
        st.title("MindTrace AI")
        st.caption("Explainable Mental Health NLP Research")
        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Text Analysis",
                "Explainable AI",
                "Model Performance",
                "Dataset Analytics",
                "About & Methodology",
            ],
        )
        st.divider()
        checkpoint_input = st.text_input(
            "Model Checkpoint Directory",
            value=str(cfg.model.checkpoint_dir),
            help="Directory containing a fine-tuned DistilBERT or BERT checkpoint.",
        )
        classifier = load_classifier(checkpoint_input)
        if classifier is not None:
            meta = classifier.training_metadata
            if meta and meta.get("training_completed"):
                st.caption(
                    f"Model Status: Trained Checkpoint Loaded ({meta.get('backbone', 'Transformer')})"
                )
            else:
                st.caption("Model Status: Checkpoint Loaded (verify training provenance)")
        else:
            st.caption("Model Status: Not Trained / Setup Required")
        st.caption("Privacy: Ephemeral session memory only")

    render_disclaimer()
    st.write("")

    if page == "Overview":
        st.header("Overview")
        st.caption("Research summary, Transformer architecture, dataset readiness, and held-out evaluation status.")
        st.subheader(cfg.app_subtitle)
        col1, col2, col3 = st.columns(3)
        col1.metric("Primary Backbone", "DistilBERT / BERT")
        col2.metric("Explainability", "SHAP & LIME")
        col3.metric(
            "Saved Model Checkpoint",
            "Loaded" if classifier else "Not Trained / Missing",
        )

        if classifier is not None and classifier.training_metadata:
            meta = classifier.training_metadata
            st.info(
                f"Loaded trained checkpoint ({meta.get('backbone')}) · Seed={meta.get('seed')} · "
                f"Splits: Train={meta.get('train_samples')}, Val={meta.get('val_samples')}, "
                f"Test={meta.get('test_samples')} · Completed={meta.get('completed_at_utc')}"
            )
        elif classifier is None:
            st.info(
                "No fine-tuned model checkpoint is loaded. Supply a labeled CSV dataset and run "
                "`python -m src.train --data_path data/your_dataset.csv` before running inference."
            )

        if "last_eval_summary" in st.session_state:
            ev: EvaluationSummary = st.session_state["last_eval_summary"]
            st.subheader("Held-Out Evaluation Summary")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Accuracy", f"{ev.accuracy:.3f}")
            e2.metric("Macro Precision", f"{ev.macro_precision:.3f}")
            e3.metric("Macro Recall", f"{ev.macro_recall:.3f}")
            e4.metric("Macro F1", f"{ev.macro_f1:.3f}")
        else:
            st.info("Evaluation summary is empty until held-out predictions are evaluated on Model Performance.")

    elif page == "Text Analysis":
        st.header("Text Analysis")
        st.caption("Tokenize input text, inspect raw Transformer logits and softmax probabilities, and preview explanations.")
        raw_text = st.text_area(
            "Social Media Text Input (Ephemeral — not stored)",
            height=170,
            placeholder="Paste an anonymized social media post for preprocessing and Transformer analysis...",
        )
        c1, c2, c3 = st.columns(3)
        mask_urls = c1.checkbox("Mask URLs ([URL])", value=True)
        anon_mentions = c2.checkbox("Anonymize @mentions ([USER])", value=True)
        lowercase = c3.checkbox("Lowercase text", value=False)

        if st.button("Analyse Text", type="primary"):
            prep_cfg = PreprocessingConfig(
                mask_urls=mask_urls, anonymize_mentions=anon_mentions, lowercase=lowercase
            )
            validation = validate_text(raw_text, prep_cfg)
            if not validation.is_valid:
                st.error(validation.error_message)
            else:
                for warn in validation.warnings:
                    st.warning(warn)
                cleaned = preprocess_text(raw_text, prep_cfg)
                st.subheader("Preprocessed Text")
                st.code(cleaned.cleaned_text, language="text")
                st.caption(
                    f"Characters: {cleaned.char_count_cleaned} · Words: {cleaned.word_count_cleaned} · "
                    f"URLs masked: {cleaned.urls_masked} · Mentions masked: {cleaned.mentions_masked}"
                )

                if classifier is None:
                    st.warning(
                        f"Setup Required: No trained checkpoint found at `{checkpoint_input}`. "
                        "Predictions are never simulated. Fine-tune the model using `python -m src.train` first."
                    )
                    st.code(REQUIRED_DATASET_GUIDE, language="text")
                    if should_show_supportive_guidance(raw_text):
                        render_supportive_resources()
                else:
                    tok_info = inspect_tokens(cleaned.cleaned_text, classifier.tokenizer)
                    prediction = classifier.predict(cleaned.cleaned_text)
                    st.session_state["last_raw_text"] = raw_text
                    st.session_state["last_cleaned_text"] = cleaned.cleaned_text
                    st.session_state["last_prediction"] = prediction

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Class", prediction.predicted_label)
                    m2.metric("Model Confidence", f"{prediction.confidence * 100:.1f}%")
                    m3.metric("Uncertainty (Entropy)", f"{prediction.normalized_entropy:.3f}")

                    st.info(CONFIDENCE_NOT_CLINICAL_PROBABILITY)

                    if prediction.is_uncertain:
                        st.warning(
                            "High predictive entropy detected. The model is uncertain across classes; "
                            "interpret this output cautiously."
                        )

                    st.subheader("Class Probability Distribution & Raw Logits")
                    dist_df = pd.DataFrame(
                        [
                            {
                                "Class": cls,
                                "Raw Logit": round(prediction.logits[cls], 4),
                                "Probability": round(prediction.probabilities[cls], 4),
                            }
                            for cls in prediction.probabilities.keys()
                        ]
                    )
                    st.dataframe(dist_df, use_container_width=True)
                    st.bar_chart(dist_df.set_index("Class")["Probability"])
                    st.write("**Transformer Subword Tokens:**", " ".join(f"`{t}`" for t in tok_info["tokens"]))

                    if should_show_supportive_guidance(
                        raw_text, prediction.predicted_label, cfg.distress_label_keywords
                    ):
                        render_supportive_resources()

    elif page == "Explainable AI":
        st.header("Explainable AI (SHAP & LIME)")
        st.caption("Understand how specific words and phrases influenced the Transformer model's prediction.")
        st.info(XAI_EVIDENCE_BOUNDARY)
        with st.expander("Explainability Method Information"):
            st.markdown(
                "- **SHAP (Partition Explainer):** Estimates Shapley attribution values across hierarchical text coalitions. "
                "When the Transformer tokenizer splits a word into WordPiece subwords (e.g., `##ed`, `##ing`), MindTrace AI "
                "sums those additive subword Shapley values back to whole words.\n"
                "- **LIME (Local Surrogate):** Generates perturbed versions of the input text by masking words, queries the "
                "Transformer model, and fits a local weighted linear model to identify influential words.\n"
                "- **Positive vs. Negative Contributions:** Positive weights (teal) increased the model's score for the predicted "
                "class; negative weights (amber) pulled the score away from the predicted class."
            )
        if classifier is None:
            st.warning(
                "Explainability requires a loaded trained Transformer checkpoint. Configure a valid "
                "model directory in the sidebar and analyze a text sample on Text Analysis first."
            )
        elif "last_prediction" not in st.session_state:
            st.info("Submit text on the **Text Analysis** page to inspect SHAP or LIME attributions.")
        else:
            raw_text = st.session_state.get("last_raw_text", st.session_state["last_cleaned_text"])
            cleaned_text = st.session_state["last_cleaned_text"]
            prediction = st.session_state["last_prediction"]

            st.subheader("Original Submitted Text")
            st.text_area("Submitted input", value=raw_text, height=110, disabled=True)

            method = st.radio("Select Explainability Method", ["SHAP", "LIME"], horizontal=True)
            if st.button("Compute Token Explanation", type="primary"):
                with st.spinner(f"Running {method} perturbation attributions against the loaded Transformer..."):
                    engine = ExplainabilityEngine(classifier, cfg.explainability)
                    report = engine.explain(
                        cleaned_text,
                        prediction,
                        method=method,  # type: ignore[arg-type]
                        original_text=raw_text,
                    )
                st.session_state["last_xai_report"] = report

            if "last_xai_report" in st.session_state:
                report = st.session_state["last_xai_report"]
                st.subheader("Highlighted Influential Words")
                st.markdown(report.highlighted_html, unsafe_allow_html=True)

                st.subheader("Feature Importance Chart")
                st.plotly_chart(plot_token_importance(report), use_container_width=True)

                st.subheader("Explanation Summary")
                st.write(report.natural_language_summary)

                export_payload = {
                    "model_name": prediction.model_name,
                    "original_text": report.original_text,
                    "cleaned_text": report.analyzed_text,
                    "predicted_label": prediction.predicted_label,
                    "confidence": prediction.confidence,
                    "logits": prediction.logits,
                    "probabilities": prediction.probabilities,
                    "normalized_entropy": prediction.normalized_entropy,
                    "is_uncertain": prediction.is_uncertain,
                    "tokens": prediction.tokens,
                    "method": report.method,
                    "attributions": [
                        {
                            "token": a.token,
                            "weight": a.weight,
                            "direction": a.direction,
                            "start": a.start,
                            "end": a.end,
                        }
                        for a in report.attributions
                    ],
                    "natural_language_summary": report.natural_language_summary,
                }
                st.download_button(
                    "Export XAI Report as JSON",
                    data=json.dumps(export_payload, indent=2),
                    file_name="mindtrace_xai_report.json",
                    mime="application/json",
                )

    elif page == "Model Performance":
        st.header("Model Performance (Held-Out Evaluation)")
        st.caption("Audit held-out accuracy, precision, recall, F1, confusion matrix, model comparisons, and training curves.")
        uploaded_eval = st.file_uploader(
            "Upload Held-Out Evaluation CSV (requires true_label, predicted_label, optional prob_<class>)",
            type=["csv"],
        )
        run_label = st.text_input("Run Label (for multi-model comparison)", value="DistilBERT")
        eval_df: Optional[pd.DataFrame] = None
        if uploaded_eval is not None:
            eval_df = pd.read_csv(uploaded_eval)
        elif DEFAULT_EVAL_CSV.exists():
            eval_df = pd.read_csv(DEFAULT_EVAL_CSV)

        if eval_df is None:
            st.info(
                "Setup Required: No held-out evaluation predictions file is loaded. Upload a CSV "
                "containing `true_label` and `predicted_label` columns (generated by `src/train.py`) "
                "to compute genuine Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC."
            )
        else:
            summary = evaluate_held_out_predictions(eval_df)
            st.session_state["last_eval_summary"] = summary
            runs = st.session_state.setdefault("eval_runs", {})
            runs[run_label.strip() or "Model Run"] = summary

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Samples (N)", summary.sample_count)
            c2.metric("Accuracy", f"{summary.accuracy:.3f}")
            c3.metric("Macro Precision", f"{summary.macro_precision:.3f}")
            c4.metric("Macro Recall", f"{summary.macro_recall:.3f}")
            c5.metric("Macro F1", f"{summary.macro_f1:.3f}")
            if summary.roc_auc_macro is not None:
                st.metric("Macro ROC-AUC", f"{summary.roc_auc_macro:.3f}")

            st.plotly_chart(plot_confusion_matrix_plotly(summary), use_container_width=True)
            roc_fig = plot_roc_curves_plotly(summary)
            if roc_fig is not None:
                st.plotly_chart(roc_fig, use_container_width=True)
            if len(runs) >= 2:
                st.plotly_chart(plot_model_comparison(runs), use_container_width=True)
            st.subheader("Per-Class Classification Report")
            st.dataframe(summary.per_class_df, use_container_width=True)

        st.divider()
        st.subheader("Training & Validation Curves")
        uploaded_history = st.file_uploader(
            "Upload Training History CSV (epoch, train_loss, eval_loss, optional eval_accuracy/eval_f1)",
            type=["csv"],
        )
        default_history = DEFAULT_DATA_DIR / "training_history.csv"
        history_df: Optional[pd.DataFrame] = None
        if uploaded_history is not None:
            history_df = pd.read_csv(uploaded_history)
        elif default_history.exists():
            history_df = pd.read_csv(default_history)
        if history_df is not None and "epoch" in history_df.columns:
            st.plotly_chart(plot_training_history(history_df), use_container_width=True)
        else:
            st.info("No training_history.csv loaded yet. Run `python -m src.train` or upload epoch logs above.")

    elif page == "Dataset Analytics":
        st.header("Dataset Analytics")
        st.caption("Inspect dataset columns, target labels, class imbalance, text lengths, splits, and data quality.")
        uploaded_corpus = st.file_uploader("Upload Labeled Social Media CSV", type=["csv"])
        if uploaded_corpus is None:
            st.info("Setup Required: Upload a labeled CSV corpus to inspect real statistics.")
            st.code(REQUIRED_DATASET_GUIDE, language="text")
        else:
            corpus_df = pd.read_csv(uploaded_corpus)
            cols = list(corpus_df.columns)
            col_a, col_b = st.columns(2)
            text_col = col_a.selectbox("Text Column", cols, index=cols.index("text") if "text" in cols else 0)
            label_col = col_b.selectbox("Label Column", cols, index=cols.index("label") if "label" in cols else min(1, len(cols) - 1))

            inspection = inspect_dataset(corpus_df, text_col=text_col, label_col=label_col)
            summary = analyze_corpus(corpus_df, text_col=text_col, label_col=label_col)
            st.session_state["last_dataset_summary"] = summary

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Valid Rows", summary.valid_rows)
            m2.metric("Classes", summary.num_classes)
            m3.metric("Duplicate Texts", summary.duplicate_texts)
            m4.metric("Imbalance Ratio", f"{inspection.imbalance_ratio:.2f}x")

            st.plotly_chart(plot_class_distribution(summary), use_container_width=True)
            st.plotly_chart(plot_word_length_histogram(corpus_df, text_col, label_col), use_container_width=True)

            if "split" in corpus_df.columns:
                st.subheader("Dataset Splitting Distribution")
                st.dataframe(corpus_df["split"].value_counts().reset_index(), use_container_width=True)
            else:
                st.caption(
                    f"Projected 70/15/15 Stratified Split (after deduplication): Train ≈ {int(summary.valid_rows * 0.7)}, "
                    f"Validation ≈ {int(summary.valid_rows * 0.15)}, Test ≈ {summary.valid_rows - int(summary.valid_rows * 0.7) - int(summary.valid_rows * 0.15)}."
                )

    elif page == "About & Methodology":
        st.header("About & Methodology")
        st.caption("Academic research design, Transformer modeling choices, ethical safeguards, and limitations.")
        st.markdown(f"**Privacy Policy:** {PRIVACY_STATEMENT}")
        st.subheader("Supported Transformer Backbones")
        for name, hf_id in SUPPORTED_BACKBONES.items():
            st.markdown(f"- **{name}**: `{hf_id}`")
        st.subheader("Dataset Bias & Model Limitations")
        for item in DATASET_LIMITATIONS:
            st.markdown(f"- **{item['title']}**: {item['detail']}")
        st.subheader("Supportive Crisis Resources")
        for res in CRISIS_RESOURCES:
            st.markdown(f"- **{res['region']} — {res['service']}**: {res['contact']}")


if __name__ == "__main__":
    main()
