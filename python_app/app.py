"""MindTrace AI — Multi-page Streamlit Academic Research Dashboard."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    AppConfig,
    DEFAULT_EVAL_CSV,
    DEFAULT_EVAL_SUMMARY_JSON,
    DEFAULT_HISTORY_CSV,
    PreprocessingConfig,
    SUPPORTED_BACKBONES,
)
from src.data_loader import (
    detect_dataset_columns,
    resolve_default_dataset_path,
)
from src.dataset_analytics import (
    DatasetSummary,
    analyze_corpus,
    plot_class_distribution,
    plot_split_distribution,
    plot_word_length_histogram,
)
from src.ethics import (
    ACADEMIC_METHODOLOGY_SECTIONS,
    ACADEMIC_REFERENCES,
    CONFIDENCE_NOT_CLINICAL_PROBABILITY,
    CRISIS_RESOURCES,
    DATASET_LIMITATIONS,
    IMMEDIATE_SAFETY_MESSAGE,
    PRIVACY_STATEMENT,
    RESEARCH_DISCLAIMER,
    XAI_EVIDENCE_BOUNDARY,
    detect_immediate_self_harm_cues,
    should_show_supportive_guidance,
)
from src.evaluation import (
    EvaluationSummary,
    evaluate_classifier_on_dataframe,
    evaluate_held_out_predictions,
    load_saved_evaluation_summary,
    plot_confusion_matrix_plotly,
    plot_roc_curves_plotly,
    save_evaluation_summary,
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
    """Load and cache the configured Transformer checkpoint across interactions."""
    cfg = AppConfig().model
    updated = type(cfg)(checkpoint_dir=Path(checkpoint_path))
    try:
        return MentalHealthTransformerClassifier(updated)
    except ModelNotReadyError:
        return None


@st.cache_data(show_spinner=False)
def load_default_dataset_cached() -> Optional[Tuple[str, pd.DataFrame, DatasetSummary, str, str]]:
    """Auto-load and analyze the configured default CSV dataset (cached across tabs)."""
    dataset_path = resolve_default_dataset_path()
    if dataset_path is None:
        return None
    df = pd.read_csv(dataset_path, encoding="utf-8")
    if df.empty or len(df.columns) < 2:
        return None
    text_col, label_col, split_col = detect_dataset_columns(df)
    summary = analyze_corpus(
        df,
        text_col=text_col,
        label_col=label_col,
        dataset_name=dataset_path.stem,
        dataset_source=f"Configured default dataset ({dataset_path.name})",
        split_col=split_col,
    )
    return str(dataset_path), df, summary, text_col, label_col


@st.cache_data(show_spinner=False)
def load_saved_evaluation_cached() -> Optional[Dict[str, Any]]:
    """Automatically load persisted evaluation summary JSON or held_out_predictions.csv."""
    saved = load_saved_evaluation_summary(DEFAULT_EVAL_SUMMARY_JSON)
    if saved is not None:
        return saved
    if DEFAULT_EVAL_CSV.exists():
        eval_df = pd.read_csv(DEFAULT_EVAL_CSV, encoding="utf-8")
        summary = evaluate_held_out_predictions(eval_df)
        return {
            "model_name": "Saved Transformer Checkpoint",
            "source_file": str(DEFAULT_EVAL_CSV),
            "summary": summary,
        }
    return None


@st.cache_data(show_spinner=False)
def load_saved_history_cached() -> Optional[pd.DataFrame]:
    """Automatically load persisted training_history.csv if generated after training."""
    if not DEFAULT_HISTORY_CSV.exists():
        return None
    try:
        df = pd.read_csv(DEFAULT_HISTORY_CSV, encoding="utf-8")
        return df if "epoch" in df.columns else None
    except Exception:
        return None


def render_disclaimer() -> None:
    st.markdown(
        f'<div class="mindtrace-disclaimer">'
        f'<strong>Educational Research Disclaimer:</strong> {RESEARCH_DISCLAIMER}<br/>'
        f'<strong>{PRIVACY_STATEMENT.split(":", 1)[0]}:</strong>{PRIVACY_STATEMENT.split(":", 1)[1]}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_supportive_resources(immediate_safety: bool = False) -> None:
    if immediate_safety:
        st.info(IMMEDIATE_SAFETY_MESSAGE)
    else:
        st.warning(
            "Distress-associated language or risk categories were observed in this session. "
            "MindTrace AI cannot provide clinical care. Reaching out to a trusted person or "
            "free confidential support can help:"
        )
    for item in CRISIS_RESOURCES:
        st.markdown(f"- **{item['region']} — {item['service']}**: {item['contact']}")


def plot_model_comparison(runs: Dict[str, EvaluationSummary]) -> go.Figure:
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    fig = go.Figure()
    palette = ["#0F2942", "#0D7377", "#2563EB", "#64748B"]
    for idx, (run_name, summary) in enumerate(runs.items()):
        fig.add_trace(
            go.Bar(
                name=f"{run_name} (N={summary.sample_count})",
                x=metrics,
                y=[
                    summary.accuracy,
                    summary.macro_precision,
                    summary.macro_recall,
                    summary.macro_f1,
                    summary.weighted_f1,
                ],
                marker_color=palette[idx % len(palette)],
            )
        )
    fig.update_layout(
        barmode="group",
        title="Held-Out Transformer Model Comparison",
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
    checkpoint_dir = st.session_state.get("custom_checkpoint_dir", str(cfg.model.checkpoint_dir))
    classifier = load_classifier(checkpoint_dir)
    default_dataset_bundle = load_default_dataset_cached()
    saved_eval_bundle = load_saved_evaluation_cached()
    saved_history_df = load_saved_history_cached()

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
        if classifier is not None:
            meta = classifier.training_metadata
            if meta and meta.get("training_completed"):
                st.caption(
                    f"Active Model: Trained ({meta.get('backbone', 'Transformer')})"
                )
            else:
                st.caption("Active Model: Checkpoint Loaded")
        else:
            st.caption("Active Model: Not Trained / Setup Needed")

        if default_dataset_bundle is not None:
            st.caption(f"Dataset: {default_dataset_bundle[2].valid_rows} usable posts ready")
        st.caption("Privacy: Ephemeral session memory only")

    render_disclaimer()
    st.write("")

    if page == "Overview":
        st.header("Overview")
        st.caption(
            "Research summary, Transformer architecture, auto-loaded dataset statistics, and persisted evaluation status."
        )
        st.subheader(cfg.app_subtitle)
        col1, col2, col3 = st.columns(3)
        col1.metric("Primary Backbone", "DistilBERT / BERT")
        col2.metric("Explainability", "SHAP & LIME")
        col3.metric("Model Checkpoint", "Loaded" if classifier else "Not Yet Trained")

        if classifier is not None and classifier.training_metadata:
            meta = classifier.training_metadata
            st.info(
                f"Active checkpoint ({meta.get('backbone')}) · Seed={meta.get('seed')} · "
                f"Splits: Train={meta.get('train_samples')}, Val={meta.get('val_samples')}, "
                f"Test={meta.get('test_samples')} · Completed={meta.get('completed_at_utc')}"
            )
        elif classifier is None:
            st.info(
                f"No fine-tuned checkpoint found yet at `{checkpoint_dir}`. Run `python -m src.train` "
                "to fine-tune and activate local DistilBERT / BERT inference."
            )

        active_ds = st.session_state.get("custom_dataset_summary") or (
            default_dataset_bundle[2] if default_dataset_bundle else None
        )
        if active_ds is not None:
            st.subheader("Dataset Summary & Classification Distribution")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Dataset", active_ds.dataset_name)
            d2.metric("Usable Records", active_ds.valid_rows)
            d3.metric("Classes", active_ds.num_classes)
            d4.metric("Imbalance Ratio", f"{active_ds.imbalance_ratio:.2f}x")
            st.plotly_chart(plot_class_distribution(active_ds), use_container_width=True)
        else:
            st.info(
                "No default dataset CSV found at `data/default_dataset.csv`. Place a labeled CSV there "
                "to populate dataset summary charts automatically."
            )

        active_eval: Optional[EvaluationSummary] = st.session_state.get("last_eval_summary") or (
            saved_eval_bundle["summary"] if saved_eval_bundle else None
        )
        if active_eval is not None:
            st.subheader("Saved Model Evaluation Summary")
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Accuracy", f"{active_eval.accuracy:.3f}")
            e2.metric("Macro Precision", f"{active_eval.macro_precision:.3f}")
            e3.metric("Macro Recall", f"{active_eval.macro_recall:.3f}")
            e4.metric("Macro F1", f"{active_eval.macro_f1:.3f}")
            e5.metric("Weighted F1", f"{active_eval.weighted_f1:.3f}")
        else:
            st.info("No saved model evaluation yet. Visit Model Performance to run evaluation.")

    elif page == "Text Analysis":
        st.header("Text Analysis")
        st.caption(
            "Analyze social media text directly using the automatically loaded Transformer model."
        )
        raw_text = st.text_area(
            "Social Media Text Input (Ephemeral — not stored)",
            value=st.session_state.get("session_input_text", ""),
            height=170,
            placeholder="Paste or type a social media post for Transformer analysis...",
        )
        st.session_state["session_input_text"] = raw_text

        with st.expander("Advanced Options (Preprocessing & Checkpoint Directory)"):
            c1, c2, c3 = st.columns(3)
            mask_urls = c1.checkbox("Mask URLs ([URL])", value=True)
            anon_mentions = c2.checkbox("Anonymize @mentions ([USER])", value=True)
            lowercase = c3.checkbox("Lowercase text", value=False)
            new_ckpt = st.text_input(
                "Model Checkpoint Directory",
                value=checkpoint_dir,
                help="Optional path to a custom fine-tuned Transformer checkpoint directory.",
            )
            if new_ckpt != checkpoint_dir:
                st.session_state["custom_checkpoint_dir"] = new_ckpt
                st.rerun()

        if classifier is None:
            st.warning(
                f"Setup Notice: No saved model checkpoint was found at `{checkpoint_dir}`. "
                "Run `python -m src.train` to fine-tune and save a Transformer checkpoint before analysing text."
            )

        if st.button("Analyse Text", type="primary"):
            prep_cfg = PreprocessingConfig(
                mask_urls=mask_urls, anonymize_mentions=anon_mentions, lowercase=lowercase
            )
            validation = validate_text(raw_text, prep_cfg)
            immediate_safety = detect_immediate_self_harm_cues(raw_text)
            if immediate_safety:
                render_supportive_resources(immediate_safety=True)

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
                    st.error(
                        f"Cannot execute Transformer inference because no checkpoint exists at `{checkpoint_dir}`."
                    )
                    if not immediate_safety and should_show_supportive_guidance(raw_text):
                        render_supportive_resources(immediate_safety=False)
                else:
                    tok_info = inspect_tokens(cleaned.cleaned_text, classifier.tokenizer)
                    prediction = classifier.predict(cleaned.cleaned_text)
                    st.session_state["last_raw_text"] = raw_text
                    st.session_state["last_cleaned_text"] = cleaned.cleaned_text
                    st.session_state["last_prediction"] = prediction
                    st.session_state.pop("last_xai_report", None)

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

                    if not immediate_safety and should_show_supportive_guidance(
                        raw_text, prediction.predicted_label, cfg.distress_label_keywords
                    ):
                        render_supportive_resources(immediate_safety=False)

    elif page == "Explainable AI":
        st.header("Explainable AI (SHAP & LIME)")
        st.caption("Automatic SHAP and LIME attribution for the latest analyzed text.")
        st.info(XAI_EVIDENCE_BOUNDARY)

        if "last_prediction" not in st.session_state or classifier is None:
            st.info("Analyse some text to view its explanation.")
        else:
            raw_text = st.session_state.get("last_raw_text", st.session_state["last_cleaned_text"])
            cleaned_text = st.session_state["last_cleaned_text"]
            prediction = st.session_state["last_prediction"]

            st.subheader("Latest Analyzed Text")
            st.text_area("Analyzed input", value=raw_text, height=110, disabled=True)

            method = st.radio("Select Explainability Method", ["SHAP", "LIME"], horizontal=True)
            cached_report = st.session_state.get("last_xai_report")

            if cached_report is None or cached_report.method != method:
                try:
                    with st.spinner(
                        f"Computing {method} attributions against the active Transformer..."
                    ):
                        engine = ExplainabilityEngine(classifier, cfg.explainability)
                        report = engine.explain(
                            cleaned_text,
                            prediction,
                            method=method,  # type: ignore[arg-type]
                            original_text=raw_text,
                        )
                    st.session_state["last_xai_report"] = report
                    cached_report = report
                except Exception as exc:
                    st.error(f"Could not generate {method} explanation: {exc}")

            if cached_report is not None:
                st.subheader("Highlighted Influential Words")
                st.markdown(cached_report.highlighted_html, unsafe_allow_html=True)

                st.subheader("Feature Importance Chart")
                st.plotly_chart(plot_token_importance(cached_report), use_container_width=True)

                st.subheader("Explanation Summary")
                st.write(cached_report.natural_language_summary)

                with st.expander("Explainability Method Details"):
                    st.markdown(
                        "- **SHAP (Partition Explainer):** Estimates Shapley attribution values across hierarchical text coalitions. "
                        "WordPiece subwords are summed back to whole words.\n"
                        "- **LIME (Local Surrogate):** Perturbs words around the input post and fits a local linear model.\n"
                        "- **Positive vs. Negative Contributions:** Positive weights (teal) increased the score for the predicted class; "
                        "negative weights (amber) pushed the score away."
                    )

                export_payload = {
                    "model_name": prediction.model_name,
                    "original_text": cached_report.original_text,
                    "cleaned_text": cached_report.analyzed_text,
                    "predicted_label": prediction.predicted_label,
                    "confidence": prediction.confidence,
                    "logits": prediction.logits,
                    "probabilities": prediction.probabilities,
                    "normalized_entropy": prediction.normalized_entropy,
                    "is_uncertain": prediction.is_uncertain,
                    "tokens": prediction.tokens,
                    "method": cached_report.method,
                    "attributions": [
                        {
                            "token": a.token,
                            "weight": a.weight,
                            "direction": a.direction,
                            "start": a.start,
                            "end": a.end,
                        }
                        for a in cached_report.attributions
                    ],
                    "phrase_attributions": [
                        {
                            "phrase": p.phrase,
                            "weight": p.weight,
                            "direction": p.direction,
                            "start": p.start,
                            "end": p.end,
                            "tokens": p.tokens,
                        }
                        for p in cached_report.phrase_attributions
                    ],
                    "natural_language_summary": cached_report.natural_language_summary,
                }
                st.download_button(
                    "Export XAI Report as JSON",
                    data=json.dumps(export_payload, indent=2),
                    file_name="mindtrace_xai_report.json",
                    mime="application/json",
                )

    elif page == "Model Performance":
        st.header("Model Performance (Saved & Live Evaluation)")
        st.caption(
            "Automatically displays saved evaluation results, or runs fresh evaluation directly on the active model and default dataset."
        )

        # Direct in-app evaluation trigger without requiring uploads
        eval_col_a, eval_col_b = st.columns([2, 1])
        eval_col_a.write(
            "Evaluation results generated by `python -m src.train` or **Run Model Evaluation** are saved automatically."
        )
        if eval_col_b.button("Run Model Evaluation", type="primary"):
            if classifier is None:
                st.error(
                    f"Cannot run model evaluation because no checkpoint is loaded at `{checkpoint_dir}`."
                )
            elif default_dataset_bundle is None:
                st.error(
                    "Cannot run model evaluation because no default dataset was found in `data/`."
                )
            else:
                try:
                    with st.spinner("Evaluating active Transformer checkpoint on held-out records..."):
                        _, df_raw, _, t_col, l_col = default_dataset_bundle
                        split_c = "split" if "split" in df_raw.columns else None
                        eval_df_live = evaluate_classifier_on_dataframe(
                            classifier, df_raw, text_col=t_col, label_col=l_col, split_col=split_c
                        )
                        DEFAULT_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
                        eval_df_live.to_csv(DEFAULT_EVAL_CSV, index=False)
                        live_summary = evaluate_held_out_predictions(eval_df_live)
                        model_label = (
                            classifier.training_metadata.get("backbone", "Active Transformer")
                            if classifier.training_metadata
                            else "Active Transformer"
                        )
                        save_evaluation_summary(
                            live_summary,
                            model_name=model_label,
                            source_file=str(DEFAULT_EVAL_CSV),
                            output_path=DEFAULT_EVAL_SUMMARY_JSON,
                        )
                        load_saved_evaluation_cached.clear()
                        st.session_state["last_eval_summary"] = live_summary
                        runs = st.session_state.setdefault("eval_runs", {})
                        runs[model_label] = live_summary
                        st.success("Model evaluation completed and saved to disk.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

        summary: Optional[EvaluationSummary] = st.session_state.get("last_eval_summary") or (
            saved_eval_bundle["summary"] if saved_eval_bundle else None
        )
        runs: Dict[str, EvaluationSummary] = st.session_state.setdefault("eval_runs", {})
        if saved_eval_bundle and saved_eval_bundle["model_name"] not in runs:
            runs[saved_eval_bundle["model_name"]] = saved_eval_bundle["summary"]

        if summary is None:
            st.info(
                "No saved evaluation results exist yet. Click **Run Model Evaluation** above or run "
                "`python -m src.train` to generate and save genuine evaluation metrics."
            )
        else:
            st.session_state["last_eval_summary"] = summary
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Samples (N)", summary.sample_count)
            c2.metric("Accuracy", f"{summary.accuracy:.3f}")
            c3.metric("Macro Precision", f"{summary.macro_precision:.3f}")
            c4.metric("Macro Recall", f"{summary.macro_recall:.3f}")
            c5.metric("Macro F1", f"{summary.macro_f1:.3f}")
            c6.metric("Weighted F1", f"{summary.weighted_f1:.3f}")
            st.caption(
                "Accuracy shows overall correct predictions. Precision measures how often a predicted label is right; "
                "Recall measures how many true cases the model catches. Macro F1 weighs every class equally, while "
                "Weighted F1 accounts for class sizes."
            )

            if summary.roc_auc_macro is not None:
                st.metric("Macro ROC-AUC", f"{summary.roc_auc_macro:.3f}")

            st.plotly_chart(plot_confusion_matrix_plotly(summary), use_container_width=True)
            st.caption(
                "What this shows: Numbers along the top-left to bottom-right diagonal are correct predictions. "
                "Off-diagonal cells highlight which categories the model confused with one another."
            )

            roc_fig = plot_roc_curves_plotly(summary)
            if roc_fig is not None:
                st.plotly_chart(roc_fig, use_container_width=True)
                st.caption(
                    "What this shows: Curves closer to the top-left corner (AUC near 1.0) indicate stronger separation "
                    "between that category and all others across confidence thresholds; 0.50 equals random guessing."
                )

            if len(runs) >= 2:
                st.plotly_chart(plot_model_comparison(runs), use_container_width=True)
                st.caption(
                    "What this shows: Side-by-side comparison of Transformer runs evaluated on held-out test predictions."
                )

            st.subheader("Classification Report")
            st.dataframe(summary.per_class_df, use_container_width=True)

        st.divider()
        st.subheader("Training & Validation Curves")
        active_history = st.session_state.get("custom_history_df")
        if active_history is None:
            active_history = saved_history_df

        if active_history is not None and "epoch" in active_history.columns:
            st.plotly_chart(plot_training_history(active_history), use_container_width=True)
            st.caption(
                "What this shows: Steady decreases in both training and validation loss indicate healthy learning. "
                "If validation loss rises while training loss falls, the model is beginning to overfit."
            )
        else:
            st.info("No saved training logs found yet. Training curves populate automatically after `python -m src.train`.")

        with st.expander("Advanced Options (Optional Custom Evaluation & Training History CSVs)"):
            run_label = st.selectbox("Custom Run Label", ["DistilBERT", "BERT-base", "Custom Model"])
            if run_label == "Custom Model":
                run_label = st.text_input("Custom Model Name", value="Custom Transformer").strip() or "Custom Transformer"
            uploaded_eval = st.file_uploader(
                "Optional: Inspect External Evaluation CSV (true_label, predicted_label)",
                type=["csv"],
            )
            if uploaded_eval is not None:
                try:
                    ext_df = pd.read_csv(uploaded_eval)
                    ext_summary = evaluate_held_out_predictions(ext_df)
                    st.session_state["last_eval_summary"] = ext_summary
                    runs[run_label] = ext_summary
                    st.success(f"Loaded custom evaluation run '{run_label}'.")
                except Exception as exc:
                    st.error(f"Invalid evaluation CSV format: {exc}")

            uploaded_history = st.file_uploader(
                "Optional: Inspect External Training History CSV",
                type=["csv"],
            )
            if uploaded_history is not None:
                try:
                    st.session_state["custom_history_df"] = pd.read_csv(uploaded_history)
                    st.success("Loaded custom training history CSV.")
                except Exception as exc:
                    st.error(f"Unable to read training history CSV: {exc}")

    elif page == "Dataset Analytics":
        st.header("Dataset Analytics")
        st.caption(
            "Automatically inspects the configured default dataset: usable records, class balance, splits, text lengths, missingness, and duplicates."
        )

        active_df: Optional[pd.DataFrame] = st.session_state.get("custom_dataset_df")
        active_summary: Optional[DatasetSummary] = st.session_state.get("custom_dataset_summary")
        active_tcol: str = st.session_state.get("custom_dataset_tcol", "text")
        active_lcol: str = st.session_state.get("custom_dataset_lcol", "label")

        if active_summary is None and default_dataset_bundle is not None:
            _, active_df, active_summary, active_tcol, active_lcol = default_dataset_bundle

        if active_summary is None or active_df is None:
            st.warning(
                "No default dataset was found at `data/default_dataset.csv`. Place a labeled CSV "
                "in `data/` or expand Advanced Options below to inspect a custom file."
            )
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Records", active_summary.total_rows)
            m2.metric("Usable Records", active_summary.valid_rows)
            m3.metric("Classes", active_summary.num_classes)
            m4.metric("Duplicates", active_summary.duplicate_texts)
            m5.metric("Imbalance Ratio", f"{active_summary.imbalance_ratio:.2f}x")

            st.caption(
                f"Dataset: **{active_summary.dataset_name}** · Source: {active_summary.dataset_source} · "
                f"Missing text: {active_summary.missing_text_rows} · Missing label: {active_summary.missing_label_rows}"
            )

            st.plotly_chart(plot_class_distribution(active_summary), use_container_width=True)
            st.caption(
                "What this shows: Number of usable posts per label. Large differences between tallest and shortest bars "
                f"(currently {active_summary.imbalance_ratio:.2f}x between '{active_summary.majority_class}' and '{active_summary.minority_class}') "
                "indicate class imbalance handled during training via inverse-frequency class weights."
            )

            st.plotly_chart(plot_split_distribution(active_summary), use_container_width=True)
            st.caption(
                "What this shows: How records are divided across training (model learning), validation (early stopping), "
                "and held-out testing (unbiased final evaluation)."
            )

            st.plotly_chart(
                plot_word_length_histogram(active_df, active_tcol, active_lcol),
                use_container_width=True,
            )
            st.caption(
                "What this shows: Distribution of post lengths in words. Longer posts carry richer context, while posts "
                "exceeding the Transformer's 256-token window are truncated at the tail."
            )

        with st.expander("Advanced Options (Inspect Custom Dataset CSV)"):
            if "custom_dataset_summary" in st.session_state:
                if st.button("Restore Default Dataset"):
                    for k in (
                        "custom_dataset_df",
                        "custom_dataset_summary",
                        "custom_dataset_tcol",
                        "custom_dataset_lcol",
                    ):
                        st.session_state.pop(k, None)
                    st.rerun()

            uploaded_corpus = st.file_uploader("Optional: Upload Custom Labeled CSV", type=["csv"])
            if uploaded_corpus is not None:
                try:
                    corpus_df = pd.read_csv(uploaded_corpus)
                    if corpus_df.empty or len(corpus_df.columns) < 2:
                        raise ValueError("CSV must contain at least 2 columns and 1 data row.")
                    cols = list(corpus_df.columns)
                    col_a, col_b, col_c = st.columns(3)
                    t_col = col_a.selectbox(
                        "Text Column", cols, index=cols.index("text") if "text" in cols else 0
                    )
                    l_col = col_b.selectbox(
                        "Label Column",
                        cols,
                        index=cols.index("label") if "label" in cols else min(1, len(cols) - 1),
                    )
                    split_options = ["None"] + cols
                    split_choice = col_c.selectbox(
                        "Split Column (optional)",
                        split_options,
                        index=split_options.index("split") if "split" in split_options else 0,
                    )
                    d_name = st.text_input("Dataset Name", value=uploaded_corpus.name.rsplit(".", 1)[0])
                    d_source = st.text_input("Dataset Source", value="Custom local research CSV")

                    custom_summary = analyze_corpus(
                        corpus_df,
                        text_col=t_col,
                        label_col=l_col,
                        dataset_name=d_name,
                        dataset_source=d_source,
                        split_col=None if split_choice == "None" else split_choice,
                    )
                    st.session_state["custom_dataset_df"] = corpus_df
                    st.session_state["custom_dataset_summary"] = custom_summary
                    st.session_state["custom_dataset_tcol"] = t_col
                    st.session_state["custom_dataset_lcol"] = l_col
                    st.success(f"Switched analytics view to '{d_name}'.")
                except Exception as exc:
                    st.error(f"Could not analyze custom CSV: {exc}")

    elif page == "About & Methodology":
        st.header("About & Methodology")
        st.caption(
            "Formal academic documentation, Transformer and XAI methodology, technologies used, ethical governance, and references."
        )
        st.info(f"{RESEARCH_DISCLAIMER}\n\n{PRIVACY_STATEMENT}")
        for sec in ACADEMIC_METHODOLOGY_SECTIONS:
            st.subheader(f"{sec['number']}. {sec['title']}")
            st.write(sec["body"])

        st.subheader("Technologies Used & Transformer Backbones")
        st.markdown(
            "- **Deep Learning & NLP**: PyTorch, Hugging Face Transformers, Tokenizers\n"
            "- **Explainable AI**: SHAP (Partition Text Explainer), LIME Text Explainer\n"
            "- **Data & Evaluation**: Pandas, NumPy, Scikit-learn, Plotly\n"
            "- **Dashboard & API**: Streamlit, FastAPI"
        )
        for name, hf_id in SUPPORTED_BACKBONES.items():
            st.markdown(f"- **{name}**: `{hf_id}`")

        st.subheader("Dataset Bias & Technical Limitations")
        for item in DATASET_LIMITATIONS:
            st.markdown(f"- **{item['title']}**: {item['detail']}")

        st.subheader("15. References")
        for ref in ACADEMIC_REFERENCES:
            st.markdown(f"- {ref}")

        st.subheader("Supportive Crisis Resources")
        for res in CRISIS_RESOURCES:
            st.markdown(f"- **{res['region']} — {res['service']}**: {res['contact']}")


if __name__ == "__main__":
    main()
