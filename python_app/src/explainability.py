"""Genuine post-hoc Explainable AI (SHAP and LIME) with subword aggregation and span alignment."""

from dataclasses import dataclass
from html import escape
import re
from typing import Dict, List, Literal, Optional
import numpy as np
import plotly.graph_objects as go
import shap
from lime.lime_text import LimeTextExplainer

from src.config import ExplainabilityConfig
from src.ethics import XAI_EVIDENCE_BOUNDARY
from src.model_inference import MentalHealthTransformerClassifier, PredictionResult


ExplanationMethod = Literal["SHAP", "LIME"]

_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "<pad>", "<unk>"}


@dataclass
class TokenAttribution:
    """Contribution of an aggregated word or phrase aligned to text character offsets."""

    token: str
    weight: float
    direction: Literal["supporting", "opposing", "neutral"]
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class ExplanationReport:
    """Complete explainability output for one text sample."""

    method: ExplanationMethod
    original_text: str
    analyzed_text: str
    target_label: str
    confidence: float
    attributions: List[TokenAttribution]
    natural_language_summary: str
    highlighted_html: str
    evidence_boundary_note: str


def _classify_direction(weight: float, eps: float = 1e-4) -> Literal["supporting", "opposing", "neutral"]:
    if weight > eps:
        return "supporting"
    if weight < -eps:
        return "opposing"
    return "neutral"


def _aggregate_subword_attributions(
    raw_tokens: List[str],
    raw_values: np.ndarray,
    text: str,
) -> List[TokenAttribution]:
    """Align tokenizer/SHAP pieces to text offsets and sum WordPiece subwords into whole words.

    Because Shapley values are additive across subword coalitions, summing the weights
    of WordPiece fragments belonging to a single word yields an exact word-level score.
    """
    pieces: List[Dict[str, object]] = []
    cursor = 0
    lowered_text = text.lower()

    for tok, val in zip(raw_tokens, raw_values):
        raw_str = str(tok)
        stripped = raw_str.strip()
        if not stripped or stripped in _SPECIAL_TOKENS:
            continue

        is_wordpiece_cont = stripped.startswith("##")
        clean_piece = stripped[2:] if is_wordpiece_cont else stripped
        if not clean_piece:
            continue

        # Locate piece sequentially in text to preserve exact character spans
        idx = text.find(clean_piece, cursor)
        if idx == -1:
            idx = lowered_text.find(clean_piece.lower(), cursor)

        if idx != -1:
            start = idx
            end = idx + len(clean_piece)
            cursor = end
            surface = text[start:end]
        else:
            start = None
            end = None
            surface = clean_piece

        pieces.append(
            {
                "surface": surface,
                "weight": float(val),
                "start": start,
                "end": end,
                "is_cont": is_wordpiece_cont,
            }
        )

    merged: List[TokenAttribution] = []
    current_surface = ""
    current_weight = 0.0
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    def flush() -> None:
        nonlocal current_surface, current_weight, current_start, current_end
        if current_surface:
            merged.append(
                TokenAttribution(
                    token=current_surface,
                    weight=current_weight,
                    direction=_classify_direction(current_weight),
                    start=current_start,
                    end=current_end,
                )
            )
        current_surface = ""
        current_weight = 0.0
        current_start = None
        current_end = None

    for p in pieces:
        p_surface = str(p["surface"])
        p_weight = float(p["weight"])  # type: ignore[arg-type]
        p_start = p["start"]  # type: ignore[assignment]
        p_end = p["end"]  # type: ignore[assignment]
        p_cont = bool(p["is_cont"])

        contiguous_word_join = (
            current_surface
            and (
                p_cont
                or (
                    current_end is not None
                    and p_start is not None
                    and current_end == p_start
                    and p_surface.isalnum()
                    and current_surface[-1].isalnum()
                )
            )
        )

        if contiguous_word_join:
            current_surface += p_surface
            current_weight += p_weight
            if p_end is not None:
                current_end = p_end
        else:
            flush()
            current_surface = p_surface
            current_weight = p_weight
            current_start = p_start
            current_end = p_end

    flush()
    return merged


def build_grounded_summary(
    method: ExplanationMethod,
    target_label: str,
    confidence: float,
    attributions: List[TokenAttribution],
) -> str:
    """Generate a plain-language explanation strictly from computed attribution weights."""
    ranked = sorted(attributions, key=lambda a: abs(a.weight), reverse=True)
    supporting = [a.token for a in ranked if a.weight > 1e-4][:4]
    opposing = [a.token for a in ranked if a.weight < -1e-4][:4]

    sup_str = ", ".join(f"'{tok}'" for tok in supporting) if supporting else "none above threshold"
    opp_str = ", ".join(f"'{tok}'" for tok in opposing) if opposing else "none above threshold"

    return (
        f"Using {method}, the Transformer model predicted '{target_label}' with "
        f"{confidence * 100:.1f}% softmax confidence. The words that pushed the model's score "
        f"most strongly toward '{target_label}' were {sup_str}. The words that pushed the score "
        f"away from '{target_label}' were {opp_str}. Important: this feature attribution shows "
        f"how the model reacted to linguistic patterns in the text; it does not establish the "
        f"actual psychological reasons behind a person's writing."
    )


def render_highlighted_html(text: str, attributions: List[TokenAttribution]) -> str:
    """Highlight influential words directly inside the analyzed text using character offsets."""
    if not text:
        return ""
    max_abs = max((abs(a.weight) for a in attributions), default=1.0) or 1.0
    span_attrs = sorted(
        [a for a in attributions if a.start is not None and a.end is not None],
        key=lambda a: int(a.start),  # type: ignore[arg-type]
    )

    if not span_attrs:
        return escape(text)

    parts: List[str] = []
    cursor = 0
    for attr in span_attrs:
        start = int(attr.start)  # type: ignore[arg-type]
        end = int(attr.end)  # type: ignore[arg-type]
        if start < cursor or end > len(text):
            continue
        if start > cursor:
            parts.append(escape(text[cursor:start]))

        slice_text = text[start:end]
        if attr.direction == "neutral":
            parts.append(escape(slice_text))
        else:
            norm = min(abs(attr.weight) / max_abs, 1.0)
            alpha = 0.12 + 0.36 * norm
            if attr.direction == "supporting":
                bg = f"rgba(13, 115, 119, {alpha:.2f})"
                border = "#0D7377"
            else:
                bg = f"rgba(180, 83, 9, {alpha:.2f})"
                border = "#B45309"
            title = escape(f"{slice_text}: {attr.weight:+.4f} ({attr.direction})")
            parts.append(
                f'<span title="{title}" style="background:{bg};border-bottom:2px solid {border};'
                f'padding:1px 4px;border-radius:4px;">{escape(slice_text)}</span>'
            )
        cursor = end

    if cursor < len(text):
        parts.append(escape(text[cursor:]))

    return "".join(parts)


def plot_token_importance(report: ExplanationReport, top_k: int = 12) -> go.Figure:
    """Create a diverging horizontal Plotly bar chart of influential words."""
    non_zero = [a for a in report.attributions if abs(a.weight) > 1e-5]
    pool = non_zero if non_zero else report.attributions
    top_items = sorted(pool, key=lambda a: abs(a.weight), reverse=True)[:top_k]
    ordered = list(reversed(top_items))
    tokens = [a.token for a in ordered]
    weights = [a.weight for a in ordered]
    colors = ["#0D7377" if w >= 0 else "#B45309" for w in weights]

    fig = go.Figure(
        go.Bar(
            x=weights,
            y=tokens,
            orientation="h",
            marker_color=colors,
            hovertemplate="Word: %{y}<br>Contribution: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{report.method} Word Contributions for '{report.target_label}'",
        xaxis_title="Attribution Weight (+ pushes toward class, - pushes away)",
        yaxis_title="Word / Feature",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        margin=dict(l=20, r=20, t=50, b=40),
        height=max(320, 32 * len(ordered) + 90),
    )
    fig.add_vline(x=0, line_width=1, line_color="#94A3B8")
    return fig


class ExplainabilityEngine:
    """Computes genuine SHAP and LIME explanations directly connected to the Transformer model."""

    def __init__(
        self,
        classifier: MentalHealthTransformerClassifier,
        config: ExplainabilityConfig = ExplainabilityConfig(),
    ) -> None:
        self.classifier = classifier
        self.config = config
        self.class_names = classifier.class_names
        self._lime = LimeTextExplainer(class_names=self.class_names)
        self._shap_masker = shap.maskers.Text(classifier.tokenizer)
        self._shap_explainer = shap.Explainer(
            classifier.predict_proba,
            self._shap_masker,
            output_names=self.class_names,
        )

    def explain_with_shap(
        self,
        cleaned_text: str,
        prediction: PredictionResult,
        original_text: Optional[str] = None,
    ) -> ExplanationReport:
        """Compute Partition SHAP attributions and aggregate WordPiece subwords to whole words."""
        target_idx = self.class_names.index(prediction.predicted_label)
        shap_values = self._shap_explainer([cleaned_text], max_evals=self.config.shap_max_evals)

        tokens = list(shap_values.data[0])
        values = np.asarray(shap_values.values[0, :, target_idx], dtype=float)
        attributions = _aggregate_subword_attributions(tokens, values, cleaned_text)

        summary = build_grounded_summary(
            "SHAP", prediction.predicted_label, prediction.confidence, attributions
        )
        return ExplanationReport(
            method="SHAP",
            original_text=original_text or cleaned_text,
            analyzed_text=cleaned_text,
            target_label=prediction.predicted_label,
            confidence=prediction.confidence,
            attributions=attributions,
            natural_language_summary=summary,
            highlighted_html=render_highlighted_html(cleaned_text, attributions),
            evidence_boundary_note=XAI_EVIDENCE_BOUNDARY,
        )

    def explain_with_lime(
        self,
        cleaned_text: str,
        prediction: PredictionResult,
        original_text: Optional[str] = None,
    ) -> ExplanationReport:
        """Compute local surrogate LIME attributions aligned directly to word offsets."""
        target_idx = self.class_names.index(prediction.predicted_label)
        lime_exp = self._lime.explain_instance(
            cleaned_text,
            self.classifier.predict_proba,
            labels=(target_idx,),
            num_features=self.config.lime_num_features,
            num_samples=self.config.lime_num_samples,
        )

        feature_weights: Dict[str, float] = dict(lime_exp.as_list(label=target_idx))
        lowered_weights = {k.lower(): v for k, v in feature_weights.items()}

        attributions: List[TokenAttribution] = []
        for match in re.finditer(r"[A-Za-z0-9_'\[\]]+", cleaned_text):
            word = match.group(0)
            weight = feature_weights.get(
                word, lowered_weights.get(word.lower(), 0.0)
            )
            attributions.append(
                TokenAttribution(
                    token=word,
                    weight=float(weight),
                    direction=_classify_direction(float(weight)),
                    start=match.start(),
                    end=match.end(),
                )
            )

        summary = build_grounded_summary(
            "LIME", prediction.predicted_label, prediction.confidence, attributions
        )
        return ExplanationReport(
            method="LIME",
            original_text=original_text or cleaned_text,
            analyzed_text=cleaned_text,
            target_label=prediction.predicted_label,
            confidence=prediction.confidence,
            attributions=attributions,
            natural_language_summary=summary,
            highlighted_html=render_highlighted_html(cleaned_text, attributions),
            evidence_boundary_note=XAI_EVIDENCE_BOUNDARY,
        )

    def explain(
        self,
        cleaned_text: str,
        prediction: PredictionResult,
        method: ExplanationMethod = "SHAP",
        original_text: Optional[str] = None,
    ) -> ExplanationReport:
        if method == "LIME":
            return self.explain_with_lime(cleaned_text, prediction, original_text=original_text)
        return self.explain_with_shap(cleaned_text, prediction, original_text=original_text)
