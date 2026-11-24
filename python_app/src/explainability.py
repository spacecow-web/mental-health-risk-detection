"""Genuine post-hoc Explainable AI (SHAP and LIME) with subword aggregation, stopword filtering, and phrase alignment."""

from dataclasses import dataclass, field
from html import escape
import math
import re
from typing import Dict, List, Literal, Optional, Set
import numpy as np
import plotly.graph_objects as go
import shap
from lime.lime_text import LimeTextExplainer

from src.config import ExplainabilityConfig
from src.ethics import XAI_EVIDENCE_BOUNDARY
from src.model_inference import MentalHealthTransformerClassifier, PredictionResult


ExplanationMethod = Literal["SHAP", "LIME"]
Direction = Literal["supporting", "opposing", "neutral"]

_SPECIAL_TOKENS: Set[str] = {
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[UNK]",
    "[MASK]",
    "[URL]",
    "[USER]",
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
}

# Negation words, negative contractions, and intensity modifiers that directly alter
# emotional polarity and must NEVER be discarded by stopword filtering.
PROTECTED_NEGATION_AND_MODIFIER_WORDS: Set[str] = {
    "not",
    "no",
    "never",
    "nobody",
    "nothing",
    "neither",
    "nor",
    "nowhere",
    "none",
    "hardly",
    "barely",
    "scarcely",
    "without",
    "lack",
    "lacking",
    "lacks",
    "unable",
    "cannot",
    "can't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "mustn't",
    "needn't",
    "shan't",
    "mightn't",
    "ain't",
    "nothing's",
    "nobody's",
    "anymore",
    "very",
    "too",
    "completely",
    "totally",
    "always",
    "everything",
    "everyone",
    "alone",
    "only",
}

# Common grammatical function words that do not carry standalone emotional meaning.
# Kept intact in text passed to Transformer models, and filtered only at the
# explanation / highlighting stage unless part of an active multi-word phrase.
GRAMMATICAL_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "while",
    "where",
    "because",
    "so",
    "than",
    "as",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "for",
    "with",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "once",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    "although",
    "though",
    "since",
    "until",
    "unless",
    "whether",
    "whereas",
    "despite",
    "yet",
    "either",
    "every",
    "ever",
    "why",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "there",
    "here",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "own",
    "same",
    "just",
    "also",
    "even",
    "still",
    "already",
    "really",
    "actually",
    "basically",
    "literally",
    "get",
    "gets",
    "got",
    "getting",
    "feel",
    "feels",
    "feeling",
    "felt",
    "seem",
    "seems",
    "seemed",
    "keep",
    "keeps",
    "kept",
}

_CONTRACTION_STOPWORDS: Set[str] = {
    "i'm",
    "it's",
    "that's",
    "there's",
    "we're",
    "they're",
    "you're",
    "he's",
    "she's",
}

# Safe connectors allowed inside a coherent multi-word phrase; clause conjunctions
# ("and", "because", "but", "or") are excluded so unrelated clauses stay separate.
INTRA_PHRASE_BRIDGES: Set[str] = {
    "to",
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "a",
    "an",
    "the",
    "my",
    "me",
    "myself",
    "so",
    "get",
    "gets",
    "getting",
    "got",
    "feel",
    "feels",
    "feeling",
    "felt",
    "seem",
    "seems",
    "seemed",
    "keep",
    "keeps",
    "kept",
}

_NEGATIVE_CONTRACTION_RE = re.compile(r"^[a-z]+n't$")
_WORD_SPAN_RE = re.compile(r"\[URL\]|\[USER\]|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
_CLAUSE_PUNCT_RE = re.compile(r"[.,!?;:\n—–]")


@dataclass
class TokenAttribution:
    """Contribution of an aggregated word aligned to text character offsets."""

    token: str
    weight: float
    direction: Direction
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class PhraseAttribution:
    """Coherent meaningful phrase built from salient words and safe intra-phrase connectors."""

    phrase: str
    weight: float
    direction: Direction
    start: int
    end: int
    tokens: List[str] = field(default_factory=list)


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
    phrase_attributions: List[PhraseAttribution] = field(default_factory=list)


def normalize_word_for_filter(token: str) -> str:
    clean = token.replace("##", "") if token.startswith("##") else token
    return clean.replace("’", "'").strip().lower()


def is_protected_negation_or_cue(token: str) -> bool:
    norm = normalize_word_for_filter(token)
    if not norm:
        return False
    if norm in PROTECTED_NEGATION_AND_MODIFIER_WORDS:
        return True
    return bool(_NEGATIVE_CONTRACTION_RE.match(norm))


def is_standalone_stopword_or_punctuation(token: str) -> bool:
    cleaned = token[2:] if token.startswith("##") else token
    cleaned = cleaned.strip()
    if not cleaned:
        return True
    if cleaned in _SPECIAL_TOKENS or cleaned.upper() in _SPECIAL_TOKENS:
        return True
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return True

    norm = normalize_word_for_filter(cleaned)
    if is_protected_negation_or_cue(norm):
        return False
    if norm in _CONTRACTION_STOPWORDS:
        return True
    return norm in GRAMMATICAL_STOPWORDS


def compute_adaptive_threshold(weights: List[float]) -> float:
    """Distribution-based salience threshold derived from median, MAD, and peak magnitude."""
    abs_vals = [abs(w) for w in weights if abs(w) > 1e-6]
    if not abs_vals:
        return math.inf

    max_abs = max(abs_vals)
    floor = 0.006
    if max_abs < floor:
        return floor

    arr = np.asarray(abs_vals, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    candidate = max(floor, 0.16 * max_abs, median + 0.35 * mad)
    return min(candidate, max(floor, 0.52 * max_abs))


def _classify_direction(weight: float, threshold: float) -> Direction:
    if weight >= threshold:
        return "supporting"
    if weight <= -threshold:
        return "opposing"
    return "neutral"


def _aggregate_subword_attributions(
    raw_tokens: List[str],
    raw_values: np.ndarray,
    text: str,
) -> List[TokenAttribution]:
    """Align tokenizer/SHAP pieces to text offsets, merge subwords/contractions, and filter stopwords.

    Because Shapley values are additive across subword coalitions, summing the weights
    of WordPiece fragments belonging to a single word (or contraction) yields an exact
    word-level score without dropping context from the Transformer input.
    """
    pieces: List[Dict[str, object]] = []
    cursor = 0
    lowered_text = text.lower()

    for tok, val in zip(raw_tokens, raw_values):
        raw_str = str(tok)
        stripped = raw_str.strip()
        if not stripped or stripped in _SPECIAL_TOKENS or stripped.upper() in _SPECIAL_TOKENS:
            continue

        is_wordpiece_cont = stripped.startswith("##")
        clean_piece = stripped[2:] if is_wordpiece_cont else stripped
        if not clean_piece:
            continue

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
                    direction="neutral",
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

        touches_prev = (
            current_surface
            and current_end is not None
            and p_start is not None
            and current_end == p_start
        )
        is_contraction_join = bool(
            touches_prev
            and (
                p_surface in {"'", "’"}
                or re.match(r"^['’][A-Za-z]+$", p_surface)
                or ((current_surface.endswith("'") or current_surface.endswith("’")) and p_surface.isalnum())
            )
        )
        contiguous_word_join = bool(
            current_surface
            and (
                p_cont
                or is_contraction_join
                or (
                    touches_prev
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

    meaningful = [a for a in merged if not is_standalone_stopword_or_punctuation(a.token)]
    threshold = compute_adaptive_threshold([a.weight for a in meaningful])
    return [
        TokenAttribution(
            token=a.token,
            weight=a.weight,
            direction=_classify_direction(a.weight, threshold),
            start=a.start,
            end=a.end,
        )
        for a in meaningful
    ]


def group_meaningful_phrases(
    text: str,
    attributions: List[TokenAttribution],
) -> List[PhraseAttribution]:
    """Group adjacent salient words and safe intra-phrase connectors into readable phrases."""
    if not text.strip() or not attributions:
        return []

    meaningful = [a for a in attributions if not is_standalone_stopword_or_punctuation(a.token)]
    if not meaningful:
        return []

    threshold = compute_adaptive_threshold([a.weight for a in meaningful])
    normalized = [
        TokenAttribution(
            token=a.token,
            weight=a.weight,
            direction=_classify_direction(a.weight, threshold),
            start=a.start,
            end=a.end,
        )
        for a in meaningful
    ]

    positions: List[Dict[str, object]] = [
        {"word": m.group(0), "start": m.start(), "end": m.end(), "attr": None}
        for m in _WORD_SPAN_RE.finditer(text)
    ]
    if not positions:
        return []

    seq_cursor = 0
    for attr in normalized:
        matched = False
        if attr.start is not None and attr.end is not None:
            for pos in positions:
                if abs(int(pos["start"]) - attr.start) <= 1 and abs(int(pos["end"]) - attr.end) <= 1:  # type: ignore[arg-type]
                    pos["attr"] = attr
                    matched = True
                    break
        if matched:
            continue
        clean_tok = attr.token.lower()
        for idx in range(seq_cursor, len(positions)):
            if str(positions[idx]["word"]).lower() == clean_tok and positions[idx]["attr"] is None:
                positions[idx]["attr"] = attr
                seq_cursor = idx + 1
                break

    phrases: List[PhraseAttribution] = []
    used_indices: Set[int] = set()

    for i, start_pos in enumerate(positions):
        if i in used_indices:
            continue
        start_attr: Optional[TokenAttribution] = start_pos["attr"]  # type: ignore[assignment]
        if start_attr is None or start_attr.direction == "neutral":
            continue

        group_indices: List[int] = [i]
        start_dir = start_attr.direction
        starts_with_negation = is_protected_negation_or_cue(str(start_pos["word"]))
        last_salient_idx = i
        bridge_run = 0

        for j in range(i + 1, len(positions)):
            prev_pos = positions[j - 1]
            curr_pos = positions[j]
            gap = text[int(prev_pos["end"]) : int(curr_pos["start"])]  # type: ignore[arg-type]
            if _CLAUSE_PUNCT_RE.search(gap):
                break

            norm_curr = normalize_word_for_filter(str(curr_pos["word"]))
            curr_attr: Optional[TokenAttribution] = curr_pos["attr"]  # type: ignore[assignment]

            if curr_attr is not None and curr_attr.direction != "neutral":
                compatible = curr_attr.direction == start_dir or (
                    starts_with_negation and j - last_salient_idx <= 5
                )
                if compatible:
                    for k in range(last_salient_idx + 1, j + 1):
                        group_indices.append(k)
                    last_salient_idx = j
                    bridge_run = 0
                    continue
                break

            is_bridge = norm_curr in INTRA_PHRASE_BRIDGES or (
                starts_with_negation and norm_curr not in GRAMMATICAL_STOPWORDS
            )
            if is_bridge and bridge_run < 4:
                bridge_run += 1
                continue
            break

        final_indices = [idx for idx in group_indices if idx <= last_salient_idx]
        used_indices.update(final_indices)

        first_pos = positions[final_indices[0]]
        last_pos = positions[final_indices[-1]]
        p_start = int(first_pos["start"])  # type: ignore[arg-type]
        p_end = int(last_pos["end"])  # type: ignore[arg-type]
        phrase_text = text[p_start:p_end].strip()

        total_weight = sum(
            positions[idx]["attr"].weight  # type: ignore[union-attr]
            for idx in final_indices
            if positions[idx]["attr"] is not None
        )
        if abs(total_weight) >= threshold:
            direction = _classify_direction(total_weight, threshold)
        elif starts_with_negation and len(final_indices) > 1:
            direction = start_dir
        else:
            direction = "neutral"

        if direction != "neutral" and phrase_text:
            phrases.append(
                PhraseAttribution(
                    phrase=phrase_text,
                    weight=float(total_weight),
                    direction=direction,
                    start=p_start,
                    end=p_end,
                    tokens=[str(positions[idx]["word"]) for idx in final_indices],
                )
            )

    phrases.sort(key=lambda p: abs(p.weight), reverse=True)
    return phrases


def build_grounded_summary(
    method: ExplanationMethod,
    target_label: str,
    confidence: float,
    attributions: List[TokenAttribution],
    phrase_attributions: Optional[List[PhraseAttribution]] = None,
) -> str:
    """Generate a plain-language explanation strictly from filtered, significant attribution weights."""
    phrases = phrase_attributions or []
    phrase_sup = [f"'{p.phrase}'" for p in phrases if p.direction == "supporting"][:3]
    phrase_opp = [f"'{p.phrase}'" for p in phrases if p.direction == "opposing"][:3]

    ranked = sorted(attributions, key=lambda a: abs(a.weight), reverse=True)
    word_sup = [f"'{a.token}'" for a in ranked if a.direction == "supporting"][:4]
    word_opp = [f"'{a.token}'" for a in ranked if a.direction == "opposing"][:4]

    sup = phrase_sup if phrase_sup else word_sup
    opp = phrase_opp if phrase_opp else word_opp

    sup_str = ", ".join(sup) if sup else "none above significance threshold"
    opp_str = ", ".join(opp) if opp else "none above significance threshold"

    return (
        f"Using {method}, the Transformer model predicted '{target_label}' with "
        f"{confidence * 100:.1f}% softmax confidence. The meaningful words and phrases that pushed "
        f"the model's score most strongly toward '{target_label}' were {sup_str}. The meaningful "
        f"words and phrases that pushed the score away from '{target_label}' were {opp_str}. "
        f"Important: this feature attribution shows how the model reacted to linguistic patterns "
        f"in the text; it does not establish the actual psychological reasons behind a person's writing."
    )


def render_highlighted_html(
    text: str,
    attributions: List[TokenAttribution],
    phrase_attributions: Optional[List[PhraseAttribution]] = None,
) -> str:
    """Highlight significant non-stopword expressions directly inside the analyzed text."""
    if not text:
        return ""

    spans: List[Dict[str, object]] = []
    if phrase_attributions:
        for p in phrase_attributions:
            if p.direction != "neutral":
                spans.append(
                    {
                        "label": p.phrase,
                        "weight": p.weight,
                        "direction": p.direction,
                        "start": p.start,
                        "end": p.end,
                    }
                )
    if not spans:
        for a in attributions:
            if a.direction != "neutral" and a.start is not None and a.end is not None:
                spans.append(
                    {
                        "label": a.token,
                        "weight": a.weight,
                        "direction": a.direction,
                        "start": a.start,
                        "end": a.end,
                    }
                )

    if not spans:
        return escape(text)

    spans.sort(key=lambda s: int(s["start"]))  # type: ignore[arg-type]
    max_abs = max((abs(float(s["weight"])) for s in spans), default=1.0) or 1.0  # type: ignore[arg-type]

    parts: List[str] = []
    cursor = 0
    for sp in spans:
        start = int(sp["start"])  # type: ignore[arg-type]
        end = int(sp["end"])  # type: ignore[arg-type]
        weight = float(sp["weight"])  # type: ignore[arg-type]
        direction = str(sp["direction"])

        if start < cursor or end > len(text) or start >= end:
            continue
        if start > cursor:
            parts.append(escape(text[cursor:start]))

        slice_text = text[start:end]
        norm = min(abs(weight) / max_abs, 1.0)
        alpha = 0.14 + 0.36 * norm
        if direction == "supporting":
            bg = f"rgba(13, 115, 119, {alpha:.2f})"
            border = "#0D7377"
        else:
            bg = f"rgba(180, 83, 9, {alpha:.2f})"
            border = "#B45309"
        title = escape(f"{slice_text}: {weight:+.4f} ({direction})")
        parts.append(
            f'<span title="{title}" style="background:{bg};border-bottom:2px solid {border};'
            f'padding:1px 4px;border-radius:4px;">{escape(slice_text)}</span>'
        )
        cursor = end

    if cursor < len(text):
        parts.append(escape(text[cursor:]))

    return "".join(parts)


def plot_token_importance(report: ExplanationReport, top_k: int = 12) -> go.Figure:
    """Create a diverging horizontal Plotly bar chart of significant words/phrases."""
    if report.phrase_attributions:
        pool = [
            (p.phrase, p.weight)
            for p in report.phrase_attributions
            if p.direction != "neutral"
        ]
    else:
        pool = [
            (a.token, a.weight)
            for a in report.attributions
            if a.direction != "neutral"
        ]

    top_items = sorted(pool, key=lambda item: abs(item[1]), reverse=True)[:top_k]
    ordered = list(reversed(top_items))
    tokens = [item[0] for item in ordered]
    weights = [item[1] for item in ordered]
    colors = ["#0D7377" if w >= 0 else "#B45309" for w in weights]

    fig = go.Figure(
        go.Bar(
            x=weights,
            y=tokens,
            orientation="h",
            marker_color=colors,
            hovertemplate="Expression: %{y}<br>Contribution: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{report.method} Meaningful Word & Phrase Contributions for '{report.target_label}'",
        xaxis_title="Attribution Weight (+ pushes toward class, - pushes away)",
        yaxis_title="Expression / Feature",
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
        # bow=False preserves word order and negation scope while keeping contractions intact
        self._lime = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=r"[^\w'’\[\]]+",
            bow=False,
        )
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
        """Compute Partition SHAP attributions, aggregate subwords/contractions, and group phrases."""
        target_idx = self.class_names.index(prediction.predicted_label)
        shap_values = self._shap_explainer([cleaned_text], max_evals=self.config.shap_max_evals)

        tokens = list(shap_values.data[0])
        values = np.asarray(shap_values.values[0, :, target_idx], dtype=float)
        attributions = _aggregate_subword_attributions(tokens, values, cleaned_text)
        phrase_attributions = group_meaningful_phrases(cleaned_text, attributions)

        summary = build_grounded_summary(
            "SHAP",
            prediction.predicted_label,
            prediction.confidence,
            attributions,
            phrase_attributions=phrase_attributions,
        )
        return ExplanationReport(
            method="SHAP",
            original_text=original_text or cleaned_text,
            analyzed_text=cleaned_text,
            target_label=prediction.predicted_label,
            confidence=prediction.confidence,
            attributions=attributions,
            natural_language_summary=summary,
            highlighted_html=render_highlighted_html(
                cleaned_text, attributions, phrase_attributions=phrase_attributions
            ),
            evidence_boundary_note=XAI_EVIDENCE_BOUNDARY,
            phrase_attributions=phrase_attributions,
        )

    def explain_with_lime(
        self,
        cleaned_text: str,
        prediction: PredictionResult,
        original_text: Optional[str] = None,
    ) -> ExplanationReport:
        """Compute local surrogate LIME attributions filtered for meaningful expressions."""
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

        raw_word_items: List[TokenAttribution] = []
        for match in _WORD_SPAN_RE.finditer(cleaned_text):
            word = match.group(0)
            if is_standalone_stopword_or_punctuation(word):
                continue
            weight = float(
                feature_weights.get(word, lowered_weights.get(word.lower(), 0.0))
            )
            raw_word_items.append(
                TokenAttribution(
                    token=word,
                    weight=weight,
                    direction="neutral",
                    start=match.start(),
                    end=match.end(),
                )
            )

        threshold = compute_adaptive_threshold([a.weight for a in raw_word_items])
        attributions = [
            TokenAttribution(
                token=a.token,
                weight=a.weight,
                direction=_classify_direction(a.weight, threshold),
                start=a.start,
                end=a.end,
            )
            for a in raw_word_items
        ]
        phrase_attributions = group_meaningful_phrases(cleaned_text, attributions)

        summary = build_grounded_summary(
            "LIME",
            prediction.predicted_label,
            prediction.confidence,
            attributions,
            phrase_attributions=phrase_attributions,
        )
        return ExplanationReport(
            method="LIME",
            original_text=original_text or cleaned_text,
            analyzed_text=cleaned_text,
            target_label=prediction.predicted_label,
            confidence=prediction.confidence,
            attributions=attributions,
            natural_language_summary=summary,
            highlighted_html=render_highlighted_html(
                cleaned_text, attributions, phrase_attributions=phrase_attributions
            ),
            evidence_boundary_note=XAI_EVIDENCE_BOUNDARY,
            phrase_attributions=phrase_attributions,
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
