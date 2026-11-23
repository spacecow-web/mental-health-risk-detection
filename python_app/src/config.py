"""Central configuration dataclasses for MindTrace AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "model"
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_EVAL_CSV = DEFAULT_DATA_DIR / "held_out_predictions.csv"


SUPPORTED_BACKBONES: Dict[str, str] = {
    "DistilBERT (Recommended)": "distilbert-base-uncased",
    "BERT Base Uncased": "bert-base-uncased",
}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration controlling deterministic social media text cleaning."""

    mask_urls: bool = True
    anonymize_mentions: bool = True
    strip_hashtag_symbol: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False
    min_chars: int = 10
    max_chars: int = 2000
    max_seq_length: int = 256


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for Transformer checkpoint loading and inference."""

    checkpoint_dir: Path = DEFAULT_ARTIFACTS_DIR
    backbone_name: str = "distilbert-base-uncased"
    max_seq_length: int = 256
    high_uncertainty_entropy_threshold: float = 0.72


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Settings for post-hoc SHAP and LIME token attribution."""

    default_method: str = "SHAP"
    lime_num_features: int = 12
    lime_num_samples: int = 300
    shap_max_evals: int = 250
    top_k_display: int = 12


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    app_title: str = "MindTrace AI"
    app_subtitle: str = (
        "Explainable Mental Health Risk Detection from Social Media Text "
        "Using Transformer Models"
    )
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    store_submitted_text_by_default: bool = False
    distress_label_keywords: List[str] = field(
        default_factory=lambda: [
            "distress",
            "depress",
            "anxiety",
            "stress",
            "suicid",
            "crisis",
            "self-harm",
            "high",
            "risk",
        ]
    )
