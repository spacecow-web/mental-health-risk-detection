"""Transformer sequence classification inference module (PyTorch + Hugging Face)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import math

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ModelConfig
from src.ethics import CONFIDENCE_NOT_CLINICAL_PROBABILITY


class ModelNotReadyError(RuntimeError):
    """Raised when no fine-tuned Transformer checkpoint is available on disk."""


@dataclass
class PredictionResult:
    """Structured inference output including raw logits, probabilities, and uncertainty."""

    predicted_label: str
    confidence: float
    logits: Dict[str, float]
    probabilities: Dict[str, float]
    normalized_entropy: float
    is_uncertain: bool
    tokens: List[str]
    model_name: str
    confidence_disclaimer: str


def compute_normalized_entropy(probs: Sequence[float]) -> float:
    """Compute normalized Shannon entropy in [0, 1] as a measure of predictive uncertainty."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    clipped = np.clip(arr, 1e-12, 1.0)
    entropy = -float(np.sum(clipped * np.log(clipped)))
    max_entropy = math.log(arr.size)
    return float(min(max(entropy / max_entropy, 0.0), 1.0))


class MentalHealthTransformerClassifier:
    """Loads a saved fine-tuned DistilBERT/BERT checkpoint and performs deterministic inference."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(config.checkpoint_dir)

        if not self._has_valid_checkpoint(self.checkpoint_dir):
            raise ModelNotReadyError(
                f"No trained Transformer checkpoint found at '{self.checkpoint_dir}'. "
                "Supply a fine-tuned model directory or run `python -m src.train --data_path <csv>` first."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.checkpoint_dir))
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self._resolve_id2label()
        self.training_metadata = self._load_training_metadata()

    @staticmethod
    def _has_valid_checkpoint(checkpoint_dir: Path) -> bool:
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return False
        config_file = checkpoint_dir / "config.json"
        has_weights = any(
            (checkpoint_dir / fname).exists()
            for fname in ("model.safetensors", "pytorch_model.bin")
        )
        return config_file.exists() and has_weights

    def _resolve_id2label(self) -> Dict[int, str]:
        label_map_path = self.checkpoint_dir / "label_mapping.json"
        if label_map_path.exists():
            raw = json.loads(label_map_path.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in raw.items()}
        raw_cfg = getattr(self.model.config, "id2label", None) or {}
        return {int(k): str(v) for k, v in raw_cfg.items()}

    def _load_training_metadata(self) -> Optional[Dict[str, Any]]:
        meta_path = self.checkpoint_dir / "training_metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return None

    @property
    def class_names(self) -> List[str]:
        return [self.id2label[idx] for idx in sorted(self.id2label.keys())]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Tokenize batch of texts, compute logits, and return softmax probabilities (for SHAP/LIME)."""
        if isinstance(texts, str):
            texts = [texts]
        text_list = [str(t) for t in texts]
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)
        return probs.detach().cpu().numpy()

    def predict(self, cleaned_text: str) -> PredictionResult:
        """Tokenize text, generate logits, apply softmax, and return calibrated prediction metadata."""
        encoded = self.tokenizer(
            [cleaned_text],
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits_tensor = outputs.logits[0]
            probs_tensor = torch.softmax(logits_tensor, dim=-1)

        logits_vec = logits_tensor.detach().cpu().numpy()
        probs_vec = probs_tensor.detach().cpu().numpy()

        ordered_indices = sorted(self.id2label.keys())
        logits_map = {self.id2label[idx]: float(logits_vec[idx]) for idx in ordered_indices}
        probabilities = {self.id2label[idx]: float(probs_vec[idx]) for idx in ordered_indices}

        best_idx = int(np.argmax(probs_vec))
        predicted_label = self.id2label[best_idx]
        confidence = float(probs_vec[best_idx])
        entropy = compute_normalized_entropy(probs_vec)

        token_ids = encoded["input_ids"][0].detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        return PredictionResult(
            predicted_label=predicted_label,
            confidence=confidence,
            logits=logits_map,
            probabilities=probabilities,
            normalized_entropy=entropy,
            is_uncertain=entropy >= self.config.high_uncertainty_entropy_threshold,
            tokens=tokens,
            model_name=str(getattr(self.model.config, "_name_or_path", self.config.backbone_name)),
            confidence_disclaimer=CONFIDENCE_NOT_CLINICAL_PROBABILITY,
        )
