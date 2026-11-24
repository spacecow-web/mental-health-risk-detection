"""Optional FastAPI bridge enabling the live web workbench to invoke local PyTorch + SHAP/LIME."""

from functools import lru_cache
from typing import Dict, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import AppConfig, PreprocessingConfig
from src.ethics import detect_immediate_self_harm_cues, should_show_supportive_guidance
from src.explainability import ExplainabilityEngine
from src.model_inference import MentalHealthTransformerClassifier, ModelNotReadyError
from src.preprocessing import preprocess_text, validate_text


app = FastAPI(title="MindTrace AI Local Inference Bridge", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    method: Literal["SHAP", "LIME"] = "SHAP"
    mask_urls: bool = True
    anonymize_mentions: bool = True
    strip_hashtag_symbol: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False


@lru_cache(maxsize=1)
def get_classifier() -> MentalHealthTransformerClassifier:
    return MentalHealthTransformerClassifier(AppConfig().model)


@lru_cache(maxsize=1)
def get_explainer() -> ExplainabilityEngine:
    return ExplainabilityEngine(get_classifier(), AppConfig().explainability)


@app.get("/health")
def health() -> Dict[str, object]:
    try:
        clf = get_classifier()
        meta = clf.training_metadata or {}
        return {
            "ready": True,
            "labels": clf.class_names,
            "model": meta.get("backbone", clf.config.backbone_name),
            "training_metadata": clf.training_metadata,
        }
    except ModelNotReadyError as exc:
        return {"ready": False, "detail": str(exc)}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, object]:
    prep_cfg = PreprocessingConfig(
        mask_urls=req.mask_urls,
        anonymize_mentions=req.anonymize_mentions,
        strip_hashtag_symbol=req.strip_hashtag_symbol,
        normalize_whitespace=req.normalize_whitespace,
        lowercase=req.lowercase,
    )
    validation = validate_text(req.text, prep_cfg)
    if not validation.is_valid:
        raise HTTPException(status_code=400, detail=validation.error_message)

    cleaned = preprocess_text(req.text, prep_cfg)
    try:
        clf = get_classifier()
        explainer = get_explainer()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    prediction = clf.predict(cleaned.cleaned_text)
    report = explainer.explain(
        cleaned.cleaned_text,
        prediction,
        method=req.method,
        original_text=req.text,
    )

    return {
        "original_text": req.text,
        "cleaned_text": cleaned.cleaned_text,
        "predicted_label": prediction.predicted_label,
        "confidence": prediction.confidence,
        "logits": prediction.logits,
        "probabilities": prediction.probabilities,
        "normalized_entropy": prediction.normalized_entropy,
        "is_uncertain": prediction.is_uncertain,
        "tokens": prediction.tokens,
        "model_name": prediction.model_name,
        "confidence_disclaimer": prediction.confidence_disclaimer,
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
        "phrase_attributions": [
            {
                "phrase": p.phrase,
                "weight": p.weight,
                "direction": p.direction,
                "start": p.start,
                "end": p.end,
                "tokens": p.tokens,
            }
            for p in report.phrase_attributions
        ],
        "natural_language_summary": report.natural_language_summary,
        "immediate_safety": detect_immediate_self_harm_cues(req.text),
        "supportive_guidance": should_show_supportive_guidance(
            req.text, prediction.predicted_label, AppConfig().distress_label_keywords
        ),
    }
