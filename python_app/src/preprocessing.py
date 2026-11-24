"""Deterministic text validation, anonymization, and tokenization helpers."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import PreprocessingConfig


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,50}")
HASHTAG_PATTERN = re.compile(r"#([A-Za-z0-9_]+)")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


@dataclass
class ValidationResult:
    """Result of validating raw social media input text."""

    is_valid: bool
    error_message: Optional[str]
    warnings: List[str]
    char_count: int
    word_count: int


@dataclass
class PreprocessedText:
    """Container holding raw text, cleaned text, and transformation metadata."""

    raw_text: str
    cleaned_text: str
    urls_masked: int
    mentions_masked: int
    char_count_raw: int
    char_count_cleaned: int
    word_count_cleaned: int


def validate_text(text: str, config: PreprocessingConfig = PreprocessingConfig()) -> ValidationResult:
    """Validate input length and content before tokenization or inference."""
    if not isinstance(text, str):
        return ValidationResult(False, "Input must be a valid UTF-8 text string.", [], 0, 0)

    stripped = text.strip()
    char_count = len(stripped)
    words = [w for w in stripped.split() if w]
    word_count = len(words)

    if char_count == 0:
        return ValidationResult(
            is_valid=False,
            error_message="Please enter social media text before running analysis.",
            warnings=[],
            char_count=0,
            word_count=0,
        )

    if char_count < config.min_chars:
        return ValidationResult(
            is_valid=False,
            error_message=f"Input is too short ({char_count} characters). Provide at least {config.min_chars} characters.",
            warnings=[],
            char_count=char_count,
            word_count=word_count,
        )

    warnings: List[str] = []
    if char_count > config.max_chars:
        warnings.append(
            f"Input exceeds {config.max_chars} characters ({char_count}). "
            f"Transformer tokenization will truncate sequences beyond {config.max_seq_length} subword tokens."
        )
    if word_count < 4:
        warnings.append(
            "Very short snippets (< 4 words) lack conversational context and increase prediction uncertainty."
        )

    return ValidationResult(
        is_valid=True,
        error_message=None,
        warnings=warnings,
        char_count=char_count,
        word_count=word_count,
    )


def preprocess_text(
    text: str,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> PreprocessedText:
    """Clean and anonymize social media text deterministically without altering core semantics."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = CONTROL_CHARS_PATTERN.sub(" ", normalized)

    urls_masked = 0
    if config.mask_urls:
        normalized, urls_masked = URL_PATTERN.subn("[URL]", normalized)

    mentions_masked = 0
    if config.anonymize_mentions:
        normalized, mentions_masked = MENTION_PATTERN.subn("[USER]", normalized)

    if config.strip_hashtag_symbol:
        normalized = HASHTAG_PATTERN.sub(r"\1", normalized)

    if config.normalize_whitespace:
        normalized = MULTI_SPACE_PATTERN.sub(" ", normalized).strip()

    if config.lowercase:
        normalized = normalized.lower()

    words = [w for w in normalized.split() if w]
    return PreprocessedText(
        raw_text=text,
        cleaned_text=normalized,
        urls_masked=urls_masked,
        mentions_masked=mentions_masked,
        char_count_raw=len(text),
        char_count_cleaned=len(normalized),
        word_count_cleaned=len(words),
    )


def inspect_tokens(
    cleaned_text: str,
    tokenizer: Any,
    max_seq_length: int = 256,
) -> Dict[str, Any]:
    """Tokenize cleaned text using a Hugging Face tokenizer and return inspectable metadata."""
    encoded = tokenizer(
        cleaned_text,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
    )
    input_ids: List[int] = encoded["input_ids"]
    tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids)
    # Encode up to max_seq_length + 1 to detect truncation without triggering sequence-length warnings
    probe_ids = tokenizer.encode(cleaned_text, truncation=True, max_length=max_seq_length + 1)
    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "token_count": len(tokens),
        "truncated": len(probe_ids) > max_seq_length,
    }
