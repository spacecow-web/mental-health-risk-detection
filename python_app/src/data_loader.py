"""Dataset loading, inspection, deduplication, leakage-free splitting, and class weighting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import PreprocessingConfig
from src.preprocessing import preprocess_text


REQUIRED_DATASET_GUIDE = """
No labeled dataset was supplied. MindTrace AI requires a UTF-8 CSV file containing at least:
  1. A text column (default: 'text') holding raw social media posts.
  2. A target label column (default: 'label') holding categorical class labels
     (binary e.g. Non-Distress / Distress, or multi-class e.g. Neutral / Stress / Anxiety / Depression).

Example CSV structure:
  text,label
  "I feel exhausted and cannot sleep before exams.",Stress
  "Had a relaxing morning walk with coffee today.",Neutral
""".strip()


class DatasetNotSuppliedError(FileNotFoundError):
    """Raised when a required labeled CSV dataset is missing or unreadable."""


@dataclass
class DatasetInspectionReport:
    """Summary of raw dataset columns, missingness, duplicates, and label distribution."""

    total_rows: int
    columns: List[str]
    text_col: str
    label_col: str
    missing_text_rows: int
    missing_label_rows: int
    duplicate_raw_texts: int
    label_counts: Dict[str, int]
    imbalance_ratio: float


@dataclass
class PreparedSplits:
    """Container holding leakage-free stratified train, validation, and test DataFrames."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    removed_invalid_rows: int
    removed_duplicate_rows: int


def load_labeled_dataset(
    data_path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load a labeled social media CSV dataset and verify required columns exist.

    Raises DatasetNotSuppliedError with dataset schema instructions if the file is missing.
    """
    path = Path(data_path)
    if not path.exists() or not path.is_file():
        raise DatasetNotSuppliedError(
            f"Dataset file not found at '{path}'.\n\n{REQUIRED_DATASET_GUIDE}"
        )

    df = pd.read_csv(path, encoding="utf-8")
    if df.empty:
        raise ValueError(f"Dataset file '{path}' contains zero rows.")

    missing_cols = [c for c in (text_col, label_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}.\n\n{REQUIRED_DATASET_GUIDE}"
        )
    return df


def inspect_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> DatasetInspectionReport:
    """Inspect column schema, missing values, raw duplicates, and target label imbalance."""
    missing_text = int(df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == "").sum())
    missing_label = int(df[label_col].isna().sum() + (df[label_col].astype(str).str.strip() == "").sum())
    non_null_texts = df[text_col].dropna().astype(str).str.strip()
    duplicate_raw = int(non_null_texts.duplicated().sum())

    valid_labels = df.dropna(subset=[text_col, label_col])[label_col].astype(str).str.strip()
    counts_series = valid_labels.value_counts()
    label_counts = {str(k): int(v) for k, v in counts_series.items()}

    max_count = max(label_counts.values()) if label_counts else 0
    min_count = min(label_counts.values()) if label_counts else 0
    imbalance_ratio = float(max_count / min_count) if min_count > 0 else 0.0

    return DatasetInspectionReport(
        total_rows=len(df),
        columns=list(df.columns),
        text_col=text_col,
        label_col=label_col,
        missing_text_rows=missing_text,
        missing_label_rows=missing_label,
        duplicate_raw_texts=duplicate_raw,
        label_counts=label_counts,
        imbalance_ratio=imbalance_ratio,
    )


def clean_and_deduplicate_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    prep_config: PreprocessingConfig = PreprocessingConfig(),
) -> Tuple[pd.DataFrame, int, int]:
    """Drop invalid/short rows, clean social media text, and deduplicate BEFORE splitting.

    Deduplicating on normalized text prior to splitting prevents identical or near-identical
    posts from leaking across train, validation, and held-out test sets.
    """
    initial_count = len(df)
    cleaned_df = df[[text_col, label_col]].dropna(subset=[text_col, label_col]).copy()
    cleaned_df[text_col] = cleaned_df[text_col].astype(str).str.strip()
    cleaned_df[label_col] = cleaned_df[label_col].astype(str).str.strip()

    cleaned_df = cleaned_df[(cleaned_df[text_col] != "") & (cleaned_df[label_col] != "")]
    cleaned_df["cleaned_text"] = cleaned_df[text_col].apply(
        lambda t: preprocess_text(t, prep_config).cleaned_text
    )
    cleaned_df = cleaned_df[cleaned_df["cleaned_text"].str.len() >= prep_config.min_chars].copy()
    removed_invalid = initial_count - len(cleaned_df)

    # Canonical key for duplicate detection across case/whitespace differences
    cleaned_df["_dedup_key"] = cleaned_df["cleaned_text"].str.lower().str.strip()
    before_dedup = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["_dedup_key"], keep="first").drop(
        columns=["_dedup_key"]
    )
    removed_duplicates = before_dedup - len(cleaned_df)

    return cleaned_df.reset_index(drop=True), removed_invalid, removed_duplicates


def stratified_train_val_test_split(
    cleaned_df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> PreparedSplits:
    """Split deduplicated data into stratified train, validation, and test sets with zero leakage."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-4):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    labels_sorted = sorted(cleaned_df[label_col].unique().tolist())
    if len(labels_sorted) < 2:
        raise ValueError(
            f"Classification requires at least 2 distinct labels; found {labels_sorted}."
        )

    min_class_count = int(cleaned_df[label_col].value_counts().min())
    if min_class_count < 3:
        raise ValueError(
            f"Every class needs at least 3 samples for stratified 3-way splitting (min={min_class_count})."
        )

    label2id = {lbl: idx for idx, lbl in enumerate(labels_sorted)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    work_df = cleaned_df.copy()
    work_df["label_id"] = work_df[label_col].map(label2id)

    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        work_df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=work_df["label_id"],
    )

    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["label_id"],
    )

    # Strict zero-leakage verification across splits
    train_texts = set(train_df["cleaned_text"].str.lower())
    val_texts = set(val_df["cleaned_text"].str.lower())
    test_texts = set(test_df["cleaned_text"].str.lower())
    if (train_texts & val_texts) or (train_texts & test_texts) or (val_texts & test_texts):
        raise RuntimeError("Data leakage detected: overlapping text across train/val/test splits.")

    return PreparedSplits(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        label2id=label2id,
        id2label=id2label,
        removed_invalid_rows=0,
        removed_duplicate_rows=0,
    )


def compute_balanced_class_weights(train_label_ids: List[int], num_classes: int) -> np.ndarray:
    """Compute inverse-frequency balanced class weights strictly from the training split."""
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(train_label_ids, dtype=int),
    )
    return weights.astype(np.float32)
