"""Reproducible fine-tuning pipeline for DistilBERT / BERT mental health classification."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import DEFAULT_EVAL_SUMMARY_JSON, PreprocessingConfig
from src.data_loader import (
    DatasetNotSuppliedError,
    REQUIRED_DATASET_GUIDE,
    clean_and_deduplicate_dataset,
    compute_balanced_class_weights,
    inspect_dataset,
    load_labeled_dataset,
    resolve_default_dataset_path,
    stratified_train_val_test_split,
)
from src.evaluation import evaluate_held_out_predictions, save_evaluation_summary


def set_reproducibility_seed(seed: int) -> None:
    """Fix Python, NumPy, PyTorch, and Hugging Face random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


class TokenizedTextDataset(torch.utils.data.Dataset):
    """PyTorch Dataset storing tokenized encodings and integer class labels."""

    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


class WeightedLossTrainer(Trainer):
    """Custom Hugging Face Trainer applying inverse-frequency class weights to handle imbalance."""

    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_trainer_metrics(eval_pred) -> Dict[str, float]:
    """Compute validation accuracy and macro-F1 during training epochs."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float(accuracy_score(labels, preds))
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "macro_f1": float(macro_f1)}


def export_training_history(log_history: List[Dict[str, Any]], history_path: Path) -> None:
    """Persist epoch-level training and validation metrics for dashboard plotting."""
    epoch_rows: Dict[float, Dict[str, float]] = {}
    for entry in log_history:
        if "epoch" not in entry:
            continue
        ep = round(float(entry["epoch"]), 2)
        bucket = epoch_rows.setdefault(ep, {"epoch": ep})
        if "loss" in entry:
            bucket["train_loss"] = float(entry["loss"])
        if "eval_loss" in entry:
            bucket["eval_loss"] = float(entry["eval_loss"])
        if "eval_accuracy" in entry:
            bucket["eval_accuracy"] = float(entry["eval_accuracy"])
        if "eval_macro_f1" in entry:
            bucket["eval_f1"] = float(entry["eval_macro_f1"])

    if not epoch_rows:
        return
    history_df = pd.DataFrame(sorted(epoch_rows.values(), key=lambda r: r["epoch"]))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)


def run_training_pipeline(
    data_path: str,
    text_col: str = "text",
    label_col: str = "label",
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "artifacts/model",
    eval_output: str = "data/held_out_predictions.csv",
    history_output: str = "data/training_history.csv",
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    early_stopping_patience: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end dataset loading, leakage-free splitting, fine-tuning, and test export."""
    set_reproducibility_seed(seed)

    # 1-2. Load and inspect raw dataset
    raw_df = load_labeled_dataset(data_path, text_col=text_col, label_col=label_col)
    inspection = inspect_dataset(raw_df, text_col=text_col, label_col=label_col)
    print(
        f"Loaded {inspection.total_rows} rows across columns {inspection.columns}. "
        f"Label counts: {inspection.label_counts} (imbalance ratio: {inspection.imbalance_ratio:.2f}x)."
    )

    # 3-4. Clean and deduplicate prior to splitting to prevent data leakage
    prep_cfg = PreprocessingConfig(max_seq_length=max_length)
    cleaned_df, removed_invalid, removed_dups = clean_and_deduplicate_dataset(
        raw_df, text_col=text_col, label_col=label_col, prep_config=prep_cfg
    )
    if cleaned_df.empty:
        raise ValueError(
            "Zero valid rows remain after cleaning and minimum-length filtering."
        )
    print(
        f"Cleaned corpus: {len(cleaned_df)} unique valid rows "
        f"(removed {removed_invalid} invalid/short, {removed_dups} duplicates)."
    )

    # 5-6. Stratified leakage-free train / validation / test split
    splits = stratified_train_val_test_split(cleaned_df, label_col=label_col, seed=seed)
    print(
        f"Stratified splits -> train: {len(splits.train_df)}, "
        f"val: {len(splits.val_df)}, test: {len(splits.test_df)}."
    )

    # Compute balanced class weights strictly from the training split
    train_label_ids = splits.train_df["label_id"].tolist()
    class_weights_np = compute_balanced_class_weights(train_label_ids, len(splits.id2label))
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

    # 7. Tokenize each split independently using the Transformer tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenizer(
        splits.train_df["cleaned_text"].tolist(), truncation=True, max_length=max_length
    )
    val_enc = tokenizer(
        splits.val_df["cleaned_text"].tolist(), truncation=True, max_length=max_length
    )
    test_enc = tokenizer(
        splits.test_df["cleaned_text"].tolist(), truncation=True, max_length=max_length
    )

    train_dataset = TokenizedTextDataset(train_enc, train_label_ids)
    val_dataset = TokenizedTextDataset(val_enc, splits.val_df["label_id"].tolist())
    test_dataset = TokenizedTextDataset(test_enc, splits.test_df["label_id"].tolist())

    # 8. Initialize and fine-tune Transformer model with early stopping and weighted loss
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(splits.id2label),
        id2label=splits.id2label,
        label2id=splits.label2id,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()

    # 9. Save best fine-tuned model, tokenizer, label mapping, and training history
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(out_path)
    (out_path / "label_mapping.json").write_text(
        json.dumps(splits.id2label, indent=2), encoding="utf-8"
    )
    export_training_history(trainer.state.log_history, Path(history_output))

    # Evaluate strictly on held-out test set and export predictions CSV
    pred_output = trainer.predict(test_dataset)
    logits = pred_output.predictions
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pred_ids = np.argmax(probs, axis=1)

    y_true = splits.test_df[label_col].tolist()
    y_pred = [splits.id2label[int(i)] for i in pred_ids]
    test_acc = float(accuracy_score(y_true, y_pred))
    _, _, test_macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    eval_df = pd.DataFrame({
        "text": splits.test_df["cleaned_text"].tolist(),
        "true_label": y_true,
        "predicted_label": y_pred,
    })
    for idx, lbl in splits.id2label.items():
        eval_df[f"prob_{lbl}"] = probs[:, idx]

    eval_path = Path(eval_output)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(eval_path, index=False)

    eval_summary = evaluate_held_out_predictions(eval_df)
    save_evaluation_summary(
        eval_summary,
        model_name=model_name,
        source_file=str(eval_path),
        output_path=DEFAULT_EVAL_SUMMARY_JSON,
    )

    metadata = {
        "training_completed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": model_name,
        "seed": seed,
        "train_samples": len(splits.train_df),
        "val_samples": len(splits.val_df),
        "test_samples": len(splits.test_df),
        "labels": splits.id2label,
        "class_weights": {
            splits.id2label[i]: float(w) for i, w in enumerate(class_weights_np)
        },
        "held_out_accuracy": test_acc,
        "held_out_macro_f1": float(test_macro_f1),
    }
    (out_path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Training complete. Saved model to {out_path} | "
        f"Held-out Accuracy={test_acc:.4f}, Macro-F1={test_macro_f1:.4f}."
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT or BERT on labeled social media mental health text."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to labeled CSV dataset (defaults to data/default_dataset.csv).",
    )
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        choices=["distilbert-base-uncased", "bert-base-uncased"],
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/model")
    parser.add_argument("--eval_output", type=str, default="data/held_out_predictions.csv")
    parser.add_argument("--history_output", type=str, default="data/training_history.csv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    resolved_data = Path(args.data_path) if args.data_path else resolve_default_dataset_path()
    if resolved_data is None:
        print(REQUIRED_DATASET_GUIDE)
        raise SystemExit(1)

    try:
        run_training_pipeline(
            data_path=str(resolved_data),
            text_col=args.text_col,
            label_col=args.label_col,
            model_name=args.model_name,
            output_dir=args.output_dir,
            eval_output=args.eval_output,
            history_output=args.history_output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            early_stopping_patience=args.patience,
            seed=args.seed,
        )
    except DatasetNotSuppliedError as exc:
        print(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
