# MindTrace AI

**Explainable Mental Health Risk Detection from Social Media Text Using Transformer Models**

MindTrace AI is an academic and educational NLP research prototype designed to analyze social media text for linguistic patterns associated with psychological distress using fine-tuned Transformer models (`DistilBERT` / `BERT`) and post-hoc Explainable AI (`SHAP` and `LIME`).

> **Important Disclaimer:** MindTrace AI is strictly a research and educational prototype. It is **not** a clinical diagnostic tool, medical device, or substitute for professional mental health assessment. Model confidence is derived from neural network logits across training labels and is **not** the probability that a person has a mental health condition.

---

## 1. System Architecture

```text
Labeled CSV Corpus
        │
        ▼
┌───────────────────────────────────────────┐
│ Dataset Loader & Leakage Guard            │  src/data_loader.py
│ • Schema & label imbalance inspection     │
│ • Invalid/short row removal               │
│ • Pre-split deduplication                 │
│ • Stratified 70/15/15 split verification  │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│ Transformer Fine-Tuning & Calibration     │  src/train.py
│ • DistilBERT / BERT + Hugging Face        │
│ • Class-weighted CrossEntropyLoss         │
│ • EarlyStoppingCallback + fixed seeds     │
│ • Saves model, tokenizer, & metadata      │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│ Inference & Explainability                │  src/model_inference.py
│ • Tokenizes cleaned input                 │  src/explainability.py
│ • Computes logits + softmax probabilities │
│ • Partition SHAP & LIME attributions      │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│ Streamlit Academic Dashboard              │  app.py
│ • 6-Page Research Interface               │
│ • Held-Out Scikit-learn Evaluation        │
│ • Privacy-First & Crisis Guardrails       │
└───────────────────────────────────────────┘
```

---

## 2. Folder Structure

```text
python_app/
├── app.py                     # Multi-page Streamlit academic dashboard
├── requirements.txt           # Python package dependencies
├── README.md                  # Documentation and setup instructions
├── artifacts/                 # Saved trained model checkpoints (created after training)
├── data/                      # User-supplied datasets, training history, & held-out predictions
└── src/
    ├── __init__.py
    ├── config.py              # Central dataclasses, paths, and ethical defaults
    ├── data_loader.py         # Dataset loading, inspection, deduplication, stratified splitting
    ├── preprocessing.py       # Validation, anonymization, and tokenization helpers
    ├── model_inference.py     # PyTorch / Hugging Face Transformer classifier wrapper
    ├── explainability.py      # Genuine SHAP & LIME token attribution and Plotly charts
    ├── evaluation.py          # Scikit-learn held-out evaluation & confusion/ROC plots
    ├── dataset_analytics.py   # Pandas/NumPy exploratory data analysis & n-grams
    ├── ethics.py              # Disclaimers, privacy policy, bias notes, crisis resources
    ├── train.py               # Reproducible fine-tuning CLI with early stopping & class weights
    └── api_bridge.py          # Optional FastAPI bridge for the live web workbench
```

---

## 3. Module Descriptions

- **`src/data_loader.py`**: Loads labeled CSV datasets (`load_labeled_dataset`), inspects columns and label imbalance (`inspect_dataset`), removes invalid/duplicate records prior to splitting (`clean_and_deduplicate_dataset`), performs zero-leakage stratified 70/15/15 splitting (`stratified_train_val_test_split`), and computes balanced inverse-frequency class weights (`compute_balanced_class_weights`).
- **`src/preprocessing.py`**: Performs deterministic validation (`validate_text`), PII anonymization (masking URLs as `[URL]` and `@handles` as `[USER]`), whitespace normalization, and token previewing with Hugging Face tokenizers.
- **`src/train.py`**: Reproducible CLI fine-tuning script (`run_training_pipeline`) using fixed random seeds, `WeightedLossTrainer` for imbalanced labels, `EarlyStoppingCallback`, validation macro-F1 checkpoint selection, and held-out test prediction export.
- **`src/model_inference.py`**: Loads a saved Transformer checkpoint (`MentalHealthTransformerClassifier`), tokenizes input text, computes raw logits, converts logits into softmax probabilities, calculates normalized Shannon entropy, and attaches the non-clinical confidence disclaimer.
- **`src/explainability.py`**: Implements `ExplainabilityEngine` using `shap.Explainer` (`shap.maskers.Text`) and `lime.lime_text.LimeTextExplainer` to compute positive and negative token attributions.
- **`src/evaluation.py`**: Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and One-vs-Rest ROC-AUC strictly from held-out test predictions using `scikit-learn`.
- **`src/dataset_analytics.py`**: Computes corpus class balance, post length quartiles, and per-class frequent terms.
- **`src/ethics.py`**: Centralizes research disclaimers, privacy-by-default rules, dataset limitations, and crisis helpline resources.
- **`src/api_bridge.py`**: Optional local FastAPI server exposing `/health` and `/analyze` for the MindTrace AI web workbench.
- **`app.py`**: Six-page Streamlit dashboard connected directly to `MentalHealthTransformerClassifier` and `ExplainabilityEngine`.

---

## 4. Dataset Requirements

MindTrace AI does **not** ship with fabricated data or synthetic metrics. Supply a UTF-8 labeled CSV dataset containing at least:

- **`text`**: Raw social media post text.
- **`label`**: Ground-truth category label (binary, e.g., `Non-Distress` / `Distress`, or multi-class, e.g., `Neutral`, `Stress`, `Anxiety`, `Depression`).

---

## 5. Installation & Quick Start

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Fine-tune a Transformer model on your labeled dataset**:
   ```bash
   python -m src.train \
     --data_path data/your_dataset.csv \
     --text_col text \
     --label_col label \
     --model_name distilbert-base-uncased \
     --output_dir artifacts/model \
     --eval_output data/held_out_predictions.csv \
     --history_output data/training_history.csv \
     --epochs 4
   ```

4. **Launch the Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```

5. *(Optional)* **Start the local API bridge for the live web workbench**:
   ```bash
   uvicorn src.api_bridge:app --host 127.0.0.1 --port 8000
   ```
