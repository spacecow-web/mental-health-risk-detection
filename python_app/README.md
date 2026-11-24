# MindTrace AI

**Explainable Mental Health Risk Detection from Social Media Text Using Transformer Models**

MindTrace AI is an academic and educational NLP research prototype designed to analyze social media text for linguistic patterns associated with psychological distress using fine-tuned Transformer models (`DistilBERT` / `BERT`) and post-hoc Explainable AI (`SHAP` and `LIME`). The dashboard loads its configured default dataset, active model checkpoint, and saved evaluation metrics automatically on startup without requiring manual uploads.

> **Educational Research Disclaimer:** This application is an educational research prototype. Its predictions indicate patterns in text learned from a dataset and are not medical diagnoses or validated clinical assessments. The results should not be used to label, monitor or make decisions about individuals.
>
> **Privacy Notice:** Submitted text is processed strictly in ephemeral session memory and is not stored, logged, or saved by default. Safety checks for immediate danger or self-harm run independently of model predictions alone.

---

## 1. System Architecture

```text
Default Research Dataset (data/default_dataset.csv)
        │
        ▼
┌───────────────────────────────────────────┐
│ Dataset Loader & Leakage Guard            │  src/data_loader.py
│ • Auto-discovery & schema inspection      │
│ • Invalid/short row removal               │
│ • Pre-split deduplication                 │
│ • Stratified 70/15/15 split verification  │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│ Transformer Fine-Tuning & Persistence     │  src/train.py
│ • DistilBERT / BERT + Hugging Face        │
│ • Class-weighted CrossEntropyLoss         │
│ • EarlyStoppingCallback + fixed seeds     │
│ • Persists model, held-out & JSON metrics │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│ Inference & Explainability                │  src/model_inference.py
│ • Auto-cached checkpoint loading          │  src/explainability.py
│ • Computes logits + softmax probabilities │
│ • Partition SHAP & LIME attributions      │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│ Streamlit Academic Dashboard              │  app.py
│ • Zero-upload 6-page research interface   │
│ • Auto-loaded persisted evaluations       │
│ • Privacy-First & Crisis Guardrails       │
└───────────────────────────────────────────┘
```

---

## 2. Folder Structure

```text
python_app/
├── app.py                     # Zero-upload 6-page Streamlit academic dashboard
├── requirements.txt           # Python package dependencies
├── README.md                  # Documentation and setup instructions
├── artifacts/                 # Saved trained model checkpoints & evaluation_summary.json
├── data/                      # Default dataset (default_dataset.csv), history, & predictions
└── src/
    ├── __init__.py
    ├── config.py              # Central dataclasses, default paths, and ethical defaults
    ├── data_loader.py         # Dataset discovery, inspection, deduplication, stratified splitting
    ├── preprocessing.py       # Validation, anonymization, and tokenization helpers
    ├── model_inference.py     # PyTorch / Hugging Face Transformer classifier wrapper
    ├── explainability.py      # Genuine SHAP & LIME token attribution and Plotly charts
    ├── evaluation.py          # Scikit-learn metrics, live classifier evaluation, JSON persistence
    ├── dataset_analytics.py   # Pandas/NumPy exploratory data analysis & n-grams
    ├── ethics.py              # Disclaimers, privacy policy, bias notes, crisis resources
    ├── train.py               # Reproducible fine-tuning CLI saving persisted metrics
    └── api_bridge.py          # Optional FastAPI bridge for the live web workbench
```

---

## 3. Dataset Setup

Place your UTF-8 labeled research CSV at `data/default_dataset.csv` (with `text` and `label` columns, and an optional `split` column). Once placed, Overview and Dataset Analytics inspect it automatically without requiring manual uploads. Optional custom CSV uploads remain available under expandable **Advanced Options**.

---

## 4. Installation & Quick Start

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Fine-tune and persist a Transformer checkpoint**:
   ```bash
   python -m src.train \
     --data_path data/default_dataset.csv \
     --model_name distilbert-base-uncased \
     --output_dir artifacts/model \
     --eval_output data/held_out_predictions.csv \
     --history_output data/training_history.csv \
     --epochs 4
   ```
   *(If `--data_path` is omitted, `src.train` automatically resolves `data/default_dataset.csv`.)*

4. **Launch the Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```

5. *(Optional)* **Start the local API bridge for the live web workbench**:
   ```bash
   uvicorn src.api_bridge:app --host 127.0.0.1 --port 8000
   ```
