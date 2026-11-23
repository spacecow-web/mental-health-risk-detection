import JSZip from 'jszip';

import readmeRaw from '../../python_app/README.md?raw';
import requirementsRaw from '../../python_app/requirements.txt?raw';
import appRaw from '../../python_app/app.py?raw';
import initRaw from '../../python_app/src/__init__.py?raw';
import configRaw from '../../python_app/src/config.py?raw';
import dataLoaderRaw from '../../python_app/src/data_loader.py?raw';
import preprocessingRaw from '../../python_app/src/preprocessing.py?raw';
import modelInferenceRaw from '../../python_app/src/model_inference.py?raw';
import explainabilityRaw from '../../python_app/src/explainability.py?raw';
import evaluationRaw from '../../python_app/src/evaluation.py?raw';
import datasetAnalyticsRaw from '../../python_app/src/dataset_analytics.py?raw';
import ethicsRaw from '../../python_app/src/ethics.py?raw';
import trainRaw from '../../python_app/src/train.py?raw';
import apiBridgeRaw from '../../python_app/src/api_bridge.py?raw';

export interface PythonProjectFile {
  path: string;
  role: string;
  content: string;
}

export const PYTHON_PROJECT_FILES: PythonProjectFile[] = [
  {
    path: 'README.md',
    role: 'Architecture, folder structure, dataset schema, and installation guide',
    content: readmeRaw,
  },
  {
    path: 'requirements.txt',
    role: 'Pinned Python dependencies (Streamlit, PyTorch, Transformers, SHAP, LIME, Scikit-learn)',
    content: requirementsRaw,
  },
  {
    path: 'app.py',
    role: 'Six-page Streamlit academic dashboard UI connected to saved Transformer inference',
    content: appRaw,
  },
  {
    path: 'src/__init__.py',
    role: 'Python package initializer',
    content: initRaw,
  },
  {
    path: 'src/config.py',
    role: 'Central dataclasses for preprocessing, Transformer backbones, and XAI defaults',
    content: configRaw,
  },
  {
    path: 'src/data_loader.py',
    role: 'Dataset loading, column/label inspection, pre-split deduplication, stratified splitting, and class weights',
    content: dataLoaderRaw,
  },
  {
    path: 'src/preprocessing.py',
    role: 'Text validation, PII masking ([URL]/[USER]), and Hugging Face tokenizer inspection',
    content: preprocessingRaw,
  },
  {
    path: 'src/model_inference.py',
    role: 'DistilBERT/BERT PyTorch inference wrapper with predictive Shannon entropy',
    content: modelInferenceRaw,
  },
  {
    path: 'src/explainability.py',
    role: 'Genuine SHAP Partition & LIME text explainers, Plotly charts, and HTML highlights',
    content: explainabilityRaw,
  },
  {
    path: 'src/evaluation.py',
    role: 'Held-out Scikit-learn metrics (Accuracy, P/R/F1, Confusion Matrix, ROC-AUC)',
    content: evaluationRaw,
  },
  {
    path: 'src/dataset_analytics.py',
    role: 'Pandas/NumPy corpus quality audit, class balance, and vocabulary frequencies',
    content: datasetAnalyticsRaw,
  },
  {
    path: 'src/ethics.py',
    role: 'Non-diagnostic disclaimers, privacy policy, bias disclosures, and crisis helplines',
    content: ethicsRaw,
  },
  {
    path: 'src/train.py',
    role: 'CLI fine-tuning script saving checkpoints and held_out_predictions.csv',
    content: trainRaw,
  },
  {
    path: 'src/api_bridge.py',
    role: 'Optional FastAPI bridge connecting local PyTorch + SHAP/LIME to this web workbench',
    content: apiBridgeRaw,
  },
];

export async function downloadPythonKitZip(): Promise<void> {
  const zip = new JSZip();
  const root = zip.folder('mindtrace_ai_python');
  if (!root) {
    throw new Error('Could not initialize ZIP archive.');
  }
  for (const file of PYTHON_PROJECT_FILES) {
    root.file(file.path, file.content);
  }
  const blob = await zip.generateAsync({ type: 'blob' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'mindtrace-ai-python.zip';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}
