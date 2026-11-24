import { useState, type ChangeEvent } from 'react';
import { Check, ChevronDown, Copy, Download, Plug, Upload } from 'lucide-react';
import {
  EXACT_RESEARCH_DISCLAIMER,
  PRIVACY_NOTICE_TEXT,
  SupportiveGuidancePanel,
} from '@/src/components/EthicsBanner';
import { PageHeader } from '@/src/components/PageHeader';
import { WorkflowFlowchart } from '@/src/components/WorkflowFlowchart';
import { DatasetAnalyticsPage } from '@/src/pages/DatasetAnalyticsPage';
import { ExplainableAIPage } from '@/src/pages/ExplainableAIPage';
import { ModelPerformancePage } from '@/src/pages/ModelPerformancePage';
import type {
  DatasetSummary,
  EndpointConfig,
  InferenceExplanationBundle,
  LoadedCorpusData,
  NamedEvaluationRun,
  PageId,
  PreprocessingOptions,
  TrainingHistorySummary,
  TransformerBackbone,
} from '@/src/types/mindtrace';
import { preprocessSocialText, previewWordPieceTokens } from '@/src/utils/nlpPipeline';
import { PYTHON_PROJECT_FILES } from '@/src/utils/pythonKitRegistry';

type DetailsTab =
  | 'overview'
  | 'how-it-works'
  | 'models'
  | 'xai'
  | 'dataset'
  | 'performance'
  | 'ethics';

const DETAILS_TABS: { id: DetailsTab; number: string; label: string }[] = [
  { id: 'overview', number: '1', label: 'Project Overview' },
  { id: 'how-it-works', number: '2', label: 'How It Works' },
  { id: 'models', number: '3', label: 'Models Used' },
  { id: 'xai', number: '4', label: 'Explainable AI' },
  { id: 'dataset', number: '5', label: 'Dataset' },
  { id: 'performance', number: '6', label: 'Model Performance' },
  { id: 'ethics', number: '7', label: 'Limitations and Ethics' },
];

const ARCHITECTURE_LAYERS = [
  {
    layer: '01 · Data Ingestion & Leakage Guard',
    module: 'src/data_loader.py',
    detail:
      'Loads the configured research corpus automatically, removes missing/duplicate records prior to splitting, and partitions 70/15/15 stratified.',
  },
  {
    layer: '02 · Validation & PII Anonymization',
    module: 'src/preprocessing.py',
    detail:
      'Applies NFKC normalization, masks URLs ([URL]) and handles ([USER]), and prepares WordPiece tokens capped at 256 subwords.',
  },
  {
    layer: '03 · Transformer Fine-Tuning & Inference',
    module: 'src/train.py · src/model_inference.py',
    detail:
      'Fine-tunes DistilBERT or BERT-base with inverse-frequency class weights and early stopping, outputting logits, probabilities, and entropy.',
  },
  {
    layer: '04 · Post-Hoc Explainable AI Engine',
    module: 'src/explainability.py',
    detail:
      'Runs Partition SHAP and LIME perturbations against the active Transformer classifier and maps subword weights back to words.',
  },
  {
    layer: '05 · Evaluation & Ethical Guardrails',
    module: 'src/evaluation.py · src/ethics.py',
    detail:
      'Computes held-out Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrix, ROC-AUC, and independent India crisis safety guidance.',
  },
];

const REFERENCES = [
  {
    id: '[1]',
    citation:
      'Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS 2017).',
    source: 'https://arxiv.org/abs/1706.03762',
  },
  {
    id: '[2]',
    citation:
      'Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019, 4171–4186.',
    source: 'https://aclanthology.org/N19-1423/',
  },
  {
    id: '[3]',
    citation:
      'Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing.',
    source: 'https://arxiv.org/abs/1910.01108',
  },
  {
    id: '[4]',
    citation:
      'Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). Advances in Neural Information Processing Systems (NeurIPS 2017), 4765–4774.',
    source: 'https://arxiv.org/abs/1705.07874',
  },
  {
    id: '[5]',
    citation:
      'Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME). Proceedings of the 22nd ACM SIGKDD Conference, 1135–1144.',
    source: 'https://arxiv.org/abs/1602.04938',
  },
  {
    id: '[6]',
    citation:
      'Turcan, E., & McKeown, K. (2019). Dreaddit: A Reddit Dataset for Stress Analysis in Social Media. Proceedings of the Tenth International Workshop on Health Text Mining and Information Analysis (LOUHI), EMNLP-IJCNLP.',
    source: 'https://aclanthology.org/D19-6213/',
  },
  {
    id: '[7]',
    citation:
      'Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP System Demonstrations & Hugging Face Model Hub Documentation.',
    source: 'https://huggingface.co/docs/transformers/',
  },
];

interface ProjectDetailsPageProps {
  inputText: string;
  onChangeInputText: (text: string) => void;
  options: PreprocessingOptions;
  onChangeOptions: (options: PreprocessingOptions) => void;
  endpointConfig: EndpointConfig;
  onUpdateEndpointConfig: (config: EndpointConfig) => void;
  activeBundle: InferenceExplanationBundle | null;
  onSetActiveBundle: (bundle: InferenceExplanationBundle | null) => void;
  evaluationRuns: NamedEvaluationRun[];
  activeEvalRunId: string | null;
  onAddOrUpdateRun: (run: NamedEvaluationRun, history?: TrainingHistorySummary) => void;
  onSelectRun: (runId: string) => void;
  onRemoveRun: (runId: string) => void;
  trainingHistory: TrainingHistorySummary | null;
  onSetTrainingHistory: (history: TrainingHistorySummary | null) => void;
  datasetSummary: DatasetSummary | null;
  loadedCorpus: LoadedCorpusData | null;
  loadingDefaultDataset: boolean;
  defaultDatasetError: string | null;
  isCustomDatasetActive: boolean;
  onRetryDefaultDataset: () => void;
  onApplyCustomDataset: (summary: DatasetSummary, corpus: LoadedCorpusData) => void;
  onDownloadPythonKit: () => void;
  downloadingKit: boolean;
  onNavigate: (page: PageId) => void;
}

export function ProjectDetailsPage({
  inputText,
  onChangeInputText,
  options,
  onChangeOptions,
  endpointConfig,
  onUpdateEndpointConfig,
  activeBundle,
  onSetActiveBundle,
  evaluationRuns,
  activeEvalRunId,
  onAddOrUpdateRun,
  onSelectRun,
  onRemoveRun,
  trainingHistory,
  onSetTrainingHistory,
  datasetSummary,
  loadedCorpus,
  loadingDefaultDataset,
  defaultDatasetError,
  isCustomDatasetActive,
  onRetryDefaultDataset,
  onApplyCustomDataset,
  onDownloadPythonKit,
  downloadingKit,
  onNavigate,
}: ProjectDetailsPageProps) {
  const [activeTab, setActiveTab] = useState<DetailsTab>('overview');
  const [switchingModel, setSwitchingModel] = useState(false);
  const [testingBridge, setTestingBridge] = useState(false);
  const [bridgeMsg, setBridgeMsg] = useState<string | null>(null);
  const [selectedPath, setSelectedPath] = useState<string>(PYTHON_PROJECT_FILES[0].path);
  const [copied, setCopied] = useState(false);

  const selectedFile =
    PYTHON_PROJECT_FILES.find((f) => f.path === selectedPath) ?? PYTHON_PROJECT_FILES[0];

  const previewSourceText =
    inputText.trim() ||
    activeBundle?.originalText ||
    'I have been overwhelmed at @work_team lately and barely slept https://example.com';
  const preprocessed = preprocessSocialText(previewSourceText, options);
  const tokenPreview = previewWordPieceTokens(preprocessed.cleanedText);

  async function handleSelectServerModel(modelId: string) {
    setSwitchingModel(true);
    setBridgeMsg(null);
    try {
      const resp = await fetch('/api/model/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelId }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || 'Could not switch active model.');
      }
      onUpdateEndpointConfig({
        ...endpointConfig,
        mode: 'built-in',
        url: '/api',
        status: 'connected',
        activeServerModelId: data.activeModelId || modelId,
        availableServerModels: Array.isArray(data.availableModels)
          ? data.availableModels
          : endpointConfig.availableServerModels,
        detectedLabels: Array.isArray(data.labels) ? data.labels : endpointConfig.detectedLabels,
        lastError: null,
      });
    } catch (err) {
      setBridgeMsg(err instanceof Error ? err.message : 'Failed to switch model.');
    } finally {
      setSwitchingModel(false);
    }
  }

  async function handleSwitchBackendMode(nextMode: 'built-in' | 'external-python') {
    setBridgeMsg(null);
    if (nextMode === 'built-in') {
      try {
        const resp = await fetch('/api/health');
        const data = resp.ok ? await resp.json() : null;
        onUpdateEndpointConfig({
          ...endpointConfig,
          mode: 'built-in',
          url: '/api',
          status: data?.ready === false ? 'unreachable' : 'connected',
          activeServerModelId: data?.activeModelId || endpointConfig.activeServerModelId,
          availableServerModels: Array.isArray(data?.availableModels)
            ? data.availableModels
            : endpointConfig.availableServerModels,
          detectedLabels: Array.isArray(data?.labels) ? data.labels : endpointConfig.detectedLabels,
          lastError: null,
        });
      } catch {
        onUpdateEndpointConfig({
          ...endpointConfig,
          mode: 'built-in',
          url: '/api',
          status: 'unreachable',
          lastError: 'Built-in Transformer backend did not respond.',
        });
      }
    } else {
      onUpdateEndpointConfig({
        ...endpointConfig,
        mode: 'external-python',
        url: endpointConfig.url === '/api' ? 'http://127.0.0.1:8000' : endpointConfig.url,
        status: 'unconfigured',
        lastError: null,
      });
    }
  }

  async function handleTestConnection() {
    setTestingBridge(true);
    setBridgeMsg(null);
    const trimmedUrl = endpointConfig.url.replace(/\/+$/, '');
    try {
      const resp = await fetch(`${trimmedUrl}/health`);
      if (!resp.ok) {
        throw new Error(`Endpoint returned HTTP ${resp.status}`);
      }
      const data = await resp.json();
      if (!data.ready) {
        onUpdateEndpointConfig({
          ...endpointConfig,
          url: trimmedUrl,
          status: 'unreachable',
          lastError: data.detail || 'Python bridge running, but no model checkpoint is loaded.',
        });
      } else {
        onUpdateEndpointConfig({
          ...endpointConfig,
          url: trimmedUrl,
          status: 'connected',
          detectedLabels: Array.isArray(data.labels) ? data.labels : [],
          lastError: null,
        });
        setBridgeMsg('Connected to local Python Transformer bridge.');
      }
    } catch (err) {
      const msg =
        err instanceof Error
          ? `Could not reach ${trimmedUrl}/health (${err.message}).`
          : 'Bridge unreachable.';
      setBridgeMsg(msg);
      onUpdateEndpointConfig({
        ...endpointConfig,
        url: trimmedUrl,
        status: 'unreachable',
        lastError: msg,
      });
    } finally {
      setTestingBridge(false);
    }
  }

  async function handleImportJson(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBridgeMsg(null);
    try {
      const raw = await file.text();
      const data = JSON.parse(raw);
      if (!data.predicted_label || !data.probabilities || !Array.isArray(data.attributions)) {
        throw new Error('JSON must include predicted_label, probabilities, and attributions.');
      }
      const bundle: InferenceExplanationBundle = {
        source: 'imported-json',
        modelName: String(data.model_name || endpointConfig.backbone),
        originalText: String(data.original_text || data.cleaned_text || inputText || ''),
        cleanedText: String(data.cleaned_text || preprocessed.cleanedText || ''),
        predictedLabel: String(data.predicted_label),
        confidence: Number(data.confidence ?? Math.max(...Object.values<number>(data.probabilities))),
        logits: data.logits || undefined,
        probabilities: data.probabilities,
        normalizedEntropy: Number(data.normalized_entropy ?? 0),
        isUncertain: Boolean(data.is_uncertain),
        tokens: Array.isArray(data.tokens) ? data.tokens : [],
        method: data.method === 'LIME' ? 'LIME' : 'SHAP',
        attributions: data.attributions,
        phraseAttributions: Array.isArray(data.phrase_attributions)
          ? data.phrase_attributions
          : undefined,
        naturalLanguageSummary: String(data.natural_language_summary || ''),
        supportiveGuidance: Boolean(data.supportive_guidance),
        confidenceDisclaimer: data.confidence_disclaimer || undefined,
      };
      if (bundle.cleanedText && !inputText.trim()) {
        onChangeInputText(bundle.cleanedText);
      }
      onSetActiveBundle(bundle);
      setActiveTab('xai');
    } catch (err) {
      setBridgeMsg(err instanceof Error ? err.message : 'Failed to parse JSON file.');
    } finally {
      e.target.value = '';
    }
  }

  async function handleCopyFile() {
    await navigator.clipboard.writeText(selectedFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Project Details"
        description="Technical architecture, Transformer models, Explainable AI inspection, dataset analytics, evaluation metrics, and academic documentation for project review and viva."
      />

      {/* 7 Structured Subsection Tabs */}
      <nav aria-label="Project details subsections" className="neu-card p-3">
        <div className="flex flex-wrap gap-2">
          {DETAILS_TABS.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`rounded-2xl px-3.5 py-2 text-xs font-medium transition-all whitespace-nowrap ${
                  active
                    ? 'neu-btn-accent text-[#203653] font-semibold'
                    : 'text-[#526783] hover:text-[#203653] hover:bg-[#EAF0FF]/60'
                }`}
              >
                {tab.number}. {tab.label}
              </button>
            );
          })}
        </div>
      </nav>

      {/* 1. Project Overview */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <section className="neu-card-elevated p-7 space-y-3">
              <div className="font-mono text-xs font-semibold text-[#4D6FAF]">
                1.1 · Project Abstract
              </div>
              <h2 className="font-display text-lg font-semibold text-[#203653]">
                Explainable Mental Health Risk Detection from Social Media Text Using Transformer
                Models
              </h2>
              <p className="text-sm leading-relaxed text-[#526783]">
                MindTrace AI is an undergraduate computer science research prototype that evaluates
                pre-trained NLI Transformers and fine-tunes bidirectional encoder Transformers (
                <code className="font-mono text-xs text-[#203653]">DistilBERT</code> and{' '}
                <code className="font-mono text-xs text-[#203653]">BERT-base</code>) on social
                media corpora to recognize linguistic expressions associated with emotional
                distress. Post-hoc Explainable AI (Partition SHAP and LIME) reveals which words
                drove each prediction alongside uncertainty scores and ethical guardrails.
              </p>
            </section>

            <section className="neu-card p-7 space-y-3">
              <div className="font-mono text-xs font-semibold text-[#4D6FAF]">
                1.2 · Problem Statement &amp; Why Developed
              </div>
              <h2 className="font-display text-lg font-semibold text-[#203653]">
                Making AI Language Analysis Transparent and Safe
              </h2>
              <p className="text-sm leading-relaxed text-[#526783]">
                People express stress and emotional strain online using subtle context, negation,
                and everyday metaphors. Simple keyword lists miss context, while black-box neural
                models output bare predictions without showing why. MindTrace AI bridges accurate
                contextual Transformers with human-readable explanations and strict non-clinical
                boundaries.
              </p>
            </section>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <section className="neu-card p-6 space-y-2.5">
              <h3 className="font-display text-sm font-semibold text-[#203653]">
                Project Objectives
              </h3>
              <ul className="space-y-1.5 text-xs leading-relaxed text-[#526783] list-disc pl-4">
                <li>Prevent train/test leakage with pre-split deduplication and 70/15/15 splits.</li>
                <li>Fine-tune and benchmark DistilBERT vs. BERT-base on identical test sets.</li>
                <li>Provide word-level SHAP and LIME explanations mapped to original text.</li>
                <li>Report real held-out Scikit-learn evaluation metrics with zero synthetic data.</li>
              </ul>
            </section>

            <section className="neu-card p-6 space-y-2.5">
              <h3 className="font-display text-sm font-semibold text-[#203653]">
                Existing System Limitations
              </h3>
              <ul className="space-y-1.5 text-xs leading-relaxed text-[#526783] list-disc pl-4">
                <li>Lexicon &amp; TF-IDF baselines fail on negation (&ldquo;no longer hopeless&rdquo;).</li>
                <li>Standard deep classifiers lack token-level accountability or entropy checks.</li>
                <li>Duplicate posts across splits inflate reported accuracy artificially.</li>
                <li>Many demos lack crisis safety triggers or clear non-diagnostic warnings.</li>
              </ul>
            </section>

            <section className="neu-card p-6 space-y-2.5">
              <h3 className="font-display text-sm font-semibold text-[#203653]">
                Proposed System Highlights
              </h3>
              <ul className="space-y-1.5 text-xs leading-relaxed text-[#526783] list-disc pl-4">
                <li>Calm, guided UI for everyday users separated from technical inspection tabs.</li>
                <li>Automatic backend model loading &amp; persisted held-out metrics.</li>
                <li>Subword-aware SHAP/LIME attribution and normalized Shannon entropy.</li>
                <li>Independent self-harm safety detection with verified India helpline resources.</li>
              </ul>
            </section>
          </div>

          <section className="neu-card p-7 space-y-4">
            <h2 className="font-display text-base font-semibold text-[#203653]">
              System Architecture Layers
            </h2>
            <div className="grid gap-3.5 md:grid-cols-5">
              {ARCHITECTURE_LAYERS.map((item) => (
                <div key={item.layer} className="neu-inset p-4 space-y-1.5">
                  <div className="font-mono text-[11px] font-semibold text-[#203653]">
                    {item.layer}
                  </div>
                  <div className="font-mono text-[11px] text-[#4D6FAF]">{item.module}</div>
                  <p className="text-xs leading-relaxed text-[#526783]">{item.detail}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      {/* 2. How It Works */}
      {activeTab === 'how-it-works' && (
        <div className="space-y-6">
          <section className="neu-card-elevated p-7 space-y-4">
            <div>
              <h2 className="font-display text-base font-semibold text-[#203653]">
                End-to-End Processing Flowchart
              </h2>
              <p className="text-xs text-[#526783] mt-1">
                Every message submitted in Check Your Text passes through this 6-stage pipeline.
              </p>
            </div>
            <WorkflowFlowchart />
          </section>

          <section className="neu-card p-7 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Live Preprocessing &amp; WordPiece Tokenization Inspector
                </h2>
                <p className="text-xs text-[#526783]">
                  Shows how NFKC normalization, PII masking, and subword tokenization transform the
                  current session text before it enters the Transformer encoder.
                </p>
              </div>
              <span className="font-mono text-xs text-[#4D6FAF]">
                URLs masked: {preprocessed.urlsMasked} · Mentions masked: {preprocessed.mentionsMasked}
              </span>
            </div>

            <div className="neu-inset p-4 text-xs text-[#203653] font-mono break-words">
              {preprocessed.cleanedText}
            </div>

            <div className="flex flex-wrap gap-1.5 font-mono text-xs">
              {tokenPreview.tokens.map((tok, idx) => (
                <span
                  key={`${tok}-${idx}`}
                  className="rounded-xl bg-[#EAF0FF] border border-[#D2DFFF] px-2.5 py-1 text-[#203653]"
                >
                  {tok}
                </span>
              ))}
            </div>

            <div className="pt-3 border-t border-[#AFC5F5]/55 flex flex-wrap gap-x-6 gap-y-2.5 text-xs text-[#526783]">
              {(
                [
                  ['maskUrls', 'Mask URLs ([URL])'],
                  ['anonymizeMentions', 'Anonymize @mentions ([USER])'],
                  ['stripHashtagSymbol', 'Strip # symbol'],
                  ['normalizeWhitespace', 'Normalize whitespace'],
                  ['lowercase', 'Lowercase'],
                ] as [keyof PreprocessingOptions, string][]
              ).map(([key, label]) => (
                <label key={key} className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={options[key]}
                    onChange={(e) => onChangeOptions({ ...options, [key]: e.target.checked })}
                    className="h-4 w-4 rounded border-[#AFC5F5] accent-[#4D6FAF]"
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </section>

          <details className="group neu-card">
            <summary className="flex cursor-pointer list-none items-center justify-between px-7 py-4 text-xs font-semibold text-[#203653] select-none">
              <span>Backend Bridge Configuration &amp; Offline JSON Prediction Import</span>
              <ChevronDown className="h-4 w-4 text-[#4D6FAF] transition-transform group-open:rotate-180" />
            </summary>

            <div className="border-t border-[#AFC5F5]/55 p-7 grid gap-6 md:grid-cols-2 text-xs">
              <div className="space-y-3">
                <div className="font-semibold text-[#203653]">Inference Backend Mode</div>
                <div className="neu-inset grid grid-cols-2 rounded-2xl p-1 gap-1">
                  <button
                    type="button"
                    onClick={() => handleSwitchBackendMode('built-in')}
                    className={`rounded-xl px-2.5 py-2 font-medium transition-all ${
                      endpointConfig.mode === 'built-in'
                        ? 'neu-btn-accent text-[#203653]'
                        : 'text-[#526783] hover:text-[#203653]'
                    }`}
                  >
                    Built-In (/api)
                  </button>
                  <button
                    type="button"
                    onClick={() => handleSwitchBackendMode('external-python')}
                    className={`rounded-xl px-2.5 py-2 font-medium transition-all ${
                      endpointConfig.mode === 'external-python'
                        ? 'neu-btn-accent text-[#203653]'
                        : 'text-[#526783] hover:text-[#203653]'
                    }`}
                  >
                    External Python Bridge
                  </button>
                </div>

                {endpointConfig.mode === 'external-python' && (
                  <div className="space-y-2.5 pt-1">
                    <select
                      aria-label="Python Bridge Backbone"
                      value={endpointConfig.backbone}
                      onChange={(e) =>
                        onUpdateEndpointConfig({
                          ...endpointConfig,
                          backbone: e.target.value as TransformerBackbone,
                        })
                      }
                      className="neu-input w-full px-3 py-2 text-xs text-[#203653]"
                    >
                      <option value="distilbert-base-uncased">
                        DistilBERT (distilbert-base-uncased)
                      </option>
                      <option value="bert-base-uncased">BERT Base (bert-base-uncased)</option>
                    </select>
                    <div className="flex gap-2">
                      <input
                        type="url"
                        value={endpointConfig.url}
                        onChange={(e) =>
                          onUpdateEndpointConfig({ ...endpointConfig, url: e.target.value })
                        }
                        className="neu-input flex-1 min-w-0 px-3 py-2 text-xs font-mono text-[#203653]"
                      />
                      <button
                        type="button"
                        onClick={handleTestConnection}
                        disabled={testingBridge}
                        className="neu-btn inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium whitespace-nowrap"
                      >
                        <Plug className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
                        <span>{testingBridge ? 'Testing...' : 'Connect'}</span>
                      </button>
                    </div>
                  </div>
                )}

                {bridgeMsg && <div className="text-xs text-[#4D6FAF]">{bridgeMsg}</div>}
              </div>

              <div className="space-y-2.5">
                <div className="font-semibold text-[#203653]">
                  Optional Offline Prediction JSON Import
                </div>
                <p className="text-[#526783]">
                  Inspect a JSON prediction bundle exported from offline Python CLI tools.
                </p>
                <label className="neu-btn inline-flex cursor-pointer items-center gap-2 px-4 py-2 font-medium">
                  <Upload className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
                  <span>Import Prediction JSON</span>
                  <input
                    type="file"
                    accept=".json,application/json"
                    onChange={handleImportJson}
                    className="sr-only"
                  />
                </label>
              </div>
            </div>
          </details>
        </div>
      )}

      {/* 3. Models Used */}
      {activeTab === 'models' && (
        <div className="space-y-6">
          <section className="neu-card-elevated p-7 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Active Transformer Model Selector
                </h2>
                <p className="text-xs text-[#526783] mt-1">
                  Select which built-in Transformer checkpoint powers live inference on Check Your
                  Text.
                </p>
              </div>
              {endpointConfig.mode === 'built-in' && (
                <select
                  aria-label="Active built-in Transformer model"
                  value={endpointConfig.activeServerModelId}
                  disabled={switchingModel}
                  onChange={(e) => handleSelectServerModel(e.target.value)}
                  className="neu-input px-3.5 py-2.5 text-xs text-[#203653] max-w-sm"
                >
                  {endpointConfig.availableServerModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              )}
            </div>
            {bridgeMsg && <div className="text-xs text-[#6B2A3A]">{bridgeMsg}</div>}
          </section>

          <div className="grid gap-6 lg:grid-cols-3">
            <section className="neu-card p-6 space-y-2.5">
              <div className="font-mono text-xs text-[#4D6FAF]">Default Zero-Shot NLI</div>
              <h3 className="font-display text-base font-semibold text-[#203653]">
                DeBERTa-v3-xsmall NLI
              </h3>
              <p className="text-xs leading-relaxed text-[#526783]">
                Runs immediately on startup via ONNX Runtime without waiting for local PyTorch
                fine-tuning. Evaluates candidate text categories using Natural Language Inference
                entailment scoring.
              </p>
            </section>

            <section className="neu-card p-6 space-y-2.5">
              <div className="font-mono text-xs text-[#4D6FAF]">Distilled Encoder</div>
              <h3 className="font-display text-base font-semibold text-[#203653]">
                DistilBERT (distilbert-base-uncased)
              </h3>
              <p className="text-xs leading-relaxed text-[#526783]">
                6 Transformer encoder layers (~66M parameters) distilled from BERT-base. Retains
                strong contextual sensitivity to negation and modifiers at nearly double the speed.
              </p>
            </section>

            <section className="neu-card p-6 space-y-2.5">
              <div className="font-mono text-xs text-[#4D6FAF]">Full Encoder Benchmark</div>
              <h3 className="font-display text-base font-semibold text-[#203653]">
                BERT-base (bert-base-uncased)
              </h3>
              <p className="text-xs leading-relaxed text-[#526783]">
                12 bidirectional self-attention layers, 12 heads (~110M parameters). Fine-tuned on
                the identical 70/15/15 split for direct capacity-vs-speed comparison against
                DistilBERT.
              </p>
            </section>
          </div>

          <section className="neu-card p-7 space-y-3">
            <h3 className="font-display text-base font-semibold text-[#203653]">
              Why Bidirectional Self-Attention Matters
            </h3>
            <p className="text-xs leading-relaxed text-[#526783]">
              Unlike bag-of-words or TF-IDF counters that look at words in isolation, encoder
              Transformers attend across every token pair simultaneously. The pooled{' '}
              <code className="font-mono text-[#203653]">[CLS]</code> representation passes through
              dropout and a linear head to yield logits <code className="font-mono">z</code>, which
              are normalized into probabilities with <code className="font-mono">softmax(z)</code>{' '}
              and checked via normalized Shannon entropy for ambiguity.
            </p>
          </section>
        </div>
      )}

      {/* 4. Explainable AI */}
      {activeTab === 'xai' && (
        <div className="space-y-6">
          <section className="neu-card p-7 space-y-4">
            <h2 className="font-display text-base font-semibold text-[#203653]">
              How SHAP and LIME Identify Influential Words
            </h2>
            <p className="text-xs leading-relaxed text-[#526783]">
              Transformer weights span tens of millions of parameters, so Explainable AI measures
              influence directly by masking or perturbing words in the input and watching how the
              model&apos;s output score shifts.
            </p>
            <div className="neu-inset p-4 space-y-2 text-xs text-[#526783]">
              <div className="font-semibold text-[#203653]">Concrete Example</div>
              <p>
                Suppose a note reads: <em>&ldquo;I feel exhausted and overwhelmed at work.&rdquo;</em>{' '}
                When Partition SHAP temporarily masks <em>&ldquo;exhausted&rdquo;</em> and{' '}
                <em>&ldquo;overwhelmed&rdquo;</em>, the score toward emotional strain drops sharply—so
                those words receive positive supporting attribution weights. Everyday words like{' '}
                <em>&ldquo;at&rdquo;</em> or <em>&ldquo;work&rdquo;</em> change the score very
                little, receiving near-zero weights.
              </p>
            </div>
          </section>

          <ExplainableAIPage
            embedded
            activeBundle={activeBundle}
            onSetActiveBundle={(b) => onSetActiveBundle(b)}
            endpointConfig={endpointConfig}
            preprocessingOptions={options}
            onNavigate={onNavigate}
          />
        </div>
      )}

      {/* 5. Dataset */}
      {activeTab === 'dataset' && (
        <div className="space-y-6">
          <section className="neu-card p-7 space-y-3">
            <h2 className="font-display text-base font-semibold text-[#203653]">
              Benchmark Dataset Purpose, Splitting &amp; Caveats
            </h2>
            <p className="text-xs leading-relaxed text-[#526783]">
              MindTrace AI evaluates models against the configured social media stress research
              corpus (inspired by Dreaddit Reddit stress-analysis benchmarks). Because public forum
              labels are community-sourced rather than clinician-adjudicated diagnoses, predictions
              reflect learned linguistic patterns within that dataset rather than medical truth.
              Exact duplicate texts are removed <strong>before</strong> stratified 70/15/15 splitting
              to prevent train-test leakage.
            </p>
          </section>

          <DatasetAnalyticsPage
            embedded
            datasetSummary={datasetSummary}
            loadingDefaultDataset={loadingDefaultDataset}
            defaultDatasetError={defaultDatasetError}
            isCustomDatasetActive={isCustomDatasetActive}
            onRetryDefaultDataset={onRetryDefaultDataset}
            onApplyCustomDataset={onApplyCustomDataset}
            onOpenModelPerformance={() => setActiveTab('performance')}
          />
        </div>
      )}

      {/* 6. Model Performance */}
      {activeTab === 'performance' && (
        <div className="space-y-6">
          <section className="neu-card p-7 space-y-3">
            <h2 className="font-display text-base font-semibold text-[#203653]">
              Understanding the Evaluation Metrics
            </h2>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 text-xs text-[#526783]">
              <div className="neu-inset p-3.5">
                <div className="font-semibold text-[#203653]">Accuracy</div>
                <p className="mt-1">Share of held-out test posts predicted correctly overall.</p>
              </div>
              <div className="neu-inset p-3.5">
                <div className="font-semibold text-[#203653]">Precision &amp; Recall</div>
                <p className="mt-1">
                  Precision measures false-alarm control; Recall measures how many real cases were
                  caught.
                </p>
              </div>
              <div className="neu-inset p-3.5">
                <div className="font-semibold text-[#203653]">Macro &amp; Weighted F1</div>
                <p className="mt-1">
                  Macro F1 gives every category equal weight; Weighted F1 accounts for class sizes.
                </p>
              </div>
              <div className="neu-inset p-3.5">
                <div className="font-semibold text-[#203653]">Evaluation Caveat</div>
                <p className="mt-1">
                  Metrics reflect performance on the sampled held-out benchmark split—never real-world
                  clinical accuracy.
                </p>
              </div>
            </div>
          </section>

          <ModelPerformancePage
            embedded
            evaluationRuns={evaluationRuns}
            activeRunId={activeEvalRunId}
            onAddOrUpdateRun={onAddOrUpdateRun}
            onSelectRun={onSelectRun}
            onRemoveRun={onRemoveRun}
            trainingHistory={trainingHistory}
            onSetTrainingHistory={onSetTrainingHistory}
            loadedCorpus={loadedCorpus}
            endpointConfig={endpointConfig}
            onUpdateEndpointConfig={onUpdateEndpointConfig}
          />
        </div>
      )}

      {/* 7. Limitations and Ethics */}
      {activeTab === 'ethics' && (
        <div className="space-y-6">
          <section className="neu-card-elevated p-7 space-y-4">
            <h2 className="font-display text-base font-semibold text-[#203653]">
              Non-Diagnostic Boundary &amp; Privacy Governance
            </h2>
            <div className="neu-inset p-4 text-xs leading-relaxed text-[#203653] space-y-2">
              <div>
                <strong>Educational Research Disclaimer: </strong>
                {EXACT_RESEARCH_DISCLAIMER}
              </div>
              <div className="text-[#526783]">{PRIVACY_NOTICE_TEXT}</div>
            </div>
          </section>

          <div className="grid gap-6 lg:grid-cols-2">
            <section className="neu-card p-7 space-y-2.5">
              <h3 className="font-display text-base font-semibold text-[#203653]">
                Known Technical &amp; Domain Limitations
              </h3>
              <ul className="space-y-2 text-xs leading-relaxed text-[#526783] list-disc pl-4">
                <li>
                  <strong className="text-[#203653]">Sarcasm, Metaphor &amp; Quoting:</strong> Song
                  lyrics or hyperbole can trigger false positives.
                </li>
                <li>
                  <strong className="text-[#203653]">Platform &amp; Dialect Shift:</strong> Models
                  reflect training forum vocabulary and degrade on unfamiliar dialects.
                </li>
                <li>
                  <strong className="text-[#203653]">256-Token Horizon:</strong> Very long notes are
                  truncated to fit encoder sequence limits.
                </li>
                <li>
                  <strong className="text-[#203653]">Softmax Calibration:</strong> Raw softmax scores
                  measure classifier confidence, not clinical probability.
                </li>
              </ul>
            </section>

            <section className="neu-card p-7 space-y-2.5">
              <h3 className="font-display text-base font-semibold text-[#203653]">
                Academic References
              </h3>
              <ol className="space-y-2 text-xs leading-relaxed text-[#526783]">
                {REFERENCES.map((ref) => (
                  <li key={ref.id} className="neu-inset p-3">
                    <span className="font-mono font-semibold text-[#203653] mr-1.5">{ref.id}</span>
                    <span>{ref.citation}</span>{' '}
                    <span className="font-mono text-[11px] text-[#4D6FAF] break-all">
                      {ref.source}
                    </span>
                  </li>
                ))}
              </ol>
            </section>
          </div>

          <details className="group neu-card">
            <summary className="flex cursor-pointer list-none items-center justify-between px-7 py-5 text-xs font-semibold text-[#203653] select-none">
              <span>Modular Python &amp; Streamlit Source Code Explorer / ZIP Export</span>
              <ChevronDown className="h-4 w-4 text-[#4D6FAF] transition-transform group-open:rotate-180" />
            </summary>

            <div className="border-t border-[#AFC5F5]/55 p-7 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-4 border-b border-[#AFC5F5]/55">
                <p className="text-xs text-[#526783]">
                  Inspect every Python module (PyTorch, Hugging Face, SHAP/LIME, Streamlit) or
                  download the complete source ZIP.
                </p>
                <button
                  type="button"
                  onClick={onDownloadPythonKit}
                  disabled={downloadingKit}
                  className="neu-btn-primary inline-flex items-center gap-2 px-4 py-2 text-xs disabled:opacity-60 whitespace-nowrap shrink-0 min-h-[40px]"
                >
                  <Download className="h-3.5 w-3.5 text-white" aria-hidden="true" />
                  <span>{downloadingKit ? 'Packaging ZIP...' : 'Download Full Python ZIP'}</span>
                </button>
              </div>

              <div className="grid gap-6 lg:grid-cols-12">
                <div className="lg:col-span-4 space-y-2 max-h-[420px] overflow-y-auto pr-1">
                  {PYTHON_PROJECT_FILES.map((file) => {
                    const active = file.path === selectedFile.path;
                    return (
                      <button
                        key={file.path}
                        type="button"
                        onClick={() => setSelectedPath(file.path)}
                        className={`w-full text-left px-3.5 py-2.5 transition-all ${
                          active ? 'neu-btn-accent' : 'neu-btn'
                        }`}
                      >
                        <div className="font-mono text-xs font-medium text-[#203653]">
                          {file.path}
                        </div>
                        <div className="mt-0.5 text-[11px] leading-snug text-[#526783]">
                          {file.role}
                        </div>
                      </button>
                    );
                  })}
                </div>

                <div className="lg:col-span-8 flex flex-col neu-inset overflow-hidden p-0">
                  <div className="flex items-center justify-between bg-[#AFC5F5]/45 px-4 py-3 border-b border-[#D2DFFF]">
                    <span className="font-mono text-xs font-medium text-[#203653]">
                      {selectedFile.path}
                    </span>
                    <button
                      type="button"
                      onClick={handleCopyFile}
                      className="neu-btn inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium"
                    >
                      {copied ? (
                        <>
                          <Check className="h-3.5 w-3.5 text-[#1E4D3B]" aria-hidden="true" />
                          <span>Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <pre className="p-4 bg-[#D5E1FC] text-[#203653] font-mono text-xs leading-relaxed overflow-auto max-h-[380px]">
                    <code>{selectedFile.content}</code>
                  </pre>
                </div>
              </div>
            </div>
          </details>

          <SupportiveGuidancePanel />
        </div>
      )}
    </div>
  );
}
