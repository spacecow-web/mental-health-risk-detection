import { useState, type ChangeEvent } from 'react';
import { ArrowRight, Eraser, Plug, Upload } from 'lucide-react';
import { SupportiveGuidancePanel } from '@/src/components/EthicsBanner';
import { PageHeader } from '@/src/components/PageHeader';
import type {
  EndpointConfig,
  ExplanationMethod,
  InferenceExplanationBundle,
  PageId,
  PreprocessingOptions,
  TransformerBackbone,
} from '@/src/types/mindtrace';
import {
  DEFAULT_PREPROCESSING_OPTIONS,
  detectSupportiveCues,
  preprocessSocialText,
  previewWordPieceTokens,
  validateInputText,
} from '@/src/utils/nlpPipeline';

interface TextAnalysisPageProps {
  inputText: string;
  onChangeInputText: (text: string) => void;
  endpointConfig: EndpointConfig;
  onUpdateEndpointConfig: (config: EndpointConfig) => void;
  activeBundle: InferenceExplanationBundle | null;
  onSetActiveBundle: (bundle: InferenceExplanationBundle | null) => void;
  onNavigate: (page: PageId) => void;
}

export function TextAnalysisPage({
  inputText,
  onChangeInputText,
  endpointConfig,
  onUpdateEndpointConfig,
  activeBundle,
  onSetActiveBundle,
  onNavigate,
}: TextAnalysisPageProps) {
  const [options, setOptions] = useState<PreprocessingOptions>(DEFAULT_PREPROCESSING_OPTIONS);
  const [xaiMethod, setXaiMethod] = useState<ExplanationMethod>('SHAP');
  const [hasAnalysed, setHasAnalysed] = useState(false);
  const [testingBridge, setTestingBridge] = useState(false);
  const [runningInference, setRunningInference] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const validation = validateInputText(inputText);
  const preprocessed = preprocessSocialText(inputText, options);
  const tokenPreview = previewWordPieceTokens(preprocessed.cleanedText);
  const showSupportivePanel = detectSupportiveCues(
    inputText,
    activeBundle?.predictedLabel ?? ''
  );

  async function handleTestConnection() {
    setTestingBridge(true);
    setActionError(null);
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
          lastError: data.detail || 'Bridge is running, but no trained model checkpoint is loaded.',
        });
      } else {
        onUpdateEndpointConfig({
          ...endpointConfig,
          url: trimmedUrl,
          status: 'connected',
          detectedLabels: Array.isArray(data.labels) ? data.labels : [],
          lastError: null,
        });
      }
    } catch (err) {
      onUpdateEndpointConfig({
        ...endpointConfig,
        url: trimmedUrl,
        status: 'unreachable',
        lastError:
          err instanceof Error
            ? `Could not reach ${trimmedUrl}/health (${err.message}). Start the local Python bridge to run live PyTorch inference.`
            : 'Bridge unreachable.',
      });
    } finally {
      setTestingBridge(false);
    }
  }

  async function handleAnalyseText() {
    setHasAnalysed(true);
    setActionError(null);
    if (!validation.isValid) {
      return;
    }

    if (endpointConfig.status !== 'connected') {
      return;
    }

    setRunningInference(true);
    const trimmedUrl = endpointConfig.url.replace(/\/+$/, '');
    try {
      const resp = await fetch(`${trimmedUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          method: xaiMethod,
          mask_urls: options.maskUrls,
          anonymize_mentions: options.anonymizeMentions,
          lowercase: options.lowercase,
        }),
      });
      if (!resp.ok) {
        const errPayload = await resp.json().catch(() => ({}));
        throw new Error(errPayload.detail || `Inference failed with HTTP ${resp.status}`);
      }
      const data = await resp.json();
      const bundle: InferenceExplanationBundle = {
        source: 'local-api',
        modelName: String(data.model_name || endpointConfig.backbone),
        originalText: String(data.original_text || inputText),
        cleanedText: String(data.cleaned_text || preprocessed.cleanedText),
        predictedLabel: String(data.predicted_label),
        confidence: Number(data.confidence),
        logits: data.logits || undefined,
        probabilities: data.probabilities || {},
        normalizedEntropy: Number(data.normalized_entropy ?? 0),
        isUncertain: Boolean(data.is_uncertain),
        tokens: Array.isArray(data.tokens) ? data.tokens : tokenPreview.tokens,
        method: data.method === 'LIME' ? 'LIME' : 'SHAP',
        attributions: Array.isArray(data.attributions) ? data.attributions : [],
        naturalLanguageSummary: String(data.natural_language_summary || ''),
        supportiveGuidance: Boolean(data.supportive_guidance),
        confidenceDisclaimer: data.confidence_disclaimer || undefined,
      };
      onSetActiveBundle(bundle);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Inference request failed.');
    } finally {
      setRunningInference(false);
    }
  }

  async function handleImportJson(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setActionError(null);
    try {
      const raw = await file.text();
      const data = JSON.parse(raw);
      if (!data.predicted_label || !data.probabilities || !Array.isArray(data.attributions)) {
        throw new Error(
          'JSON must include predicted_label, probabilities, and attributions exported from MindTrace Python.'
        );
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
        naturalLanguageSummary: String(data.natural_language_summary || ''),
        supportiveGuidance: Boolean(data.supportive_guidance),
        confidenceDisclaimer: data.confidence_disclaimer || undefined,
      };
      if (bundle.cleanedText && !inputText.trim()) {
        onChangeInputText(bundle.cleanedText);
      }
      onSetActiveBundle(bundle);
      setHasAnalysed(true);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to parse JSON file.');
    } finally {
      e.target.value = '';
    }
  }

  const topPreviewTokens = activeBundle
    ? [...activeBundle.attributions]
        .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
        .slice(0, 6)
    : [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Text Analysis"
        description="Validate and preprocess social media text, inspect subword tokenization, and view Transformer predictions, confidence distributions, and explanation previews."
      />

      {/* Large Text Input & Primary Analyse Text Action */}
      <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Social Media Text Input
            </h2>
            <p className="text-xs text-[#64748B]">
              Processed strictly in ephemeral memory · URLs and @handles can be masked prior to tokenization
            </p>
          </div>
          {inputText && (
            <button
              type="button"
              onClick={() => {
                onChangeInputText('');
                onSetActiveBundle(null);
                setHasAnalysed(false);
                setActionError(null);
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E8F0] px-3 py-1.5 text-xs font-medium text-[#475569] hover:bg-[#F1F5F9] min-h-[36px]"
            >
              <Eraser className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Clear Input</span>
            </button>
          )}
        </div>

        <div>
          <label htmlFor="social-text-input" className="sr-only">
            Social media text input
          </label>
          <textarea
            id="social-text-input"
            rows={7}
            value={inputText}
            onChange={(e) => {
              onChangeInputText(e.target.value);
              if (hasAnalysed) setHasAnalysed(false);
            }}
            placeholder="Paste or type a social media post here (minimum 10 characters) to validate input, anonymize identifiers, inspect WordPiece subwords, and run Transformer inference..."
            className="w-full min-h-[175px] rounded-lg border border-[#CBD5E1] bg-[#F8FAFC] p-4 text-sm leading-relaxed text-[#0F172A] placeholder:text-[#94A3B8] focus:border-[#2563EB] focus:bg-white focus:outline-none"
          />
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-[#64748B] font-mono tabular-nums">
            <span>
              {validation.charCount} chars · {validation.wordCount} words ·{' '}
              {tokenPreview.subwordCount} subword tokens
            </span>
            <span>
              Validation: {validation.isValid ? 'Valid' : 'Awaiting valid input (min 10 chars)'}
            </span>
          </div>
        </div>

        {/* Input Validation Feedback */}
        {hasAnalysed && !validation.isValid && (
          <div role="alert" className="rounded-lg bg-[#FEF2F2] px-4 py-3 text-xs text-[#991B1B]">
            {validation.errorMessage}
          </div>
        )}
        {validation.warnings.map((warn) => (
          <div key={warn} className="rounded-lg bg-[#FFFBEB] px-4 py-3 text-xs text-[#92400E]">
            {warn}
          </div>
        ))}

        {/* Preprocessing Controls + Analyse Text Button */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pt-3 border-t border-[#E2E8F0]">
          <div className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-[#334155]">
            {(
              [
                ['maskUrls', 'Mask URLs'],
                ['anonymizeMentions', 'Anonymize @mentions'],
                ['stripHashtagSymbol', 'Strip # symbol'],
                ['normalizeWhitespace', 'Normalize whitespace'],
                ['lowercase', 'Lowercase'],
              ] as [keyof PreprocessingOptions, string][]
            ).map(([key, label]) => (
              <label key={key} className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={options[key]}
                  onChange={(e) =>
                    setOptions((prev) => ({ ...prev, [key]: e.target.checked }))
                  }
                  className="h-4 w-4 rounded border-[#CBD5E1] accent-[#0F2942]"
                />
                <span>{label}</span>
              </label>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-3 shrink-0">
            <button
              type="button"
              onClick={handleAnalyseText}
              disabled={runningInference}
              className="inline-flex items-center justify-center rounded-lg bg-[#0F2942] px-5 py-2.5 text-sm font-medium text-white hover:bg-[#173A5E] disabled:opacity-50 transition-colors whitespace-nowrap min-h-[42px]"
            >
              {runningInference ? 'Analysing Text...' : 'Analyse Text'}
            </button>
          </div>
        </div>
      </section>

      {/* Preprocessed Output & Tokenization Preview */}
      {(hasAnalysed || inputText.trim().length > 0) && validation.isValid && (
        <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Preprocessing & Subword Tokenization
            </h2>
            <span className="font-mono text-xs tabular-nums text-[#64748B]">
              URLs masked: {preprocessed.urlsMasked} · Mentions masked: {preprocessed.mentionsMasked}
            </span>
          </div>
          <div className="rounded-lg bg-[#F8FAFC] p-3.5 text-sm text-[#1E293B] break-words">
            {preprocessed.cleanedText}
          </div>
          <div className="flex flex-wrap gap-1.5 font-mono text-xs">
            {tokenPreview.tokens.map((tok, idx) => (
              <span
                key={`${tok}-${idx}`}
                className="rounded bg-[#F1F5F9] px-2 py-0.5 text-[#0F2942]"
              >
                {tok}
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Prediction Result, Confidence Visualisation, Explanation Preview */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Prediction Result + Bridge Setup */}
        <section className="lg:col-span-4 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
          <h2 className="font-display text-base font-semibold text-[#0F172A]">
            Prediction Result
          </h2>

          {activeBundle ? (
            <div className="space-y-4">
              <div className="rounded-lg bg-[#F8FAFC] p-4 space-y-2">
                <div className="text-xs text-[#64748B]">Predicted Category</div>
                <div className="font-display text-xl font-bold text-[#0F2942]">
                  {activeBundle.predictedLabel}
                </div>
                <div className="flex items-center justify-between text-xs pt-2 border-t border-[#E2E8F0]">
                  <span className="text-[#64748B]">Softmax Confidence</span>
                  <span className="font-mono font-semibold tabular-nums text-[#0F172A]">
                    {(activeBundle.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[#64748B]">Normalized Entropy</span>
                  <span className="font-mono font-semibold tabular-nums text-[#0F172A]">
                    {activeBundle.normalizedEntropy.toFixed(3)}
                  </span>
                </div>
              </div>

              {activeBundle.isUncertain && (
                <div className="rounded-lg bg-[#FFFBEB] p-3 text-xs text-[#92400E]">
                  High predictive entropy indicates substantial class ambiguity.
                </div>
              )}

              <p className="text-[11px] leading-relaxed text-[#475569]">
                {activeBundle.confidenceDisclaimer ||
                  'Model confidence is a normalized softmax score over classifier logits—it is NOT the probability that a person has a mental health condition.'}
              </p>
            </div>
          ) : (
            <div className="rounded-lg bg-[#F8FAFC] p-4 text-xs leading-relaxed text-[#475569] space-y-2">
              <div className="font-semibold text-[#0F172A]">No Model Checkpoint Connected</div>
              <p>
                Predictions are never simulated. Connect your local Python Transformer bridge below
                or import a prediction JSON generated by the Python pipeline.
              </p>
            </div>
          )}

          <div className="border-t border-[#E2E8F0] pt-4 space-y-3">
            <div>
              <label htmlFor="backbone-select" className="block text-xs font-medium text-[#334155] mb-1">
                Transformer Backbone
              </label>
              <select
                id="backbone-select"
                value={endpointConfig.backbone}
                onChange={(e) =>
                  onUpdateEndpointConfig({
                    ...endpointConfig,
                    backbone: e.target.value as TransformerBackbone,
                  })
                }
                className="w-full rounded-lg border border-[#CBD5E1] bg-white px-3 py-2 text-xs text-[#0F172A]"
              >
                <option value="distilbert-base-uncased">DistilBERT (distilbert-base-uncased)</option>
                <option value="bert-base-uncased">BERT Base (bert-base-uncased)</option>
              </select>
            </div>

            <div>
              <label htmlFor="bridge-url" className="block text-xs font-medium text-[#334155] mb-1">
                Local Python Bridge URL
              </label>
              <div className="flex gap-2">
                <input
                  id="bridge-url"
                  type="url"
                  value={endpointConfig.url}
                  onChange={(e) =>
                    onUpdateEndpointConfig({ ...endpointConfig, url: e.target.value })
                  }
                  className="flex-1 min-w-0 rounded-lg border border-[#CBD5E1] bg-white px-2.5 py-2 text-xs font-mono text-[#0F172A]"
                />
                <button
                  type="button"
                  onClick={handleTestConnection}
                  disabled={testingBridge}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[#CBD5E1] bg-white px-3 py-2 text-xs font-medium text-[#0F172A] hover:bg-[#F1F5F9] whitespace-nowrap"
                >
                  <Plug className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>{testingBridge ? 'Testing...' : 'Connect'}</span>
                </button>
              </div>
            </div>

            {endpointConfig.status === 'connected' && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#475569]">XAI Method</span>
                <div className="inline-flex rounded-lg bg-[#F1F5F9] p-0.5">
                  {(['SHAP', 'LIME'] as ExplanationMethod[]).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setXaiMethod(m)}
                      className={`rounded-md px-2.5 py-1 text-xs font-medium ${
                        xaiMethod === m ? 'bg-white text-[#0F2942]' : 'text-[#64748B]'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {endpointConfig.lastError && (
              <div className="text-xs text-[#64748B]">{endpointConfig.lastError}</div>
            )}

            <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-[#CBD5E1] bg-[#F8FAFC] px-3 py-2 text-xs font-medium text-[#0F2942] hover:bg-[#F1F5F9] transition-colors">
              <Upload className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Import Prediction JSON</span>
              <input
                type="file"
                accept=".json,application/json"
                onChange={handleImportJson}
                className="sr-only"
              />
            </label>

            {actionError && (
              <div className="rounded-lg bg-[#FEF2F2] p-2.5 text-xs text-[#991B1B]">{actionError}</div>
            )}
          </div>
        </section>

        {/* Confidence Visualisation */}
        <section className="lg:col-span-4 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
          <h2 className="font-display text-base font-semibold text-[#0F172A]">
            Confidence Visualisation
          </h2>

          {!activeBundle ? (
            <div className="rounded-lg bg-[#F8FAFC] p-5 text-xs leading-relaxed text-[#475569]">
              <div className="font-semibold text-[#0F172A]">Empty State</div>
              <p className="mt-1">
                Per-class softmax probability bars will render here once a trained DistilBERT or
                BERT checkpoint evaluates the text.
              </p>
            </div>
          ) : (
            <div className="space-y-3.5">
              <p className="text-xs text-[#64748B]">
                Calibrated softmax probability distribution across supported label categories.
              </p>
              {Object.entries(activeBundle.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([cls, prob]) => {
                  const isTop = cls === activeBundle.predictedLabel;
                  return (
                    <div key={cls} className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className={isTop ? 'font-semibold text-[#0F2942]' : 'text-[#334155]'}>
                          {cls}
                        </span>
                        <span className="font-mono tabular-nums text-[#0F172A]">
                          {activeBundle.logits?.[cls] !== undefined
                            ? `z=${activeBundle.logits[cls].toFixed(2)} · `
                            : ''}
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-3 w-full rounded-full bg-[#F1F5F9] overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isTop ? 'bg-[#0F2942]' : 'bg-[#94A3B8]'
                          }`}
                          style={{ width: `${Math.max(2, prob * 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </section>

        {/* Explanation Preview */}
        <section className="lg:col-span-4 rounded-xl border border-[#E2E8F0] bg-white p-6 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Explanation Preview
            </h2>

            {!activeBundle ? (
              <div className="rounded-lg bg-[#F8FAFC] p-5 text-xs leading-relaxed text-[#475569]">
                <div className="font-semibold text-[#0F172A]">Awaiting Attribution Data</div>
                <p className="mt-1">
                  Influential words and positive/negative SHAP or LIME contributions appear here
                  after model inference or JSON import.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="text-xs text-[#64748B]">
                  Top influential tokens ({activeBundle.method})
                </div>
                <div className="space-y-2">
                  {topPreviewTokens.map((item, idx) => (
                    <div
                      key={`${item.token}-${idx}`}
                      className="flex items-center justify-between rounded-lg bg-[#F8FAFC] px-3 py-2 text-xs"
                    >
                      <span className="font-mono text-[#0F172A]">{item.token}</span>
                      <span
                        className={`font-mono tabular-nums font-medium ${
                          item.weight >= 0 ? 'text-[#0D7377]' : 'text-[#B45309]'
                        }`}
                      >
                        {item.weight >= 0 ? '+' : ''}
                        {item.weight.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => onNavigate('explainable-ai')}
            className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-[#0D7377] bg-white px-4 py-2 text-xs font-semibold text-[#0D7377] hover:bg-[#F0FDFA] transition-colors min-h-[40px]"
          >
            <span>Open Explainable AI View</span>
            <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </section>
      </div>

      {/* Explicit Ethical & Non-Diagnostic Disclaimer */}
      <section className="rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] p-5 text-xs leading-relaxed text-[#475569]">
        <strong className="text-[#0F172A]">Ethical & Clinical Disclaimer: </strong>
        Automated text classification is a statistical research aid and cannot diagnose depression,
        anxiety, or suicide risk. Predictions are sensitive to dialect, sarcasm, and corpus bias,
        and submitted text is never stored outside your active browser session.
      </section>

      {showSupportivePanel && <SupportiveGuidancePanel />}
    </div>
  );
}
