import { useState } from 'react';
import { ArrowRight, Eraser, Sparkles } from 'lucide-react';
import {
  SimpleSupportiveDisclaimer,
  SupportiveGuidancePanel,
} from '@/src/components/EthicsBanner';
import type {
  EndpointConfig,
  InferenceExplanationBundle,
  PageId,
  PreprocessingOptions,
} from '@/src/types/mindtrace';
import {
  detectImmediateSelfHarmCues,
  detectSupportiveCues,
  preprocessSocialText,
  previewWordPieceTokens,
  validateInputText,
} from '@/src/utils/nlpPipeline';
import { createPlainLanguageInterpretation } from '@/src/utils/plainLanguage';

interface CheckTextPageProps {
  inputText: string;
  onChangeInputText: (text: string) => void;
  options: PreprocessingOptions;
  endpointConfig: EndpointConfig;
  activeBundle: InferenceExplanationBundle | null;
  onSetActiveBundle: (bundle: InferenceExplanationBundle | null) => void;
  onNavigate: (page: PageId) => void;
}

const SAMPLE_NOTE =
  'I have been feeling overwhelmed with everything on my plate lately and can barely sleep anymore. Nothing seems to quiet my mind at night.';

export function CheckTextPage({
  inputText,
  onChangeInputText,
  options,
  endpointConfig,
  activeBundle,
  onSetActiveBundle,
  onNavigate,
}: CheckTextPageProps) {
  const [hasAttemptedAnalyse, setHasAttemptedAnalyse] = useState(false);
  const [runningInference, setRunningInference] = useState(false);
  const [friendlyError, setFriendlyError] = useState<string | null>(null);

  const validation = validateInputText(inputText);
  const preprocessed = preprocessSocialText(inputText, options);
  const tokenPreview = previewWordPieceTokens(preprocessed.cleanedText);

  const submittedText = activeBundle?.originalText || activeBundle?.cleanedText || '';
  const showImmediateSafety =
    detectImmediateSelfHarmCues(inputText) || detectImmediateSelfHarmCues(submittedText);
  const showSupportivePanel =
    showImmediateSafety ||
    detectSupportiveCues(inputText, activeBundle?.predictedLabel ?? '');

  const plainSummary = activeBundle ? createPlainLanguageInterpretation(activeBundle) : null;

  async function handleAnalyse() {
    setHasAttemptedAnalyse(true);
    setFriendlyError(null);

    if (!validation.isValid) {
      return;
    }

    setRunningInference(true);
    const baseUrl = (endpointConfig.mode === 'built-in' ? '/api' : endpointConfig.url).replace(
      /\/+$/,
      ''
    );

    try {
      const resp = await fetch(`${baseUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          method: 'SHAP',
          modelId:
            endpointConfig.mode === 'built-in' ? endpointConfig.activeServerModelId : undefined,
          mask_urls: options.maskUrls,
          anonymize_mentions: options.anonymizeMentions,
          strip_hashtag_symbol: options.stripHashtagSymbol,
          normalize_whitespace: options.normalizeWhitespace,
          lowercase: options.lowercase,
        }),
      });

      if (!resp.ok) {
        throw new Error('We could not finish reading your text right now. Please try again in a moment.');
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
        inferenceMode:
          data.inference_mode === 'fine-tuned'
            ? 'fine-tuned'
            : data.inference_mode === 'zero-shot'
            ? 'zero-shot'
            : undefined,
        appliedOptions: { ...options },
      };
      onSetActiveBundle(bundle);
    } catch {
      setFriendlyError(
        'Something went wrong while checking your text. Please wait a moment and try again.'
      );
    } finally {
      setRunningInference(false);
    }
  }

  function handleClear() {
    onChangeInputText('');
    onSetActiveBundle(null);
    setHasAttemptedAnalyse(false);
    setFriendlyError(null);
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 py-2">
      {/* Step 1 & Step 2 */}
      <section className="neu-card-elevated p-6 sm:p-8 space-y-6">
        <div className="space-y-1.5">
          <div className="text-xs font-medium text-[#4D6FAF]">Step 1 · Enter your text</div>
          <h2 className="font-display text-xl font-bold text-[#203653]">
            Share a short note or message
          </h2>
          <p className="text-sm text-[#526783]">
            Write as much or as little as feels comfortable. Nothing you type here is saved.
          </p>
        </div>

        <div className="space-y-3">
          <label htmlFor="check-text-input" className="sr-only">
            Write or paste something you would like to understand
          </label>
          <textarea
            id="check-text-input"
            rows={6}
            value={inputText}
            onChange={(e) => {
              onChangeInputText(e.target.value);
              if (hasAttemptedAnalyse) setHasAttemptedAnalyse(false);
              if (friendlyError) setFriendlyError(null);
            }}
            placeholder="Write or paste something you'd like to understand..."
            className="neu-input w-full min-h-[170px] p-4 text-sm leading-relaxed placeholder:text-[#7B8FA8]"
          />

          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[#526783]">
            <span>
              {validation.wordCount} {validation.wordCount === 1 ? 'word' : 'words'}
            </span>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  onChangeInputText(SAMPLE_NOTE);
                  setHasAttemptedAnalyse(false);
                  setFriendlyError(null);
                }}
                className="neu-btn inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium min-h-[36px]"
              >
                <Sparkles className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
                <span>Try a sample message</span>
              </button>
              {inputText && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="neu-btn inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium min-h-[36px]"
                >
                  <Eraser className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
                  <span>Clear</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {hasAttemptedAnalyse && !validation.isValid && (
          <div
            role="alert"
            className="neu-inset px-4 py-3 text-xs text-[#6B2A3A]"
            style={{ background: '#F6E9ED' }}
          >
            {validation.errorMessage}
          </div>
        )}

        {validation.warnings.map((warn) => (
          <div
            key={warn}
            className="neu-inset px-4 py-3 text-xs text-[#5C4419]"
            style={{ background: '#F7F1E3' }}
          >
            {warn}
          </div>
        ))}

        {friendlyError && (
          <div
            role="alert"
            className="neu-inset px-4 py-3 text-xs text-[#6B2A3A]"
            style={{ background: '#F6E9ED' }}
          >
            {friendlyError}
          </div>
        )}

        <div className="pt-2 border-t border-[#AFC5F5]/55 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="text-xs font-medium text-[#4D6FAF]">Step 2 · Analyse</div>
          <button
            type="button"
            onClick={handleAnalyse}
            disabled={runningInference}
            className="neu-btn-primary inline-flex items-center justify-center px-7 py-3 text-sm min-h-[46px] disabled:opacity-55"
          >
            {runningInference ? 'Reading your text...' : 'Analyse Text'}
          </button>
        </div>

        {runningInference && (
          <div className="neu-inset px-4 py-3 text-xs text-[#203653]">
            Taking a moment to read your text and find the words that shaped the result...
          </div>
        )}
      </section>

      {showImmediateSafety && <SupportiveGuidancePanel immediateSafety />}

      {/* Step 3 · View your result */}
      {activeBundle && plainSummary && (
        <section
          aria-label="Your text result"
          className="neu-card-elevated p-6 sm:p-8 space-y-5"
        >
          <div className="space-y-1.5">
            <div className="text-xs font-medium text-[#4D6FAF]">Step 3 · View your result</div>
            <h3 className="font-display text-lg sm:text-xl font-bold text-[#203653]">
              {plainSummary.headline}
            </h3>
            <p className="text-sm leading-relaxed text-[#526783]">
              {plainSummary.noticedSummary}
            </p>
          </div>

          {plainSummary.supportingWords.length > 0 && (
            <div className="neu-inset p-4 space-y-2.5">
              <div className="text-xs font-semibold text-[#203653]">
                Words that stood out most
              </div>
              <div className="flex flex-wrap gap-2">
                {plainSummary.supportingWords.map((word) => (
                  <span
                    key={word}
                    className="rounded-xl bg-[#EAF0FF] border border-[#D2DFFF] px-3 py-1 text-xs font-medium text-[#203653]"
                  >
                    {word}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3 pt-1">
            <button
              type="button"
              onClick={() => onNavigate('understand-result')}
              className="neu-btn-primary inline-flex items-center gap-2 px-6 py-3 text-xs min-h-[44px]"
            >
              <span>Understand Your Result</span>
              <ArrowRight className="h-3.5 w-3.5 text-white" aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={handleClear}
              className="neu-btn inline-flex items-center gap-2 px-5 py-3 text-xs font-medium min-h-[44px]"
            >
              <span>Check Another Note</span>
            </button>
          </div>
        </section>
      )}

      {!showImmediateSafety && showSupportivePanel && <SupportiveGuidancePanel />}

      <SimpleSupportiveDisclaimer />
    </div>
  );
}
