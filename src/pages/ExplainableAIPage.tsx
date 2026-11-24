import { useMemo, useState, type ReactNode } from 'react';
import { ArrowLeft } from 'lucide-react';
import { SupportiveGuidancePanel } from '@/src/components/EthicsBanner';
import { PageHeader } from '@/src/components/PageHeader';
import type {
  EndpointConfig,
  ExplanationMethod,
  InferenceExplanationBundle,
  PageId,
  PreprocessingOptions,
  TokenAttribution,
} from '@/src/types/mindtrace';
import {
  buildGroundedExplanationSummary,
  detectImmediateSelfHarmCues,
  detectSupportiveCues,
} from '@/src/utils/nlpPipeline';

interface ExplainableAIPageProps {
  activeBundle: InferenceExplanationBundle | null;
  onSetActiveBundle: (bundle: InferenceExplanationBundle) => void;
  endpointConfig: EndpointConfig;
  preprocessingOptions: PreprocessingOptions;
  onNavigate: (page: PageId) => void;
  embedded?: boolean;
}

type PolarityFilter = 'all' | 'supporting' | 'opposing';

function buildHighlightedTextNodes(
  text: string,
  attributions: TokenAttribution[],
  maxAbsWeight: number
): ReactNode[] {
  if (!text) return [];

  const withOffsets = attributions
    .filter((a) => typeof a.start === 'number' && typeof a.end === 'number')
    .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));

  const resolvedSpans: { start: number; end: number; attr: TokenAttribution }[] = [];

  if (withOffsets.length > 0) {
    for (const attr of withOffsets) {
      const s = attr.start!;
      const e = attr.end!;
      if (s >= 0 && e <= text.length && s < e) {
        resolvedSpans.push({ start: s, end: e, attr });
      }
    }
  } else {
    let cursor = 0;
    const lowered = text.toLowerCase();
    for (const attr of attributions) {
      const cleanTok = attr.token.replace(/^##/, '');
      if (!cleanTok) continue;
      let idx = text.indexOf(cleanTok, cursor);
      if (idx === -1) {
        idx = lowered.indexOf(cleanTok.toLowerCase(), cursor);
      }
      if (idx !== -1) {
        resolvedSpans.push({ start: idx, end: idx + cleanTok.length, attr });
        cursor = idx + cleanTok.length;
      }
    }
  }

  if (resolvedSpans.length === 0) {
    return [<span key="plain-text">{text}</span>];
  }

  const nodes: ReactNode[] = [];
  let pos = 0;

  resolvedSpans.forEach((span, idx) => {
    if (span.start < pos) return;
    if (span.start > pos) {
      nodes.push(<span key={`gap-${idx}`}>{text.slice(pos, span.start)}</span>);
    }

    const slice = text.slice(span.start, span.end);
    const { weight } = span.attr;
    const isSup = weight > 1e-4;
    const isOpp = weight < -1e-4;

    if (!isSup && !isOpp) {
      nodes.push(<span key={`neutral-${idx}`}>{slice}</span>);
    } else {
      const norm = Math.min(Math.abs(weight) / maxAbsWeight, 1);
      const alpha = 0.24 + 0.36 * norm;
      const bg = isSup
        ? `rgba(119, 153, 216, ${alpha.toFixed(2)})`
        : `rgba(217, 149, 165, ${alpha.toFixed(2)})`;
      const borderColor = isSup ? '#4D6FAF' : '#B86B7E';

      nodes.push(
        <mark
          key={`hl-${idx}`}
          title={`${slice}: ${weight >= 0 ? '+' : ''}${weight.toFixed(4)} (${span.attr.direction})`}
          style={{
            backgroundColor: bg,
            borderBottom: `2px solid ${borderColor}`,
          }}
          className="rounded-md px-1.5 py-0.5 text-[#172840]"
        >
          {slice}
        </mark>
      );
    }

    pos = span.end;
  });

  if (pos < text.length) {
    nodes.push(<span key="tail-text">{text.slice(pos)}</span>);
  }

  return nodes;
}

export function ExplainableAIPage({
  activeBundle,
  onSetActiveBundle,
  endpointConfig,
  preprocessingOptions,
  onNavigate,
  embedded = false,
}: ExplainableAIPageProps) {
  const [polarityFilter, setPolarityFilter] = useState<PolarityFilter>('all');
  const [topK, setTopK] = useState<number>(12);
  const [recomputingMethod, setRecomputingMethod] = useState<ExplanationMethod | null>(null);
  const [recomputeError, setRecomputeError] = useState<string | null>(null);

  const rankedAttributions = useMemo(() => {
    if (!activeBundle) return [];
    const filtered = activeBundle.attributions.filter((item) => {
      if (polarityFilter === 'supporting') return item.weight > 1e-4;
      if (polarityFilter === 'opposing') return item.weight < -1e-4;
      return true;
    });
    return [...filtered]
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, topK);
  }, [activeBundle, polarityFilter, topK]);

  const supportingPhrases = useMemo(() => {
    if (!activeBundle) return [];
    return [...activeBundle.attributions]
      .filter((a) => a.weight > 1e-4)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 6);
  }, [activeBundle]);

  const opposingPhrases = useMemo(() => {
    if (!activeBundle) return [];
    return [...activeBundle.attributions]
      .filter((a) => a.weight < -1e-4)
      .sort((a, b) => a.weight - b.weight)
      .slice(0, 6);
  }, [activeBundle]);

  const maxAbsWeight = useMemo(() => {
    if (!activeBundle || activeBundle.attributions.length === 0) return 1;
    return Math.max(...activeBundle.attributions.map((a) => Math.abs(a.weight)), 1e-6);
  }, [activeBundle]);

  async function handleRecomputeMethod(method: ExplanationMethod) {
    if (!activeBundle) return;
    const textToAnalyze = (activeBundle.originalText || activeBundle.cleanedText || '').trim();
    if (textToAnalyze.length < 10) return;

    const opts = activeBundle.appliedOptions || preprocessingOptions;

    setRecomputingMethod(method);
    setRecomputeError(null);
    const baseUrl = (endpointConfig.mode === 'built-in' ? '/api' : endpointConfig.url).replace(
      /\/+$/,
      ''
    );

    try {
      const resp = await fetch(`${baseUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textToAnalyze,
          method,
          modelId:
            endpointConfig.mode === 'built-in' ? endpointConfig.activeServerModelId : undefined,
          mask_urls: opts.maskUrls,
          anonymize_mentions: opts.anonymizeMentions,
          strip_hashtag_symbol: opts.stripHashtagSymbol,
          normalize_whitespace: opts.normalizeWhitespace,
          lowercase: opts.lowercase,
        }),
      });
      if (!resp.ok) {
        const errPayload = await resp.json().catch(() => ({}));
        throw new Error(errPayload.detail || `Recompute failed with HTTP ${resp.status}`);
      }
      const data = await resp.json();
      onSetActiveBundle({
        source: 'local-api',
        modelName: String(data.model_name || activeBundle.modelName),
        originalText: String(data.original_text || textToAnalyze),
        cleanedText: String(data.cleaned_text || activeBundle.cleanedText),
        predictedLabel: String(data.predicted_label),
        confidence: Number(data.confidence),
        logits: data.logits || undefined,
        probabilities: data.probabilities || {},
        normalizedEntropy: Number(data.normalized_entropy ?? 0),
        isUncertain: Boolean(data.is_uncertain),
        tokens: Array.isArray(data.tokens) ? data.tokens : activeBundle.tokens,
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
            : activeBundle.inferenceMode,
        appliedOptions: opts,
      });
    } catch (err) {
      setRecomputeError(err instanceof Error ? err.message : 'Unable to recompute attributions.');
    } finally {
      setRecomputingMethod(null);
    }
  }

  const submittedText = activeBundle?.originalText || activeBundle?.cleanedText || '';
  const analyzedText = activeBundle?.cleanedText || submittedText;
  const showImmediateSafety = detectImmediateSelfHarmCues(submittedText);
  const showSupportive =
    showImmediateSafety ||
    (activeBundle ? detectSupportiveCues(submittedText, activeBundle.predictedLabel) : false);
  const canLiveRecompute =
    Boolean(activeBundle) &&
    submittedText.trim().length >= 10 &&
    (endpointConfig.mode === 'built-in' || endpointConfig.status === 'connected');

  return (
    <div className="space-y-7">
      {!embedded ? (
        <PageHeader
          title="Explainable AI"
          description="Inspect how SHAP and LIME measure which words in the latest analyzed text influenced the active Transformer model's prediction."
          action={
            activeBundle && canLiveRecompute ? (
              <div className="neu-inset inline-flex items-center gap-1.5 rounded-2xl p-1.5">
                {(['SHAP', 'LIME'] as ExplanationMethod[]).map((method) => {
                  const isSelected = activeBundle.method === method;
                  const isLoading = recomputingMethod === method;
                  return (
                    <button
                      key={method}
                      type="button"
                      disabled={Boolean(recomputingMethod)}
                      onClick={() => handleRecomputeMethod(method)}
                      className={`rounded-xl px-3.5 py-2 text-xs font-medium transition-all whitespace-nowrap ${
                        isSelected
                          ? 'neu-btn-accent text-[#203653]'
                          : 'text-[#526783] hover:text-[#203653]'
                      }`}
                    >
                      {isLoading ? `Running ${method}...` : `Explain with ${method}`}
                    </button>
                  );
                })}
              </div>
            ) : undefined
          }
        />
      ) : (
        activeBundle &&
        canLiveRecompute && (
          <div className="flex flex-wrap items-center justify-between gap-3 neu-card p-5">
            <div>
              <h3 className="font-display text-sm font-semibold text-[#203653]">
                Live Token Attribution Inspector
              </h3>
              <p className="text-xs text-[#526783]">
                Switch between Partition SHAP and LIME on your latest analyzed message.
              </p>
            </div>
            <div className="neu-inset inline-flex items-center gap-1.5 rounded-2xl p-1.5">
              {(['SHAP', 'LIME'] as ExplanationMethod[]).map((method) => {
                const isSelected = activeBundle.method === method;
                const isLoading = recomputingMethod === method;
                return (
                  <button
                    key={method}
                    type="button"
                    disabled={Boolean(recomputingMethod)}
                    onClick={() => handleRecomputeMethod(method)}
                    className={`rounded-xl px-3.5 py-2 text-xs font-medium transition-all whitespace-nowrap ${
                      isSelected
                        ? 'neu-btn-accent text-[#203653]'
                        : 'text-[#526783] hover:text-[#203653]'
                    }`}
                  >
                    {isLoading ? `Running ${method}...` : `Explain with ${method}`}
                  </button>
                );
              })}
            </div>
          </div>
        )
      )}

      {recomputeError && (
        <div
          className="neu-inset p-4 text-xs text-[#6B2A3A]"
          style={{ background: '#F6E9ED' }}
        >
          {recomputeError}
        </div>
      )}

      {!activeBundle ? (
        <section className="neu-card-elevated p-8 max-w-2xl">
          <div className="text-xs font-mono text-[#4D6FAF]">
            Explainable AI · Ready
          </div>
          <h2 className="mt-2 font-display text-lg font-semibold text-[#203653]">
            Analyse some text to view its explanation.
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-[#526783]">
            Submit text on the Text Analysis page using the active Transformer model. SHAP or LIME
            word-level attributions and highlighted text will appear here automatically.
          </p>

          <div className="mt-6">
            <button
              type="button"
              onClick={() => onNavigate('check-text')}
              className="neu-btn-primary inline-flex items-center gap-2 px-5 py-2.5 text-xs min-h-[42px]"
            >
              <ArrowLeft className="h-3.5 w-3.5 text-white" aria-hidden="true" />
              <span>Go to Check Your Text</span>
            </button>
          </div>
        </section>
      ) : (
        <>
          {/* 1. Original Submitted Text + 2. Highlighted Influential Words */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-5 neu-card p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Original Submitted Text
                </h2>
                <span className="font-mono text-[11px] text-[#526783]">
                  {activeBundle.source === 'local-api' ? 'Live Model' : 'Imported JSON'}
                </span>
              </div>
              <div className="neu-inset p-4 text-sm leading-relaxed text-[#203653] whitespace-pre-wrap break-words">
                {submittedText}
              </div>
              {submittedText !== analyzedText && (
                <div className="text-[11px] text-[#526783]">
                  Preprocessed input passed to tokenizer:{' '}
                  <span className="font-mono">{analyzedText}</span>
                </div>
              )}
            </section>

            <section className="lg:col-span-7 neu-card-elevated p-6 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Highlighted Influential Words
                </h2>
                <div className="flex items-center gap-4 text-xs text-[#526783]">
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="inline-block h-3 w-3 rounded"
                      style={{
                        backgroundColor: 'rgba(119, 153, 216, 0.5)',
                        border: '1px solid #4D6FAF',
                      }}
                    />
                    <span>Supporting (+)</span>
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="inline-block h-3 w-3 rounded"
                      style={{
                        backgroundColor: 'rgba(217, 149, 165, 0.5)',
                        border: '1px solid #B86B7E',
                      }}
                    />
                    <span>Opposing (-)</span>
                  </span>
                </div>
              </div>

              <div className="neu-inset p-4 text-sm leading-loose whitespace-pre-wrap break-words text-[#203653]">
                {buildHighlightedTextNodes(analyzedText, activeBundle.attributions, maxAbsWeight)}
              </div>
              <p className="text-[11px] text-[#526783]">
                Hover over any highlighted word to inspect its exact attribution weight toward{' '}
                <span className="font-semibold text-[#203653]">{activeBundle.predictedLabel}</span>.
              </p>
            </section>
          </div>

          {/* 3. Feature Importance Horizontal Bar Chart + Word Lists */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-7 neu-card p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="font-display text-base font-semibold text-[#203653]">
                    Feature Importance Chart (Top {rankedAttributions.length})
                  </h2>
                  <p className="text-xs text-[#526783]">
                    Horizontal diverging bar chart showing positive (+) and negative (-) contributions.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <div className="neu-inset inline-flex rounded-2xl p-1 gap-1">
                    {(
                      [
                        ['all', 'All'],
                        ['supporting', '+'],
                        ['opposing', '-'],
                      ] as [PolarityFilter, string][]
                    ).map(([val, lbl]) => (
                      <button
                        key={val}
                        type="button"
                        onClick={() => setPolarityFilter(val)}
                        className={`rounded-xl px-2.5 py-1 text-xs font-medium transition-all ${
                          polarityFilter === val
                            ? 'neu-btn-accent text-[#203653]'
                            : 'text-[#526783] hover:text-[#203653]'
                        }`}
                      >
                        {lbl}
                      </button>
                    ))}
                  </div>

                  <label className="flex items-center gap-1.5 text-xs text-[#526783]">
                    <span>Top</span>
                    <input
                      type="range"
                      min={5}
                      max={20}
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      className="w-16 accent-[#4D6FAF]"
                    />
                    <span className="font-mono tabular-nums w-5 text-right text-[#203653]">
                      {topK}
                    </span>
                  </label>
                </div>
              </div>

              <div className="space-y-2.5 pt-1">
                {rankedAttributions.map((item, idx) => {
                  const widthPct = Math.min((Math.abs(item.weight) / maxAbsWeight) * 50, 50);
                  const isPositive = item.weight >= 0;
                  return (
                    <div
                      key={`${item.token}-${idx}`}
                      className="grid grid-cols-12 items-center gap-2 text-xs"
                    >
                      <div className="col-span-3 truncate font-mono text-[#203653]" title={item.token}>
                        {item.token}
                      </div>
                      <div className="col-span-7 relative h-5 neu-inset rounded-full flex items-center overflow-hidden">
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[#7799D8]/70" />
                        <div
                          className="absolute h-3 rounded-full"
                          style={
                            isPositive
                              ? {
                                  left: '50%',
                                  width: `${widthPct}%`,
                                  background: 'linear-gradient(90deg, #7799D8, #4D6FAF)',
                                }
                              : {
                                  right: '50%',
                                  width: `${widthPct}%`,
                                  background: 'linear-gradient(90deg, #C77A8D, #D995A5)',
                                }
                          }
                        />
                      </div>
                      <div className="col-span-2 text-right font-mono tabular-nums text-[#526783]">
                        {item.weight >= 0 ? '+' : ''}
                        {item.weight.toFixed(4)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="lg:col-span-5 neu-card p-6 space-y-5">
              <div>
                <h3 className="font-display text-sm font-semibold text-[#2E5894]">
                  Strongest Supporting Words (+)
                </h3>
                <div className="mt-2.5 space-y-2">
                  {supportingPhrases.length === 0 ? (
                    <p className="text-xs text-[#526783]">None above threshold.</p>
                  ) : (
                    supportingPhrases.map((item, i) => (
                      <div
                        key={`${item.token}-sup-${i}`}
                        className="neu-inset flex justify-between px-3.5 py-2 text-xs"
                      >
                        <span className="font-mono text-[#203653]">{item.token}</span>
                        <span className="font-mono tabular-nums font-semibold text-[#2E5894]">
                          +{item.weight.toFixed(4)}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="border-t border-[#AFC5F5]/55 pt-4">
                <h3 className="font-display text-sm font-semibold text-[#96475B]">
                  Strongest Opposing Words (-)
                </h3>
                <div className="mt-2.5 space-y-2">
                  {opposingPhrases.length === 0 ? (
                    <p className="text-xs text-[#526783]">None above threshold.</p>
                  ) : (
                    opposingPhrases.map((item, i) => (
                      <div
                        key={`${item.token}-opp-${i}`}
                        className="neu-inset flex justify-between px-3.5 py-2 text-xs"
                      >
                        <span className="font-mono text-[#203653]">{item.token}</span>
                        <span className="font-mono tabular-nums font-semibold text-[#96475B]">
                          {item.weight.toFixed(4)}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </section>
          </div>

          {/* 4. Explanation Summary */}
          <section className="neu-card-elevated p-6 space-y-3">
            <div className="text-xs font-mono text-[#4D6FAF]">
              Method: {activeBundle.method} · Model: {activeBundle.modelName} · Predicted:{' '}
              {activeBundle.predictedLabel} ({(activeBundle.confidence * 100).toFixed(1)}%)
            </div>
            <h2 className="font-display text-base font-semibold text-[#203653]">
              Explanation Summary
            </h2>
            <p className="text-sm leading-relaxed text-[#526783]">
              {activeBundle.naturalLanguageSummary ||
                buildGroundedExplanationSummary(
                  activeBundle.method,
                  activeBundle.predictedLabel,
                  activeBundle.confidence,
                  activeBundle.attributions
                )}
            </p>
            <div className="neu-inset p-4 text-xs leading-relaxed text-[#526783]">
              <strong className="text-[#203653]">Interpretation Boundary: </strong>
              Feature attribution indicates how the Transformer model responded to specific words and
              subword tokens in the text; it does not establish the actual psychological reasons
              behind a person&apos;s writing.
            </div>
          </section>
        </>
      )}

      {/* 5. Explainability Method Information */}
      <section className="neu-card p-6 space-y-4">
        <h2 className="font-display text-base font-semibold text-[#203653]">
          Explainability Method Information
        </h2>
        <div className="grid gap-5 md:grid-cols-3 text-xs leading-relaxed text-[#526783]">
          <div className="neu-inset p-4">
            <div className="font-semibold text-[#203653]">Partition SHAP</div>
            <p className="mt-1">
              Uses the Hugging Face tokenizer hierarchy to mask token coalitions and estimate
              Shapley values directly from the Transformer&apos;s probability outputs.
            </p>
          </div>
          <div className="neu-inset p-4">
            <div className="font-semibold text-[#2E5894]">LIME Surrogate Perturbation</div>
            <p className="mt-1">
              Perturbs words in the submitted post, queries the fine-tuned Transformer classifier on
              each perturbed variant, and fits a local weighted linear model.
            </p>
          </div>
          <div className="neu-inset p-4">
            <div className="font-semibold text-[#203653]">Subword Token Handling</div>
            <p className="mt-1">
              Because BERT and DistilBERT split rare words into WordPiece subwords (such as{' '}
              <code className="font-mono">##ed</code> or <code className="font-mono">##ing</code>),
              MindTrace AI sums additive subword attributions back onto whole word spans in the
              submitted text.
            </p>
          </div>
        </div>
      </section>

      {showSupportive && <SupportiveGuidancePanel immediateSafety={showImmediateSafety} />}
    </div>
  );
}
