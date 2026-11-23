import { useMemo, useState, type ChangeEvent, type ReactNode } from 'react';
import { ArrowLeft, Download, Upload } from 'lucide-react';
import { PageHeader } from '@/src/components/PageHeader';
import type {
  InferenceExplanationBundle,
  PageId,
  TokenAttribution,
} from '@/src/types/mindtrace';
import { buildGroundedExplanationSummary } from '@/src/utils/nlpPipeline';

interface ExplainableAIPageProps {
  activeBundle: InferenceExplanationBundle | null;
  onSetActiveBundle: (bundle: InferenceExplanationBundle) => void;
  onNavigate: (page: PageId) => void;
}

type PolarityFilter = 'all' | 'supporting' | 'opposing';

function buildHighlightedTextNodes(
  text: string,
  attributions: TokenAttribution[],
  maxAbsWeight: number
): ReactNode[] {
  if (!text) return [];

  // Prefer exact (start, end) offsets returned by the Python SHAP/LIME engine
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
    // Sequential forward alignment fallback if an older JSON export lacks explicit offsets
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
      const alpha = 0.12 + 0.36 * norm;
      const bg = isSup
        ? `rgba(13, 115, 119, ${alpha.toFixed(2)})`
        : `rgba(180, 83, 9, ${alpha.toFixed(2)})`;
      const borderColor = isSup ? '#0D7377' : '#B45309';

      nodes.push(
        <mark
          key={`hl-${idx}`}
          title={`${slice}: ${weight >= 0 ? '+' : ''}${weight.toFixed(4)} (${span.attr.direction})`}
          style={{
            backgroundColor: bg,
            borderBottom: `2px solid ${borderColor}`,
          }}
          className="rounded px-1 py-0.5 text-[#0F172A]"
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
  onNavigate,
}: ExplainableAIPageProps) {
  const [polarityFilter, setPolarityFilter] = useState<PolarityFilter>('all');
  const [topK, setTopK] = useState<number>(12);
  const [importError, setImportError] = useState<string | null>(null);

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

  async function handleUploadXaiJson(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImportError(null);
    try {
      const parsed = JSON.parse(await file.text());
      if (!parsed.predicted_label || !Array.isArray(parsed.attributions)) {
        throw new Error('Invalid XAI JSON: expected predicted_label and attributions array.');
      }
      onSetActiveBundle({
        source: 'imported-json',
        modelName: String(parsed.model_name || 'distilbert-base-uncased'),
        originalText: String(parsed.original_text || parsed.cleaned_text || ''),
        cleanedText: String(parsed.cleaned_text || ''),
        predictedLabel: String(parsed.predicted_label),
        confidence: Number(parsed.confidence ?? 0),
        logits: parsed.logits || undefined,
        probabilities: parsed.probabilities || {},
        normalizedEntropy: Number(parsed.normalized_entropy ?? 0),
        isUncertain: Boolean(parsed.is_uncertain),
        tokens: Array.isArray(parsed.tokens) ? parsed.tokens : [],
        method: parsed.method === 'LIME' ? 'LIME' : 'SHAP',
        attributions: parsed.attributions,
        naturalLanguageSummary: String(parsed.natural_language_summary || ''),
        supportiveGuidance: Boolean(parsed.supportive_guidance),
      });
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Unable to load JSON file.');
    } finally {
      e.target.value = '';
    }
  }

  function handleDownloadSchemaTemplate() {
    const schemaTemplate = {
      _comment:
        'Schema exported by MindTrace AI Python api_bridge.py or app.py. Populate with actual model outputs.',
      model_name: 'distilbert-base-uncased',
      original_text: '<original user input text>',
      cleaned_text: '<preprocessed input text>',
      predicted_label: '<model_label>',
      confidence: 0.0,
      probabilities: { '<class_a>': 0.0, '<class_b>': 0.0 },
      normalized_entropy: 0.0,
      is_uncertain: false,
      method: 'SHAP',
      attributions: [{ token: '<word>', weight: 0.0, direction: 'neutral', start: 0, end: 0 }],
    };
    const blob = new Blob([JSON.stringify(schemaTemplate, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mindtrace-xai-schema.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  const submittedText = activeBundle?.originalText || activeBundle?.cleanedText || '';
  const analyzedText = activeBundle?.cleanedText || submittedText;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Explainable AI"
        description="Inspect how SHAP and LIME measure which words in the submitted text influenced the Transformer model's prediction."
      />

      {!activeBundle ? (
        <section className="rounded-xl border border-[#E2E8F0] bg-white p-8 max-w-3xl">
          <div className="text-xs font-mono text-[#0D7377]">
            Empty State · No Active Model Attribution
          </div>
          <h2 className="mt-2 font-display text-lg font-semibold text-[#0F172A]">
            Supply Real Model Output to Inspect Explanations
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-[#475569]">
            MindTrace AI never simulates feature importance scores or highlights words using a static
            dictionary. Analyse text using a connected local Transformer bridge or import an XAI
            JSON report exported from the Python pipeline.
          </p>

          {importError && (
            <div className="mt-4 rounded-lg bg-[#FEF2F2] p-3 text-xs text-[#991B1B]">
              {importError}
            </div>
          )}

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => onNavigate('text-analysis')}
              className="inline-flex items-center gap-2 rounded-lg bg-[#0F2942] px-4 py-2.5 text-xs font-medium text-white hover:bg-[#173A5E] transition-colors min-h-[40px]"
            >
              <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Go to Text Analysis</span>
            </button>

            <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-[#CBD5E1] bg-white px-4 py-2.5 text-xs font-medium text-[#0F172A] hover:bg-[#F1F5F9] transition-colors min-h-[40px]">
              <Upload className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Import XAI JSON Export</span>
              <input
                type="file"
                accept=".json,application/json"
                onChange={handleUploadXaiJson}
                className="sr-only"
              />
            </label>

            <button
              type="button"
              onClick={handleDownloadSchemaTemplate}
              className="inline-flex items-center gap-2 rounded-lg border border-[#E2E8F0] bg-[#F8FAFC] px-4 py-2.5 text-xs font-medium text-[#475569] hover:text-[#0F172A] transition-colors min-h-[40px]"
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Download JSON Schema</span>
            </button>
          </div>
        </section>
      ) : (
        <>
          {/* 1. Original Submitted Text + 2. Highlighted Influential Words */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-5 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-semibold text-[#0F172A]">
                  Original Submitted Text
                </h2>
                <span className="font-mono text-[11px] text-[#64748B]">
                  {activeBundle.source === 'local-api' ? 'Live Model' : 'Imported JSON'}
                </span>
              </div>
              <div className="rounded-lg bg-[#F8FAFC] p-4 text-sm leading-relaxed text-[#334155] whitespace-pre-wrap break-words">
                {submittedText}
              </div>
              {submittedText !== analyzedText && (
                <div className="text-[11px] text-[#64748B]">
                  Preprocessed input passed to tokenizer: <span className="font-mono">{analyzedText}</span>
                </div>
              )}
            </section>

            <section className="lg:col-span-7 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="font-display text-base font-semibold text-[#0F172A]">
                  Highlighted Influential Words
                </h2>
                <div className="flex items-center gap-4 text-xs text-[#475569]">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block h-3 w-3 rounded bg-[#0D7377]/25 border border-[#0D7377]" />
                    <span>Supporting (+)</span>
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block h-3 w-3 rounded bg-[#B45309]/25 border border-[#B45309]" />
                    <span>Opposing (-)</span>
                  </span>
                </div>
              </div>

              <div className="rounded-lg bg-[#F8FAFC] p-4 text-sm leading-loose whitespace-pre-wrap break-words">
                {buildHighlightedTextNodes(analyzedText, activeBundle.attributions, maxAbsWeight)}
              </div>
              <p className="text-[11px] text-[#64748B]">
                Hover over any highlighted word to inspect its exact attribution weight toward{' '}
                <span className="font-medium text-[#0F172A]">{activeBundle.predictedLabel}</span>.
              </p>
            </section>
          </div>

          {/* 3. Feature Importance Horizontal Bar Chart + Word Lists */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-7 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="font-display text-base font-semibold text-[#0F172A]">
                    Feature Importance Chart (Top {rankedAttributions.length})
                  </h2>
                  <p className="text-xs text-[#64748B]">
                    Horizontal diverging bar chart showing positive (+) and negative (-) contributions.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <div className="inline-flex rounded-lg bg-[#F1F5F9] p-1">
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
                        className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                          polarityFilter === val
                            ? 'bg-white text-[#0F2942] shadow-xs'
                            : 'text-[#64748B] hover:text-[#0F172A]'
                        }`}
                      >
                        {lbl}
                      </button>
                    ))}
                  </div>

                  <label className="flex items-center gap-1.5 text-xs text-[#475569]">
                    <span>Top</span>
                    <input
                      type="range"
                      min={5}
                      max={20}
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      className="w-16 accent-[#0F2942]"
                    />
                    <span className="font-mono tabular-nums w-5 text-right">{topK}</span>
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
                      <div className="col-span-3 truncate font-mono text-[#0F172A]" title={item.token}>
                        {item.token}
                      </div>
                      <div className="col-span-7 relative h-5 bg-[#F8FAFC] rounded flex items-center">
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[#CBD5E1]" />
                        <div
                          className={`absolute h-3.5 rounded-xs ${
                            isPositive ? 'bg-[#0D7377]' : 'bg-[#B45309]'
                          }`}
                          style={
                            isPositive
                              ? { left: '50%', width: `${widthPct}%` }
                              : { right: '50%', width: `${widthPct}%` }
                          }
                        />
                      </div>
                      <div className="col-span-2 text-right font-mono tabular-nums text-[#334155]">
                        {item.weight >= 0 ? '+' : ''}
                        {item.weight.toFixed(4)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="lg:col-span-5 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-5">
              <div>
                <h3 className="font-display text-sm font-semibold text-[#0D7377]">
                  Strongest Supporting Words (+)
                </h3>
                <div className="mt-2 space-y-1.5">
                  {supportingPhrases.length === 0 ? (
                    <p className="text-xs text-[#64748B]">None above threshold.</p>
                  ) : (
                    supportingPhrases.map((item, i) => (
                      <div
                        key={`${item.token}-sup-${i}`}
                        className="flex justify-between rounded-lg bg-[#F8FAFC] px-3 py-1.5 text-xs"
                      >
                        <span className="font-mono text-[#0F172A]">{item.token}</span>
                        <span className="font-mono tabular-nums font-semibold text-[#0D7377]">
                          +{item.weight.toFixed(4)}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="border-t border-[#E2E8F0] pt-4">
                <h3 className="font-display text-sm font-semibold text-[#B45309]">
                  Strongest Opposing Words (-)
                </h3>
                <div className="mt-2 space-y-1.5">
                  {opposingPhrases.length === 0 ? (
                    <p className="text-xs text-[#64748B]">None above threshold.</p>
                  ) : (
                    opposingPhrases.map((item, i) => (
                      <div
                        key={`${item.token}-opp-${i}`}
                        className="flex justify-between rounded-lg bg-[#F8FAFC] px-3 py-1.5 text-xs"
                      >
                        <span className="font-mono text-[#0F172A]">{item.token}</span>
                        <span className="font-mono tabular-nums font-semibold text-[#B45309]">
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
          <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-3">
            <div className="text-xs font-mono text-[#0D7377]">
              Method: {activeBundle.method} · Model: {activeBundle.modelName} · Predicted:{' '}
              {activeBundle.predictedLabel} ({(activeBundle.confidence * 100).toFixed(1)}%)
            </div>
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Explanation Summary
            </h2>
            <p className="text-sm leading-relaxed text-[#334155]">
              {activeBundle.naturalLanguageSummary ||
                buildGroundedExplanationSummary(
                  activeBundle.method,
                  activeBundle.predictedLabel,
                  activeBundle.confidence,
                  activeBundle.attributions
                )}
            </p>
            <div className="rounded-lg bg-[#F8FAFC] p-3.5 text-xs leading-relaxed text-[#475569]">
              <strong className="text-[#0F172A]">Interpretation Boundary: </strong>
              Feature attribution indicates how the Transformer model responded to specific words and
              subword tokens in the text; it does not establish the actual psychological reasons
              behind a person&apos;s writing.
            </div>
          </section>
        </>
      )}

      {/* 5. Explainability Method Information (always visible for academic clarity) */}
      <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
        <h2 className="font-display text-base font-semibold text-[#0F172A]">
          Explainability Method Information
        </h2>
        <div className="grid gap-6 md:grid-cols-3 text-xs leading-relaxed text-[#475569]">
          <div className="rounded-lg bg-[#F8FAFC] p-4">
            <div className="font-semibold text-[#0F2942]">Partition SHAP</div>
            <p className="mt-1">
              Uses the Hugging Face tokenizer hierarchy to mask token coalitions and estimate
              Shapley values directly from the Transformer&apos;s probability outputs.
            </p>
          </div>
          <div className="rounded-lg bg-[#F8FAFC] p-4">
            <div className="font-semibold text-[#0D7377]">LIME Surrogate Perturbation</div>
            <p className="mt-1">
              Perturbs words in the submitted post, queries the fine-tuned Transformer classifier on
              each perturbed variant, and fits a local weighted linear model.
            </p>
          </div>
          <div className="rounded-lg bg-[#F8FAFC] p-4">
            <div className="font-semibold text-[#0F172A]">Subword Token Handling</div>
            <p className="mt-1">
              Because BERT and DistilBERT split rare words into WordPiece subwords (such as{' '}
              <code className="font-mono">##ed</code> or <code className="font-mono">##ing</code>),
              MindTrace AI sums additive subword attributions back onto whole word spans in the
              submitted text.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
