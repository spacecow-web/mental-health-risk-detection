import { useMemo, useState, type ChangeEvent } from 'react';
import { ChevronDown, Cpu, Play, Trash2, Upload } from 'lucide-react';
import type { Data, Layout } from 'plotly.js';
import { PageHeader } from '@/src/components/PageHeader';
import { PlotlyChart } from '@/src/components/PlotlyChart';
import type {
  EndpointConfig,
  LoadedCorpusData,
  NamedEvaluationRun,
  TrainingHistorySummary,
} from '@/src/types/mindtrace';
import { parseCsvContent, type ParsedCsv } from '@/src/utils/csvParser';
import { computeEvaluationMetrics, parseTrainingHistoryCsv } from '@/src/utils/evaluationEngine';

interface ModelPerformancePageProps {
  evaluationRuns: NamedEvaluationRun[];
  activeRunId: string | null;
  onAddOrUpdateRun: (run: NamedEvaluationRun, history?: TrainingHistorySummary) => void;
  onSelectRun: (runId: string) => void;
  onRemoveRun: (runId: string) => void;
  trainingHistory: TrainingHistorySummary | null;
  onSetTrainingHistory: (history: TrainingHistorySummary | null) => void;
  loadedCorpus: LoadedCorpusData | null;
  endpointConfig: EndpointConfig;
  onUpdateEndpointConfig: (config: EndpointConfig) => void;
  embedded?: boolean;
}

const SERIES_COLORS = ['#4D6FAF', '#72B9D8', '#7799D8', '#91B9E8', '#83BBA5'];

export function ModelPerformancePage({
  evaluationRuns,
  activeRunId,
  onAddOrUpdateRun,
  onSelectRun,
  onRemoveRun,
  trainingHistory,
  onSetTrainingHistory,
  loadedCorpus,
  endpointConfig,
  onUpdateEndpointConfig,
  embedded = false,
}: ModelPerformancePageProps) {
  const [presetLabel, setPresetLabel] = useState<'DistilBERT' | 'BERT-base' | 'custom'>('DistilBERT');
  const [customLabel, setCustomLabel] = useState('');
  const [lastParsedCsv, setLastParsedCsv] = useState<{ fileName: string; data: ParsedCsv } | null>(null);
  const [trueCol, setTrueCol] = useState('true_label');
  const [predCol, setPredCol] = useState('predicted_label');
  const [evalError, setEvalError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [evaluatingActive, setEvaluatingActive] = useState(false);
  const [trainingBackbone, setTrainingBackbone] = useState<
    'distilbert-encoder' | 'bert-base-encoder' | null
  >(null);
  const [trainStatusNote, setTrainStatusNote] = useState<string | null>(null);

  const effectiveRunLabel =
    presetLabel === 'custom' ? customLabel.trim() || 'Custom Transformer' : presetLabel;

  const activeRun =
    evaluationRuns.find((r) => r.id === activeRunId) ?? evaluationRuns[0] ?? null;
  const evalSummary = activeRun?.summary ?? null;

  async function handleRunActiveEvaluation() {
    setEvaluatingActive(true);
    setEvalError(null);
    setTrainStatusNote(null);
    try {
      const resp = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelId: endpointConfig.activeServerModelId,
          rows: loadedCorpus?.rows,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || `Evaluation failed with HTTP ${resp.status}`);
      }
      if (data.run) {
        onAddOrUpdateRun(data.run);
        setTrainStatusNote(
          `Evaluated ${data.run.modelName} on N=${data.run.summary.sampleCount} held-out benchmark posts and persisted results.`
        );
      }
    } catch (err) {
      setEvalError(
        err instanceof Error ? err.message : 'Unable to evaluate the active model on the default corpus.'
      );
    } finally {
      setEvaluatingActive(false);
    }
  }

  async function handleTrainBackbone(backbone: 'distilbert-encoder' | 'bert-base-encoder') {
    setTrainingBackbone(backbone);
    setEvalError(null);
    setTrainStatusNote(null);

    try {
      const resp = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rows: loadedCorpus?.rows,
          backbone,
          datasetName: loadedCorpus?.datasetName,
          epochs: 12,
          maxSamples: 240,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || `Training failed with HTTP ${resp.status}`);
      }

      onAddOrUpdateRun(
        {
          id: data.runId,
          modelName: data.modelName,
          summary: data.summary,
        },
        data.trainingHistory
      );

      if (data.status) {
        onUpdateEndpointConfig({
          ...endpointConfig,
          mode: 'built-in',
          url: '/api',
          status: 'connected',
          activeServerModelId: data.status.activeModelId || backbone,
          availableServerModels: Array.isArray(data.status.availableModels)
            ? data.status.availableModels
            : endpointConfig.availableServerModels,
          detectedLabels: Array.isArray(data.status.labels)
            ? data.status.labels
            : endpointConfig.detectedLabels,
          lastError: null,
        });
      }

      setTrainStatusNote(
        `Trained & persisted ${data.modelName} on ${data.splitCounts.train} train / ${data.splitCounts.val} val / ${data.splitCounts.test} strict held-out test posts (checkpoint now active for Text Analysis).`
      );
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : 'Transformer fine-tuning failed.');
    } finally {
      setTrainingBackbone(null);
    }
  }

  async function handleUploadEvalCsv(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setEvalError(null);
    try {
      const text = await file.text();
      const parsed = parseCsvContent(text);
      const headers = parsed.headers;
      const detectedTrue =
        headers.find((h) => h.toLowerCase() === 'true_label') ??
        headers.find((h) => h.toLowerCase().includes('true')) ??
        headers[0];
      const detectedPred =
        headers.find((h) => h.toLowerCase() === 'predicted_label') ??
        headers.find((h) => h.toLowerCase().includes('pred')) ??
        headers[Math.min(1, headers.length - 1)];

      setLastParsedCsv({ fileName: file.name, data: parsed });
      setTrueCol(detectedTrue);
      setPredCol(detectedPred);

      const summary = computeEvaluationMetrics(parsed.rows, detectedTrue, detectedPred, file.name);
      const runId = `${effectiveRunLabel.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${Date.now()}`;
      onAddOrUpdateRun({ id: runId, modelName: effectiveRunLabel, summary });
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : 'Unable to evaluate custom CSV.');
    } finally {
      e.target.value = '';
    }
  }

  function handleRecomputeColumns(nextTrue: string, nextPred: string) {
    setTrueCol(nextTrue);
    setPredCol(nextPred);
    if (!lastParsedCsv || !activeRun) return;
    try {
      setEvalError(null);
      const summary = computeEvaluationMetrics(
        lastParsedCsv.data.rows,
        nextTrue,
        nextPred,
        lastParsedCsv.fileName
      );
      onAddOrUpdateRun({ ...activeRun, summary });
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : 'Invalid column selection.');
    }
  }

  async function handleUploadTrainingHistory(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setHistoryError(null);
    try {
      const text = await file.text();
      const parsed = parseCsvContent(text);
      const history = parseTrainingHistoryCsv(parsed.rows, file.name);
      onSetTrainingHistory(history);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Unable to parse training history CSV.');
    } finally {
      e.target.value = '';
    }
  }

  const cmPlot = useMemo<{ data: Data[]; layout: Partial<Layout> } | null>(() => {
    if (!evalSummary) return null;
    const z = evalSummary.confusionMatrix;
    const textMatrix = z.map((row) => row.map((v) => String(v)));
    return {
      data: [
        {
          type: 'heatmap',
          z,
          x: evalSummary.labels,
          y: evalSummary.labels,
          text: textMatrix as unknown as string[],
          texttemplate: '%{text}',
          textfont: { color: '#14263D', size: 13 },
          colorscale: [
            [0, '#EAF0FF'],
            [0.5, '#AFC5F5'],
            [1, '#7799D8'],
          ],
          hovertemplate: 'True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
          showscale: true,
        },
      ],
      layout: {
        height: 360,
        xaxis: { title: { text: 'Predicted Label' } },
        yaxis: { title: { text: 'True Label' }, autorange: 'reversed' },
        margin: { l: 90, r: 24, t: 24, b: 64 },
      },
    };
  }, [evalSummary]);

  const rocPlot = useMemo<{ data: Data[]; layout: Partial<Layout> } | null>(() => {
    if (!evalSummary || evalSummary.rocCurves.length === 0) return null;
    const traces: Data[] = evalSummary.rocCurves.map((series, idx) => ({
      type: 'scatter',
      mode: 'lines',
      name: `${series.label} (AUC = ${series.auc.toFixed(3)})`,
      x: series.points.map((p) => p.fpr),
      y: series.points.map((p) => p.tpr),
      line: { color: SERIES_COLORS[idx % SERIES_COLORS.length], width: 2.5 },
    }));
    traces.push({
      type: 'scatter',
      mode: 'lines',
      name: 'Random Baseline (0.500)',
      x: [0, 1],
      y: [0, 1],
      line: { color: '#8294AE', width: 1.5, dash: 'dash' },
    });
    return {
      data: traces,
      layout: {
        height: 360,
        xaxis: { title: { text: 'False Positive Rate' }, range: [0, 1.02] },
        yaxis: { title: { text: 'True Positive Rate' }, range: [0, 1.04] },
        legend: { orientation: 'h', y: -0.22 },
        margin: { l: 56, r: 20, t: 24, b: 84 },
      },
    };
  }, [evalSummary]);

  const comparisonPlot = useMemo<{ data: Data[]; layout: Partial<Layout> } | null>(() => {
    if (evaluationRuns.length < 2) return null;
    const metricNames = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Weighted F1'];
    const traces: Data[] = evaluationRuns.map((run, idx) => ({
      type: 'bar',
      name: `${run.modelName} (N=${run.summary.sampleCount})`,
      x: metricNames,
      y: [
        run.summary.accuracy,
        run.summary.macroPrecision,
        run.summary.macroRecall,
        run.summary.macroF1,
        run.summary.weightedF1,
      ],
      marker: {
        color: SERIES_COLORS[idx % SERIES_COLORS.length],
        line: { color: '#EAF0FF', width: 1 },
      },
      hovertemplate: '%{x}: %{y:.4f}<extra>%{fullData.name}</extra>',
    }));
    return {
      data: traces,
      layout: {
        barmode: 'group',
        height: 360,
        yaxis: { title: { text: 'Score (0.0 – 1.0)' }, range: [0, 1.06] },
        legend: { orientation: 'h', y: -0.2 },
        margin: { l: 56, r: 24, t: 24, b: 78 },
      },
    };
  }, [evaluationRuns]);

  const sameTestSplitVerified = useMemo(() => {
    if (evaluationRuns.length < 2) return true;
    const base = evaluationRuns[0].summary;
    const baseLabels = base.labels.join('|');
    return evaluationRuns.every(
      (r) =>
        r.summary.sampleCount === base.sampleCount &&
        r.summary.labels.join('|') === baseLabels
    );
  }, [evaluationRuns]);

  const bertRun = evaluationRuns.find(
    (r) => /\bbert\b/i.test(r.modelName) && !/distilbert/i.test(r.modelName)
  );
  const distilRun = evaluationRuns.find((r) => /distilbert/i.test(r.modelName));

  const historyPlot = useMemo<{ data: Data[]; layout: Partial<Layout> } | null>(() => {
    if (!trainingHistory) return null;
    const traces: Data[] = [];
    const trainPts = trainingHistory.epochs.filter((e) => e.trainLoss !== null);
    const valPts = trainingHistory.epochs.filter((e) => e.evalLoss !== null);

    if (trainPts.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Training Loss',
        x: trainPts.map((e) => e.epoch),
        y: trainPts.map((e) => e.trainLoss as number),
        line: { color: '#4D6FAF', width: 2.5 },
        marker: { color: '#4D6FAF', size: 6 },
      });
    }
    if (valPts.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Validation Loss',
        x: valPts.map((e) => e.epoch),
        y: valPts.map((e) => e.evalLoss as number),
        line: { color: '#72B9D8', width: 2.5 },
        marker: { color: '#72B9D8', size: 6 },
      });
    }
    return {
      data: traces,
      layout: {
        height: 300,
        xaxis: { title: { text: 'Epoch' }, dtick: 1 },
        yaxis: { title: { text: 'Cross-Entropy Loss' } },
        legend: { orientation: 'h', y: -0.25 },
        margin: { l: 56, r: 20, t: 24, b: 72 },
      },
    };
  }, [trainingHistory]);

  const busy = Boolean(trainingBackbone) || evaluatingActive;

  return (
    <div className="space-y-7">
      {!embedded && (
        <PageHeader
          title="Model Performance"
          description="Inspect persisted Transformer evaluation metrics or evaluate and fine-tune DistilBERT and BERT-base directly on the leak-free 70/15/15 benchmark split."
        />
      )}

      {/* Built-In Evaluation & Live Transformer Training Controls (No Upload Required) */}
      <section className="neu-card-elevated p-7 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h2 className="font-display text-base font-semibold text-[#203653]">
              Transformer Evaluation & Fine-Tuning
            </h2>
            <p className="mt-1 text-xs text-[#526783]">
              Uses the active research dataset with pre-split deduplication, stratified 70/15/15
              partitioning, class-weighted training, and automatic persistence of held-out metrics.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 shrink-0">
            <button
              type="button"
              onClick={handleRunActiveEvaluation}
              disabled={busy}
              className="neu-btn-primary inline-flex items-center gap-2 px-4 py-2.5 text-xs disabled:opacity-50 whitespace-nowrap min-h-[40px]"
            >
              <Play className="h-3.5 w-3.5 text-white" aria-hidden="true" />
              <span>{evaluatingActive ? 'Evaluating Model...' : 'Run Model Evaluation'}</span>
            </button>

            <button
              type="button"
              onClick={() => handleTrainBackbone('distilbert-encoder')}
              disabled={busy}
              className="neu-btn-primary inline-flex items-center gap-2 px-4 py-2.5 text-xs disabled:opacity-50 whitespace-nowrap min-h-[40px]"
            >
              <Cpu className="h-3.5 w-3.5 text-white" aria-hidden="true" />
              <span>
                {trainingBackbone === 'distilbert-encoder'
                  ? 'Training DistilBERT...'
                  : 'Train & Evaluate DistilBERT (6-Layer)'}
              </span>
            </button>

            <button
              type="button"
              onClick={() => handleTrainBackbone('bert-base-encoder')}
              disabled={busy}
              className="neu-btn inline-flex items-center gap-2 px-4 py-2.5 text-xs font-semibold disabled:opacity-50 whitespace-nowrap min-h-[40px]"
            >
              <Cpu className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
              <span>
                {trainingBackbone === 'bert-base-encoder'
                  ? 'Training BERT-Base...'
                  : 'Train & Evaluate BERT-Base (12-Layer)'}
              </span>
            </button>
          </div>
        </div>

        <div className="neu-inset px-4 py-3 text-xs text-[#526783] flex flex-wrap items-center justify-between gap-2">
          <span>
            Corpus:{' '}
            <strong className="text-[#203653]">
              {loadedCorpus?.datasetName || 'Default Backend Research Dataset'}
            </strong>
          </span>
          <span className="font-mono tabular-nums text-[#203653]">
            {loadedCorpus
              ? `${loadedCorpus.rows.length} labeled records ready`
              : 'Managed automatically on backend'}
          </span>
        </div>

        {evaluatingActive && (
          <div className="neu-inset px-4 py-3 text-xs text-[#203653]">
            Evaluating active Transformer model against the held-out test split and persisting metrics...
          </div>
        )}

        {trainingBackbone && (
          <div className="neu-inset px-4 py-3 text-xs text-[#203653]">
            Encoding corpus with{' '}
            {trainingBackbone === 'distilbert-encoder' ? 'DistilBERT (6-layer)' : 'BERT-base (12-layer)'}{' '}
            & running class-weighted training on stratified 70/15/15 split...
          </div>
        )}

        {trainStatusNote && (
          <div
            className="neu-inset px-4 py-3 text-xs text-[#1E4D3B] font-medium"
            style={{ background: '#E7F3EE' }}
          >
            {trainStatusNote}
          </div>
        )}

        {evaluationRuns.length > 0 && (
          <div className="pt-3 border-t border-[#AFC5F5]/55 flex flex-wrap items-center gap-2.5">
            <span className="text-xs font-medium text-[#526783] mr-1">Saved Evaluations:</span>
            {evaluationRuns.map((run) => {
              const isSelected = activeRun?.id === run.id;
              return (
                <div
                  key={run.id}
                  className={`inline-flex items-center text-xs transition-all ${
                    isSelected ? 'neu-btn-accent' : 'neu-btn'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => onSelectRun(run.id)}
                    className="px-3.5 py-1.5 font-medium whitespace-nowrap text-[#203653]"
                  >
                    {run.modelName} (N={run.summary.sampleCount})
                  </button>
                  <button
                    type="button"
                    onClick={() => onRemoveRun(run.id)}
                    aria-label={`Remove ${run.modelName}`}
                    className={`pr-3 pl-1 py-1.5 opacity-80 hover:opacity-100 ${
                      isSelected ? 'text-[#203653]' : 'text-[#526783]'
                    }`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {evalError && (
          <div
            className="neu-inset p-3.5 text-xs text-[#6B2A3A]"
            style={{ background: '#F6E9ED' }}
          >
            {evalError}
          </div>
        )}
      </section>

      {!evalSummary ? (
        <section className="neu-card p-8 max-w-3xl space-y-3">
          <div className="text-xs font-mono text-[#4D6FAF]">
            Empty State · Model Not Yet Evaluated
          </div>
          <h2 className="font-display text-lg font-semibold text-[#203653]">
            No Saved Evaluation Results Available Yet
          </h2>
          <p className="text-sm leading-relaxed text-[#526783]">
            MindTrace AI never displays synthetic accuracy values or fabricated charts. Click{' '}
            <strong className="text-[#203653]">Run Model Evaluation</strong> above to evaluate the
            active model on the default research corpus, or train{' '}
            <strong className="text-[#203653]">DistilBERT</strong> /{' '}
            <strong className="text-[#203653]">BERT-Base</strong> to generate and persist genuine
            held-out metrics automatically.
          </p>
        </section>
      ) : (
        <>
          {/* 6 Core Metrics Strip */}
          <section className="neu-card p-7 space-y-4">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
              {[
                { label: 'Accuracy', value: `${(evalSummary.accuracy * 100).toFixed(2)}%` },
                { label: 'Macro Precision', value: evalSummary.macroPrecision.toFixed(4) },
                { label: 'Macro Recall', value: evalSummary.macroRecall.toFixed(4) },
                { label: 'Macro F1-Score', value: evalSummary.macroF1.toFixed(4) },
                { label: 'Weighted F1-Score', value: evalSummary.weightedF1.toFixed(4) },
                {
                  label: 'Macro ROC-AUC',
                  value:
                    evalSummary.rocAucMacro !== null ? evalSummary.rocAucMacro.toFixed(4) : 'N/A',
                },
              ].map((kpi) => (
                <div key={kpi.label} className="neu-inset p-4">
                  <div className="text-xs text-[#526783]">{kpi.label}</div>
                  <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-[#203653]">
                    {kpi.value}
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs leading-relaxed text-[#526783]">
              <strong className="text-[#203653]">Metric Guide:</strong> <em>Accuracy</em> is the
              overall share of correct predictions (N={evalSummary.sampleCount}). <em>Precision</em>{' '}
              measures how reliable a positive prediction is, while <em>Recall</em> measures how many
              true cases were found. <em>Macro F1</em> treats every category equally regardless of
              size, whereas <em>Weighted F1</em> scales each category&apos;s F1-score by its sample
              count.
            </p>
          </section>

          {/* Interactive Plotly Confusion Matrix & ROC-AUC */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-6 neu-card p-7 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-display text-base font-semibold text-[#203653]">
                  Confusion Matrix ({activeRun?.modelName})
                </h3>
                <span className="text-xs font-mono tabular-nums text-[#526783]">
                  N={evalSummary.sampleCount}
                </span>
              </div>
              {cmPlot && <PlotlyChart data={cmPlot.data} layout={cmPlot.layout} />}
              <p className="text-xs leading-relaxed text-[#526783]">
                <strong className="text-[#203653]">How to read this chart:</strong> Deeper periwinkle
                cells along the diagonal from top-left to bottom-right represent correct predictions.
                Off-diagonal numbers show where the model confused one category for another.
              </p>
            </section>

            <section className="lg:col-span-6 neu-card p-7 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-display text-base font-semibold text-[#203653]">
                  ROC Curves & AUC (One-vs-Rest)
                </h3>
                {evalSummary.rocAucMacro !== null && (
                  <span className="font-mono text-xs tabular-nums text-[#4D6FAF]">
                    Macro AUC: {evalSummary.rocAucMacro.toFixed(4)}
                  </span>
                )}
              </div>

              {!rocPlot ? (
                <div className="neu-inset p-6 text-xs leading-relaxed text-[#526783]">
                  ROC-AUC curves require continuous per-class probability scores alongside held-out
                  predictions.
                </div>
              ) : (
                <>
                  <PlotlyChart data={rocPlot.data} layout={rocPlot.layout} />
                  <p className="text-xs leading-relaxed text-[#526783]">
                    <strong className="text-[#203653]">How to read this chart:</strong> Curves
                    arching closer to the top-left corner (AUC near 1.0) mean the model cleanly
                    separates that category from all others across confidence thresholds, whereas
                    the dashed diagonal (0.50) represents random guessing.
                  </p>
                </>
              )}
            </section>
          </div>

          {/* Classification Report */}
          <section className="neu-card p-7 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[#AFC5F5]/55">
              <h3 className="font-display text-base font-semibold text-[#203653]">
                Classification Report ({activeRun?.modelName})
              </h3>
              <span className="text-xs font-mono tabular-nums text-[#526783]">
                Source: {evalSummary.fileName}
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="border-b border-[#AFC5F5]/55 text-xs font-semibold text-[#526783]">
                    <th className="py-2.5 pr-4">Category</th>
                    <th className="py-2.5 pr-4 text-right">Precision</th>
                    <th className="py-2.5 pr-4 text-right">Recall</th>
                    <th className="py-2.5 pr-4 text-right">F1-Score</th>
                    <th className="py-2.5 text-right">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#AFC5F5]/45">
                  {evalSummary.perClassMetrics.map((row) => (
                    <tr key={row.label} className="hover:bg-[#E5ECFF]/70 transition-colors">
                      <td className="py-2.5 pr-4 font-medium text-[#203653]">{row.label}</td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#526783]">
                        {row.precision.toFixed(4)}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#526783]">
                        {row.recall.toFixed(4)}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums font-semibold text-[#203653]">
                        {row.f1Score.toFixed(4)}
                      </td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-[#526783]">
                        {row.support}
                      </td>
                    </tr>
                  ))}
                  <tr className="bg-[#D5E1FC]/75 font-medium">
                    <td className="py-2.5 pr-4 text-[#203653]">Macro Average</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#203653]">
                      {evalSummary.macroPrecision.toFixed(4)}
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#203653]">
                      {evalSummary.macroRecall.toFixed(4)}
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#4D6FAF]">
                      {evalSummary.macroF1.toFixed(4)}
                    </td>
                    <td className="py-2.5 text-right font-mono tabular-nums text-[#526783]">
                      {evalSummary.sampleCount}
                    </td>
                  </tr>
                  <tr className="bg-[#D5E1FC]/75 font-medium">
                    <td className="py-2.5 pr-4 text-[#203653]">Weighted Average</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#8294AE]">—</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#8294AE]">—</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#2F7D9E]">
                      {evalSummary.weightedF1.toFixed(4)}
                    </td>
                    <td className="py-2.5 text-right font-mono tabular-nums text-[#526783]">
                      {evalSummary.sampleCount}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {/* Multi-Model / BERT vs. DistilBERT Comparison */}
      <section className="neu-card p-7 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="font-display text-base font-semibold text-[#203653]">
              Transformer Model Comparison (BERT vs. DistilBERT)
            </h2>
            <p className="text-xs text-[#526783]">
              Compare multiple fine-tuned Transformer checkpoints evaluated on the identical held-out
              test split.
            </p>
          </div>
          {evaluationRuns.length >= 2 && (
            <span className="font-mono text-xs text-[#4D6FAF]">
              {sameTestSplitVerified
                ? `Shared Test Set Verified (N=${evaluationRuns[0].summary.sampleCount})`
                : 'Different N or Labels Across Runs'}
            </span>
          )}
        </div>

        {evaluationRuns.length < 2 || !comparisonPlot ? (
          <div className="neu-inset p-5 text-xs leading-relaxed text-[#526783]">
            <div className="font-semibold text-[#203653]">
              {evaluationRuns.length === 0
                ? 'Evaluate or Train At Least Two Models to Compare'
                : 'One Evaluation Saved — Evaluate or Train a Second Architecture'}
            </div>
            <p className="mt-1">
              Use <strong className="text-[#203653]">Train & Evaluate DistilBERT (6-Layer)</strong>{' '}
              and <strong className="text-[#203653]">Train & Evaluate BERT-Base (12-Layer)</strong>{' '}
              above to generate a direct side-by-side Plotly comparison on the same test set.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {!sameTestSplitVerified && (
              <div
                className="neu-inset p-3.5 text-xs text-[#5C4419]"
                style={{ background: '#F7F1E3' }}
              >
                Note: The saved evaluation runs have different sample counts or label sets. For a
                scientifically strict comparison, evaluate both models on the identical held-out test
                split.
              </div>
            )}

            <PlotlyChart data={comparisonPlot.data} layout={comparisonPlot.layout} />

            {bertRun && distilRun && (
              <div className="neu-inset p-5 space-y-3">
                <div className="text-xs font-semibold text-[#203653]">
                  Direct Comparison: {bertRun.modelName} vs. {distilRun.modelName}
                </div>
                <div className="grid gap-3 sm:grid-cols-4 text-xs">
                  {[
                    {
                      label: 'Δ Accuracy (BERT − DistilBERT)',
                      val: (bertRun.summary.accuracy - distilRun.summary.accuracy) * 100,
                      suffix: ' pp',
                    },
                    {
                      label: 'Δ Macro F1',
                      val: bertRun.summary.macroF1 - distilRun.summary.macroF1,
                      suffix: '',
                    },
                    {
                      label: 'Δ Weighted F1',
                      val: bertRun.summary.weightedF1 - distilRun.summary.weightedF1,
                      suffix: '',
                    },
                    {
                      label: 'Δ Macro ROC-AUC',
                      val:
                        bertRun.summary.rocAucMacro !== null &&
                        distilRun.summary.rocAucMacro !== null
                          ? bertRun.summary.rocAucMacro - distilRun.summary.rocAucMacro
                          : null,
                      suffix: '',
                    },
                  ].map((d) => (
                    <div key={d.label} className="neu-card p-3.5">
                      <div className="text-[11px] text-[#526783]">{d.label}</div>
                      <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-[#203653]">
                        {d.val === null
                          ? 'N/A'
                          : `${d.val >= 0 ? '+' : ''}${d.val.toFixed(d.suffix ? 2 : 4)}${d.suffix}`}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <p className="text-xs leading-relaxed text-[#526783]">
              <strong className="text-[#203653]">How to read this chart:</strong> Comparing
              DistilBERT (6 layers, faster inference) against BERT-base (12 layers, higher capacity)
              on the identical test split reveals whether extra model depth yields meaningful gains
              in Macro F1 and minority-class recall.
            </p>
          </div>
        )}
      </section>

      {/* Training & Validation Curves */}
      <section className="neu-card p-7 space-y-4">
        <div>
          <h2 className="font-display text-base font-semibold text-[#203653]">
            Training & Validation Curves
          </h2>
          <p className="text-xs text-[#526783]">
            Epoch loss and validation F1 progression saved automatically from Transformer training runs.
          </p>
        </div>

        {!trainingHistory || !historyPlot ? (
          <div className="neu-inset p-5 text-xs leading-relaxed text-[#526783]">
            <div className="font-semibold text-[#203653]">
              No Training Epoch Logs Recorded Yet
            </div>
            <p className="mt-1">
              Epoch training and validation loss curves populate automatically when you train{' '}
              <strong className="text-[#203653]">DistilBERT</strong> or{' '}
              <strong className="text-[#203653]">BERT-Base</strong> above.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid gap-6 lg:grid-cols-12 items-start">
              <div className="lg:col-span-6">
                <PlotlyChart data={historyPlot.data} layout={historyPlot.layout} />
              </div>
              <div className="lg:col-span-6 overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[#AFC5F5]/55 text-[#526783]">
                      <th className="py-2 pr-3">Epoch</th>
                      <th className="py-2 pr-3 text-right">Train Loss</th>
                      <th className="py-2 pr-3 text-right">Val Loss</th>
                      <th className="py-2 pr-3 text-right">Val Acc</th>
                      <th className="py-2 text-right">Val F1</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#AFC5F5]/45">
                    {trainingHistory.epochs.map((ep) => (
                      <tr key={ep.epoch} className="hover:bg-[#E5ECFF]/70 transition-colors">
                        <td className="py-2 pr-3 font-mono tabular-nums text-[#203653]">
                          {ep.epoch}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums text-[#526783]">
                          {ep.trainLoss !== null ? ep.trainLoss.toFixed(4) : '—'}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums text-[#526783]">
                          {ep.evalLoss !== null ? ep.evalLoss.toFixed(4) : '—'}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums text-[#526783]">
                          {ep.evalAccuracy !== null
                            ? `${(ep.evalAccuracy * 100).toFixed(2)}%`
                            : '—'}
                        </td>
                        <td className="py-2 text-right font-mono tabular-nums font-semibold text-[#203653]">
                          {ep.evalF1 !== null ? ep.evalF1.toFixed(4) : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <p className="text-xs leading-relaxed text-[#526783]">
              <strong className="text-[#203653]">How to read this chart:</strong> Both training and
              validation loss should decline together across epochs. If validation loss starts
              increasing while training loss keeps dropping, early stopping halts training to prevent
              memorizing training noise.
            </p>
          </div>
        )}
      </section>

      {/* Optional Advanced Options (External CLI CSV Uploads) */}
      <details className="group neu-card">
        <summary className="flex cursor-pointer list-none items-center justify-between px-7 py-5 text-xs font-semibold text-[#203653] select-none">
          <span>Advanced Options (Optional Custom Evaluation & Training History CSVs)</span>
          <ChevronDown className="h-4 w-4 text-[#4D6FAF] transition-transform group-open:rotate-180" />
        </summary>

        <div className="border-t border-[#AFC5F5]/55 p-7 space-y-5 text-xs">
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-3">
              <div className="font-semibold text-[#203653]">
                Inspect External Held-Out Predictions CSV
              </div>
              <p className="text-[#526783]">
                Optionally inspect held-out prediction files (<code className="font-mono">true_label</code>,{' '}
                <code className="font-mono">predicted_label</code>) exported from offline Python CLI
                scripts.
              </p>

              <div className="flex flex-wrap items-end gap-3">
                <div>
                  <label
                    htmlFor="run-preset-select"
                    className="block text-[11px] font-medium text-[#526783] mb-1"
                  >
                    Model Label
                  </label>
                  <select
                    id="run-preset-select"
                    value={presetLabel}
                    onChange={(e) =>
                      setPresetLabel(e.target.value as 'DistilBERT' | 'BERT-base' | 'custom')
                    }
                    className="neu-input px-3 py-2 text-xs text-[#203653]"
                  >
                    <option value="DistilBERT">DistilBERT</option>
                    <option value="BERT-base">BERT-base</option>
                    <option value="custom">Custom Model</option>
                  </select>
                </div>

                {presetLabel === 'custom' && (
                  <div>
                    <label
                      htmlFor="custom-run-input"
                      className="block text-[11px] font-medium text-[#526783] mb-1"
                    >
                      Custom Name
                    </label>
                    <input
                      id="custom-run-input"
                      type="text"
                      value={customLabel}
                      onChange={(e) => setCustomLabel(e.target.value)}
                      placeholder="e.g. RoBERTa"
                      className="neu-input w-36 px-3 py-2 text-xs text-[#203653]"
                    />
                  </div>
                )}

                <label className="neu-btn inline-flex cursor-pointer items-center gap-2 px-4 py-2 text-xs font-medium whitespace-nowrap">
                  <Upload className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
                  <span>Load Evaluation CSV</span>
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    onChange={handleUploadEvalCsv}
                    className="sr-only"
                  />
                </label>
              </div>

              {lastParsedCsv && activeRun && (
                <div className="pt-2 grid gap-3 sm:grid-cols-2">
                  <div>
                    <label htmlFor="true-col-select" className="block text-[#526783] mb-1">
                      Ground Truth Column
                    </label>
                    <select
                      id="true-col-select"
                      value={trueCol}
                      onChange={(e) => handleRecomputeColumns(e.target.value, predCol)}
                      className="neu-input w-full px-3 py-1.5 text-xs text-[#203653]"
                    >
                      {lastParsedCsv.data.headers.map((h) => (
                        <option key={h} value={h}>
                          {h}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label htmlFor="pred-col-select" className="block text-[#526783] mb-1">
                      Predicted Label Column
                    </label>
                    <select
                      id="pred-col-select"
                      value={predCol}
                      onChange={(e) => handleRecomputeColumns(trueCol, e.target.value)}
                      className="neu-input w-full px-3 py-1.5 text-xs text-[#203653]"
                    >
                      {lastParsedCsv.data.headers.map((h) => (
                        <option key={h} value={h}>
                          {h}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-3">
              <div className="font-semibold text-[#203653]">
                Inspect External Training History CSV
              </div>
              <p className="text-[#526783]">
                Optionally inspect a <code className="font-mono">training_history.csv</code> exported
                from an external PyTorch run.
              </p>
              <label className="neu-btn inline-flex cursor-pointer items-center gap-2 px-4 py-2 text-xs font-medium">
                <Upload className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
                <span>Load Training History CSV</span>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={handleUploadTrainingHistory}
                  className="sr-only"
                />
              </label>
              {historyError && (
                <div
                  className="neu-inset p-3 text-xs text-[#6B2A3A]"
                  style={{ background: '#F6E9ED' }}
                >
                  {historyError}
                </div>
              )}
            </div>
          </div>
        </div>
      </details>
    </div>
  );
}
