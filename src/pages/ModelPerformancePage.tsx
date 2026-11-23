import { useState, type ChangeEvent } from 'react';
import { Trash2, Upload } from 'lucide-react';
import { PageHeader } from '@/src/components/PageHeader';
import type { NamedEvaluationRun, TrainingHistorySummary } from '@/src/types/mindtrace';
import { parseCsvContent, type ParsedCsv } from '@/src/utils/csvParser';
import { computeEvaluationMetrics, parseTrainingHistoryCsv } from '@/src/utils/evaluationEngine';

interface ModelPerformancePageProps {
  evaluationRuns: NamedEvaluationRun[];
  activeRunId: string | null;
  onAddOrUpdateRun: (run: NamedEvaluationRun) => void;
  onSelectRun: (runId: string) => void;
  onRemoveRun: (runId: string) => void;
  trainingHistory: TrainingHistorySummary | null;
  onSetTrainingHistory: (history: TrainingHistorySummary | null) => void;
}

const SERIES_COLORS = ['#0F2942', '#0D7377', '#2563EB', '#B45309', '#64748B'];

export function ModelPerformancePage({
  evaluationRuns,
  activeRunId,
  onAddOrUpdateRun,
  onSelectRun,
  onRemoveRun,
  trainingHistory,
  onSetTrainingHistory,
}: ModelPerformancePageProps) {
  const [runLabelInput, setRunLabelInput] = useState('DistilBERT');
  const [lastParsedCsv, setLastParsedCsv] = useState<{ fileName: string; data: ParsedCsv } | null>(null);
  const [trueCol, setTrueCol] = useState('true_label');
  const [predCol, setPredCol] = useState('predicted_label');
  const [evalError, setEvalError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<{
    trueLabel: string;
    predLabel: string;
    count: number;
  } | null>(null);

  const activeRun =
    evaluationRuns.find((r) => r.id === activeRunId) ?? evaluationRuns[0] ?? null;
  const evalSummary = activeRun?.summary ?? null;

  async function handleUploadEvalCsv(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setEvalError(null);
    setSelectedCell(null);
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
      const cleanName = runLabelInput.trim() || file.name.replace(/\.csv$/i, '');
      const runId = `${cleanName.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${Date.now()}`;
      onAddOrUpdateRun({ id: runId, modelName: cleanName, summary });
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : 'Unable to evaluate uploaded CSV.');
    } finally {
      e.target.value = '';
    }
  }

  function handleRecomputeColumns(nextTrue: string, nextPred: string) {
    setTrueCol(nextTrue);
    setPredCol(nextPred);
    setSelectedCell(null);
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

  const maxCellCount = evalSummary
    ? Math.max(...evalSummary.confusionMatrix.flat(), 1)
    : 1;

  // Compute loss chart bounds if training history exists
  const lossValues = trainingHistory
    ? trainingHistory.epochs
        .flatMap((ep) => [ep.trainLoss, ep.evalLoss])
        .filter((v): v is number => v !== null)
    : [];
  const minLoss = lossValues.length > 0 ? Math.min(...lossValues) * 0.9 : 0;
  const maxLoss = lossValues.length > 0 ? Math.max(...lossValues) * 1.1 : 1;
  const lossSpan = Math.max(maxLoss - minLoss, 1e-4);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Performance"
        description="Audit held-out accuracy, precision, recall, F1-score, confusion matrix, multi-model comparisons, and epoch-level training curves from real exported logs."
      />

      {/* Held-Out Evaluation Uploader & Run Manager */}
      <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
          <div>
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Held-Out Test Predictions
            </h2>
            <p className="mt-1 text-xs text-[#64748B]">
              Upload one or more held-out CSV files (<code className="font-mono">true_label</code>,{' '}
              <code className="font-mono">predicted_label</code>, optional{' '}
              <code className="font-mono">prob_&lt;label&gt;</code>) to compute metrics and compare
              models.
            </p>
          </div>

          <div className="flex flex-wrap items-end gap-3">
            <div>
              <label htmlFor="run-label-input" className="block text-[11px] font-medium text-[#475569] mb-1">
                Model Label
              </label>
              <input
                id="run-label-input"
                type="text"
                value={runLabelInput}
                onChange={(e) => setRunLabelInput(e.target.value)}
                placeholder="e.g. DistilBERT"
                className="w-40 rounded-lg border border-[#CBD5E1] bg-white px-3 py-2 text-xs text-[#0F172A] min-h-[40px]"
              />
            </div>
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-[#0F2942] px-4 py-2 text-xs font-medium text-white hover:bg-[#173A5E] transition-colors whitespace-nowrap shrink-0 min-h-[40px]">
              <Upload className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Upload Evaluation CSV</span>
              <input type="file" accept=".csv,text/csv" onChange={handleUploadEvalCsv} className="sr-only" />
            </label>
          </div>
        </div>

        {evaluationRuns.length > 0 && (
          <div className="pt-3 border-t border-[#E2E8F0] flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-[#64748B] mr-1">Loaded Runs:</span>
            {evaluationRuns.map((run) => {
              const isSelected = activeRun?.id === run.id;
              return (
                <div
                  key={run.id}
                  className={`inline-flex items-center rounded-lg border text-xs ${
                    isSelected
                      ? 'border-[#0F2942] bg-[#0F2942] text-white'
                      : 'border-[#E2E8F0] bg-[#F8FAFC] text-[#0F172A]'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => {
                      onSelectRun(run.id);
                      setSelectedCell(null);
                    }}
                    className="px-3 py-1.5 font-medium whitespace-nowrap"
                  >
                    {run.modelName} (N={run.summary.sampleCount})
                  </button>
                  <button
                    type="button"
                    onClick={() => onRemoveRun(run.id)}
                    aria-label={`Remove ${run.modelName}`}
                    className={`pr-2.5 pl-1 py-1.5 opacity-80 hover:opacity-100 ${
                      isSelected ? 'text-white' : 'text-[#64748B]'
                    }`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {lastParsedCsv && activeRun && (
          <div className="pt-3 border-t border-[#E2E8F0] grid gap-4 sm:grid-cols-3 text-xs">
            <div>
              <span className="block text-[#64748B] mb-1">Latest CSV File</span>
              <span className="font-mono font-medium text-[#0F172A] break-all">
                {lastParsedCsv.fileName}
              </span>
            </div>
            <div>
              <label htmlFor="true-col-select" className="block text-[#64748B] mb-1">
                Ground Truth Column
              </label>
              <select
                id="true-col-select"
                value={trueCol}
                onChange={(e) => handleRecomputeColumns(e.target.value, predCol)}
                className="w-full rounded-lg border border-[#CBD5E1] bg-white px-3 py-1.5 text-xs text-[#0F172A]"
              >
                {lastParsedCsv.data.headers.map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="pred-col-select" className="block text-[#64748B] mb-1">
                Predicted Label Column
              </label>
              <select
                id="pred-col-select"
                value={predCol}
                onChange={(e) => handleRecomputeColumns(trueCol, e.target.value)}
                className="w-full rounded-lg border border-[#CBD5E1] bg-white px-3 py-1.5 text-xs text-[#0F172A]"
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

        {evalError && (
          <div className="rounded-lg bg-[#FEF2F2] p-3 text-xs text-[#991B1B]">{evalError}</div>
        )}
      </section>

      {/* Metrics + Confusion Matrix + ROC-AUC */}
      {!evalSummary ? (
        <section className="rounded-xl border border-[#E2E8F0] bg-white p-8 max-w-3xl">
          <div className="text-xs font-mono text-[#0D7377]">
            Empty State · No Held-Out Evaluation Data Loaded
          </div>
          <h3 className="mt-2 font-display text-lg font-semibold text-[#0F172A]">
            Supply Real Held-Out Predictions to Compute Evaluation Metrics
          </h3>
          <p className="mt-2 text-sm leading-relaxed text-[#475569]">
            MindTrace AI never displays synthetic Accuracy, Precision, Recall, F1-score, or
            Confusion Matrix figures. Run <code className="font-mono text-xs">python -m src.train</code>{' '}
            to produce <code className="font-mono text-xs">held_out_predictions.csv</code> and upload
            it above.
          </p>
        </section>
      ) : (
        <>
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {[
              { label: 'Held-Out Samples', value: String(evalSummary.sampleCount) },
              { label: 'Accuracy', value: `${(evalSummary.accuracy * 100).toFixed(2)}%` },
              { label: 'Macro Precision', value: evalSummary.macroPrecision.toFixed(4) },
              { label: 'Macro Recall', value: evalSummary.macroRecall.toFixed(4) },
              {
                label: evalSummary.rocAucMacro !== null ? 'Macro F1 / ROC-AUC' : 'Macro F1-Score',
                value:
                  evalSummary.rocAucMacro !== null
                    ? `${evalSummary.macroF1.toFixed(4)} / ${evalSummary.rocAucMacro.toFixed(4)}`
                    : evalSummary.macroF1.toFixed(4),
              },
            ].map((kpi) => (
              <div key={kpi.label} className="rounded-xl border border-[#E2E8F0] bg-white p-5">
                <div className="text-xs text-[#64748B]">{kpi.label}</div>
                <div className="mt-1.5 font-mono text-xl font-semibold tabular-nums text-[#0F2942]">
                  {kpi.value}
                </div>
              </div>
            ))}
          </section>

          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-display text-base font-semibold text-[#0F172A]">
                  Confusion Matrix ({activeRun?.modelName})
                </h3>
                <span className="text-xs text-[#64748B]">Rows: True · Cols: Predicted</span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-xs">
                  <thead>
                    <tr>
                      <th className="p-2 text-left text-[#64748B]">True \ Pred</th>
                      {evalSummary.labels.map((lbl) => (
                        <th
                          key={lbl}
                          className="p-2 text-center font-medium text-[#0F172A] truncate max-w-[100px]"
                          title={lbl}
                        >
                          {lbl}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {evalSummary.labels.map((trueLbl, rIdx) => (
                      <tr key={trueLbl}>
                        <td
                          className="p-2 font-medium text-[#0F172A] truncate max-w-[110px]"
                          title={trueLbl}
                        >
                          {trueLbl}
                        </td>
                        {evalSummary.labels.map((predLbl, cIdx) => {
                          const count = evalSummary.confusionMatrix[rIdx][cIdx];
                          const intensity = count / maxCellCount;
                          const isDiag = rIdx === cIdx;
                          const bg = isDiag
                            ? `rgba(15, 41, 66, ${(0.1 + 0.78 * intensity).toFixed(2)})`
                            : `rgba(180, 83, 9, ${(0.04 + 0.55 * intensity).toFixed(2)})`;
                          const textLight = isDiag && intensity > 0.55;
                          return (
                            <td key={`${trueLbl}-${predLbl}`} className="p-1">
                              <button
                                type="button"
                                onClick={() =>
                                  setSelectedCell({
                                    trueLabel: trueLbl,
                                    predLabel: predLbl,
                                    count,
                                  })
                                }
                                style={{ backgroundColor: bg }}
                                className={`w-full rounded-md py-3 px-2 font-mono font-semibold tabular-nums transition-opacity hover:opacity-90 ${
                                  textLight ? 'text-white' : 'text-[#0F172A]'
                                }`}
                              >
                                {count}
                              </button>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {selectedCell && (
                <div className="rounded-lg bg-[#F8FAFC] p-3 text-xs text-[#334155] font-mono tabular-nums">
                  True: {selectedCell.trueLabel} · Predicted: {selectedCell.predLabel} · Count:{' '}
                  {selectedCell.count}
                </div>
              )}
            </section>

            <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-display text-base font-semibold text-[#0F172A]">
                  ROC-AUC Curve (One-vs-Rest)
                </h3>
                {evalSummary.rocAucMacro !== null && (
                  <span className="font-mono text-xs tabular-nums text-[#0D7377]">
                    Macro AUC: {evalSummary.rocAucMacro.toFixed(4)}
                  </span>
                )}
              </div>

              {evalSummary.rocCurves.length === 0 ? (
                <div className="rounded-lg bg-[#F8FAFC] p-6 text-xs leading-relaxed text-[#475569]">
                  ROC-AUC curves require class probability columns named{' '}
                  <code className="font-mono">prob_&lt;label&gt;</code> for every label in the CSV.
                </div>
              ) : (
                <div className="space-y-3">
                  <svg
                    viewBox="0 0 360 240"
                    role="img"
                    aria-label="Receiver Operating Characteristic curve"
                    className="w-full h-auto rounded-lg bg-[#F8FAFC] border border-[#E2E8F0]"
                  >
                    <line x1="44" y1="204" x2="336" y2="204" stroke="#CBD5E1" strokeWidth="1" />
                    <line x1="44" y1="20" x2="44" y2="204" stroke="#CBD5E1" strokeWidth="1" />
                    <line
                      x1="44"
                      y1="204"
                      x2="336"
                      y2="20"
                      stroke="#94A3B8"
                      strokeWidth="1"
                      strokeDasharray="4 4"
                    />
                    {evalSummary.rocCurves.map((series, idx) => {
                      const d = series.points
                        .map((pt, pIdx) => {
                          const x = 44 + pt.fpr * 292;
                          const y = 204 - pt.tpr * 184;
                          return `${pIdx === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
                        })
                        .join(' ');
                      return (
                        <path
                          key={series.label}
                          d={d}
                          fill="none"
                          stroke={SERIES_COLORS[idx % SERIES_COLORS.length]}
                          strokeWidth="2.2"
                        />
                      );
                    })}
                    <text x="190" y="228" textAnchor="middle" fontSize="10" fill="#64748B">
                      False Positive Rate
                    </text>
                    <text
                      x="14"
                      y="112"
                      textAnchor="middle"
                      fontSize="10"
                      fill="#64748B"
                      transform="rotate(-90 14 112)"
                    >
                      True Positive Rate
                    </text>
                  </svg>

                  <div className="flex flex-wrap gap-4 text-xs">
                    {evalSummary.rocCurves.map((series, idx) => (
                      <div key={series.label} className="inline-flex items-center gap-1.5">
                        <span
                          className="inline-block h-2.5 w-2.5 rounded-xs"
                          style={{ backgroundColor: SERIES_COLORS[idx % SERIES_COLORS.length] }}
                        />
                        <span className="text-[#334155]">
                          {series.label}:{' '}
                          <span className="font-mono tabular-nums font-semibold">
                            {series.auc.toFixed(3)}
                          </span>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          </div>

          <section className="rounded-xl border border-[#E2E8F0] bg-white p-6">
            <div className="flex items-center justify-between pb-3 border-b border-[#E2E8F0]">
              <h3 className="font-display text-base font-semibold text-[#0F172A]">
                Per-Class Classification Report
              </h3>
              <span className="text-xs font-mono tabular-nums text-[#64748B]">
                Weighted F1: {evalSummary.weightedF1.toFixed(4)}
              </span>
            </div>
            <div className="overflow-x-auto mt-3">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="border-b border-[#E2E8F0] text-xs font-semibold text-[#64748B]">
                    <th className="py-2.5 pr-4">Category</th>
                    <th className="py-2.5 pr-4 text-right">Precision</th>
                    <th className="py-2.5 pr-4 text-right">Recall</th>
                    <th className="py-2.5 pr-4 text-right">F1-Score</th>
                    <th className="py-2.5 text-right">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E8F0]">
                  {evalSummary.perClassMetrics.map((row) => (
                    <tr key={row.label} className="hover:bg-[#F8FAFC]">
                      <td className="py-2.5 pr-4 font-medium text-[#0F172A]">{row.label}</td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#334155]">
                        {row.precision.toFixed(4)}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-[#334155]">
                        {row.recall.toFixed(4)}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums font-semibold text-[#0F2942]">
                        {row.f1Score.toFixed(4)}
                      </td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-[#475569]">
                        {row.support}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {/* Model Comparison Charts */}
      <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
        <div>
          <h2 className="font-display text-base font-semibold text-[#0F172A]">
            Model Comparison Chart
          </h2>
          <p className="text-xs text-[#64748B]">
            Side-by-side comparison of Accuracy, Macro Precision, Macro Recall, and Macro F1 across
            uploaded evaluation runs.
          </p>
        </div>

        {evaluationRuns.length < 2 ? (
          <div className="rounded-lg bg-[#F8FAFC] p-5 text-xs leading-relaxed text-[#475569]">
            <div className="font-semibold text-[#0F172A]">
              {evaluationRuns.length === 0
                ? 'Empty State — Upload Held-Out CSVs to Compare Models'
                : 'One Run Loaded — Upload a Second Model CSV (e.g., BERT-base) to Compare'}
            </div>
            <p className="mt-1">
              Change the Model Label above (for example from <code className="font-mono">DistilBERT</code>{' '}
              to <code className="font-mono">BERT-base</code>) and upload another held-out
              predictions CSV to render a real side-by-side comparison chart.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {(
              [
                ['Accuracy', (r: NamedEvaluationRun) => r.summary.accuracy],
                ['Macro Precision', (r: NamedEvaluationRun) => r.summary.macroPrecision],
                ['Macro Recall', (r: NamedEvaluationRun) => r.summary.macroRecall],
                ['Macro F1-Score', (r: NamedEvaluationRun) => r.summary.macroF1],
              ] as [string, (r: NamedEvaluationRun) => number][]
            ).map(([metricLabel, getter]) => (
              <div key={metricLabel} className="space-y-2">
                <div className="text-xs font-semibold text-[#0F172A]">{metricLabel}</div>
                <div className="space-y-1.5">
                  {evaluationRuns.map((run, idx) => {
                    const val = getter(run);
                    return (
                      <div key={`${metricLabel}-${run.id}`} className="grid grid-cols-12 items-center gap-2 text-xs">
                        <div className="col-span-3 truncate text-[#334155]">{run.modelName}</div>
                        <div className="col-span-7 h-3.5 rounded bg-[#F1F5F9] overflow-hidden">
                          <div
                            className="h-full rounded"
                            style={{
                              width: `${Math.max(2, val * 100)}%`,
                              backgroundColor: SERIES_COLORS[idx % SERIES_COLORS.length],
                            }}
                          />
                        </div>
                        <div className="col-span-2 text-right font-mono tabular-nums text-[#0F172A]">
                          {(val * 100).toFixed(2)}%
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Training and Validation Performance */}
      <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Training & Validation Performance
            </h2>
            <p className="text-xs text-[#64748B]">
              Upload <code className="font-mono">training_history.csv</code> (exported by{' '}
              <code className="font-mono">src/train.py</code>) containing{' '}
              <code className="font-mono">epoch</code>, <code className="font-mono">train_loss</code>,{' '}
              <code className="font-mono">eval_loss</code>, and optional{' '}
              <code className="font-mono">eval_accuracy</code> / <code className="font-mono">eval_f1</code>.
            </p>
          </div>
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-[#CBD5E1] bg-white px-4 py-2 text-xs font-medium text-[#0F172A] hover:bg-[#F1F5F9] whitespace-nowrap shrink-0 min-h-[40px]">
            <Upload className="h-3.5 w-3.5" aria-hidden="true" />
            <span>{trainingHistory ? 'Replace History CSV' : 'Upload Training History CSV'}</span>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleUploadTrainingHistory}
              className="sr-only"
            />
          </label>
        </div>

        {historyError && (
          <div className="rounded-lg bg-[#FEF2F2] p-3 text-xs text-[#991B1B]">{historyError}</div>
        )}

        {!trainingHistory ? (
          <div className="rounded-lg bg-[#F8FAFC] p-5 text-xs leading-relaxed text-[#475569]">
            <div className="font-semibold text-[#0F172A]">
              Empty State — Training Logs Not Supplied
            </div>
            <p className="mt-1">
              Epoch loss and validation curves appear only when real training logs are uploaded.
            </p>
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-12 items-start">
            <div className="lg:col-span-6 space-y-2">
              <div className="text-xs font-medium text-[#334155]">
                Training vs. Validation Loss ({trainingHistory.fileName})
              </div>
              <svg
                viewBox="0 0 360 220"
                role="img"
                aria-label="Training and validation loss per epoch"
                className="w-full h-auto rounded-lg bg-[#F8FAFC] border border-[#E2E8F0]"
              >
                <line x1="44" y1="184" x2="336" y2="184" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="44" y1="20" x2="44" y2="184" stroke="#CBD5E1" strokeWidth="1" />
                {(['trainLoss', 'evalLoss'] as const).map((field, sIdx) => {
                  const validPts = trainingHistory.epochs.filter((ep) => ep[field] !== null);
                  if (validPts.length === 0) return null;
                  const minEp = trainingHistory.epochs[0].epoch;
                  const maxEp = trainingHistory.epochs[trainingHistory.epochs.length - 1].epoch;
                  const epSpan = Math.max(maxEp - minEp, 1);
                  const d = validPts
                    .map((ep, idx) => {
                      const x = 44 + ((ep.epoch - minEp) / epSpan) * 292;
                      const y = 184 - (((ep[field] as number) - minLoss) / lossSpan) * 164;
                      return `${idx === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
                    })
                    .join(' ');
                  return (
                    <path
                      key={field}
                      d={d}
                      fill="none"
                      stroke={sIdx === 0 ? '#0F2942' : '#0D7377'}
                      strokeWidth="2.2"
                    />
                  );
                })}
                <text x="190" y="208" textAnchor="middle" fontSize="10" fill="#64748B">
                  Epoch
                </text>
                <text
                  x="14"
                  y="102"
                  textAnchor="middle"
                  fontSize="10"
                  fill="#64748B"
                  transform="rotate(-90 14 102)"
                >
                  Loss
                </text>
              </svg>
              <div className="flex gap-4 text-xs text-[#475569]">
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-2.5 rounded-xs bg-[#0F2942]" />
                  <span>Training Loss</span>
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-2.5 rounded-xs bg-[#0D7377]" />
                  <span>Validation Loss</span>
                </span>
              </div>
            </div>

            <div className="lg:col-span-6 overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#E2E8F0] text-[#64748B]">
                    <th className="py-2 pr-3">Epoch</th>
                    <th className="py-2 pr-3 text-right">Train Loss</th>
                    <th className="py-2 pr-3 text-right">Val Loss</th>
                    <th className="py-2 pr-3 text-right">Val Acc</th>
                    <th className="py-2 text-right">Val F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E8F0]">
                  {trainingHistory.epochs.map((ep) => (
                    <tr key={ep.epoch}>
                      <td className="py-2 pr-3 font-mono tabular-nums text-[#0F172A]">{ep.epoch}</td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums text-[#334155]">
                        {ep.trainLoss !== null ? ep.trainLoss.toFixed(4) : '—'}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums text-[#334155]">
                        {ep.evalLoss !== null ? ep.evalLoss.toFixed(4) : '—'}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums text-[#334155]">
                        {ep.evalAccuracy !== null ? `${(ep.evalAccuracy * 100).toFixed(2)}%` : '—'}
                      </td>
                      <td className="py-2 text-right font-mono tabular-nums font-semibold text-[#0F2942]">
                        {ep.evalF1 !== null ? ep.evalF1.toFixed(4) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
