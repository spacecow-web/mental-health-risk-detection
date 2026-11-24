import { useEffect, useMemo, useState, type ChangeEvent } from 'react';
import { ArrowRight, ChevronDown, RefreshCw, RotateCcw, Upload } from 'lucide-react';
import type { Data, Layout } from 'plotly.js';
import { PageHeader } from '@/src/components/PageHeader';
import { PlotlyChart } from '@/src/components/PlotlyChart';
import type {
  CorpusRecord,
  DatasetSummary,
  LoadedCorpusData,
} from '@/src/types/mindtrace';
import { parseCsvContent, type ParsedCsv } from '@/src/utils/csvParser';
import { computeDatasetAnalytics } from '@/src/utils/datasetAnalytics';

interface DatasetAnalyticsPageProps {
  datasetSummary: DatasetSummary | null;
  loadingDefaultDataset: boolean;
  defaultDatasetError: string | null;
  isCustomDatasetActive: boolean;
  onRetryDefaultDataset: () => void;
  onApplyCustomDataset: (summary: DatasetSummary, corpus: LoadedCorpusData) => void;
  onOpenModelPerformance?: () => void;
  embedded?: boolean;
}

const BLUE_BAR_COLORS = ['#4D6FAF', '#72B9D8', '#7799D8', '#91B9E8', '#AFC5F5'];

function buildCorpusRows(
  rows: Record<string, string>[],
  tCol: string,
  lCol: string,
  sCol?: string
): CorpusRecord[] {
  const result: CorpusRecord[] = [];
  for (const r of rows) {
    const text = (r[tCol] ?? '').trim();
    const label = (r[lCol] ?? '').trim();
    if (!text || !label) continue;
    result.push({
      text,
      label,
      split: sCol ? (r[sCol] ?? '').trim() || undefined : undefined,
    });
  }
  return result;
}

export function DatasetAnalyticsPage({
  datasetSummary,
  loadingDefaultDataset,
  defaultDatasetError,
  isCustomDatasetActive,
  onRetryDefaultDataset,
  onApplyCustomDataset,
  onOpenModelPerformance,
  embedded = false,
}: DatasetAnalyticsPageProps) {
  const [parsedCsv, setParsedCsv] = useState<{ fileName: string; data: ParsedCsv } | null>(null);
  const [textCol, setTextCol] = useState('text');
  const [labelCol, setLabelCol] = useState('label');
  const [splitCol, setSplitCol] = useState('');
  const [datasetNameInput, setDatasetNameInput] = useState('');
  const [datasetSourceInput, setDatasetSourceInput] = useState('Custom local research CSV');
  const [selectedClass, setSelectedClass] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (datasetSummary && datasetSummary.classCounts.length > 0) {
      if (!selectedClass || !datasetSummary.topTermsByClass[selectedClass]) {
        setSelectedClass(datasetSummary.classCounts[0].label);
      }
    }
  }, [datasetSummary, selectedClass]);

  function handleRestoreDefault() {
    setParsedCsv(null);
    setErrorMsg(null);
    onRetryDefaultDataset();
  }

  async function handleFileUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    try {
      const content = await file.text();
      const parsed = parseCsvContent(content);
      const headers = parsed.headers;
      const detectedText =
        headers.find((h) => h.toLowerCase() === 'text') ??
        headers.find((h) => h.toLowerCase().includes('post') || h.toLowerCase().includes('text')) ??
        headers[0];
      const detectedLabel =
        headers.find((h) => h.toLowerCase() === 'label') ??
        headers.find((h) => h.toLowerCase().includes('label') || h.toLowerCase().includes('class')) ??
        headers[Math.min(1, headers.length - 1)];
      const detectedSplit =
        headers.find((h) => ['split', 'subset', 'partition'].includes(h.toLowerCase())) ?? '';

      const defaultName = file.name.replace(/\.csv$/i, '');
      setParsedCsv({ fileName: file.name, data: parsed });
      setTextCol(detectedText);
      setLabelCol(detectedLabel);
      setSplitCol(detectedSplit);
      setDatasetNameInput(defaultName);

      const summary = computeDatasetAnalytics(
        parsed.rows,
        detectedText,
        detectedLabel,
        file.name,
        detectedSplit || undefined,
        defaultName,
        datasetSourceInput
      );
      onApplyCustomDataset(summary, {
        datasetName: summary.datasetName,
        datasetSource: summary.datasetSource,
        fileName: file.name,
        rows: buildCorpusRows(parsed.rows, detectedText, detectedLabel, detectedSplit || undefined),
      });
      setSelectedClass(summary.classCounts[0]?.label ?? '');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unable to parse dataset CSV.');
    } finally {
      e.target.value = '';
    }
  }

  function handleMetadataChange(nextName: string, nextSource: string) {
    setDatasetNameInput(nextName);
    setDatasetSourceInput(nextSource);
    if (datasetSummary && parsedCsv) {
      const updated: DatasetSummary = {
        ...datasetSummary,
        datasetName: nextName.trim() || datasetSummary.fileName.replace(/\.csv$/i, ''),
        datasetSource: nextSource.trim() || 'Custom local research CSV',
      };
      onApplyCustomDataset(updated, {
        datasetName: updated.datasetName,
        datasetSource: updated.datasetSource,
        fileName: parsedCsv.fileName,
        rows: buildCorpusRows(parsedCsv.data.rows, textCol, labelCol, splitCol || undefined),
      });
    }
  }

  function handleRecompute(
    nextTextCol: string,
    nextLabelCol: string,
    nextSplitCol: string
  ) {
    setTextCol(nextTextCol);
    setLabelCol(nextLabelCol);
    setSplitCol(nextSplitCol);
    if (!parsedCsv) return;
    try {
      setErrorMsg(null);
      const summary = computeDatasetAnalytics(
        parsedCsv.data.rows,
        nextTextCol,
        nextLabelCol,
        parsedCsv.fileName,
        nextSplitCol || undefined,
        datasetNameInput,
        datasetSourceInput
      );
      onApplyCustomDataset(summary, {
        datasetName: summary.datasetName,
        datasetSource: summary.datasetSource,
        fileName: parsedCsv.fileName,
        rows: buildCorpusRows(
          parsedCsv.data.rows,
          nextTextCol,
          nextLabelCol,
          nextSplitCol || undefined
        ),
      });
      setSelectedClass(summary.classCounts[0]?.label ?? '');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Invalid column mapping.');
    }
  }

  const classPlot = useMemo<{ data: Data[]; layout: Partial<Layout> } | null>(() => {
    if (!datasetSummary) return null;
    return {
      data: [
        {
          type: 'bar',
          x: datasetSummary.classCounts.map((c) => c.label),
          y: datasetSummary.classCounts.map((c) => c.count),
          text: datasetSummary.classCounts.map((c) => `${c.count} (${c.sharePercent.toFixed(1)}%)`),
          textposition: 'auto',
          textfont: { color: '#203653', size: 12 },
          marker: {
            color: datasetSummary.classCounts.map(
              (_, i) => BLUE_BAR_COLORS[i % BLUE_BAR_COLORS.length]
            ),
            line: { color: '#EAF0FF', width: 1 },
          },
          hovertemplate: '%{x}<br>Records: %{y}<extra></extra>',
        },
      ],
      layout: {
        height: 320,
        xaxis: { title: { text: 'Category Label' } },
        yaxis: { title: { text: 'Usable Records' } },
        margin: { l: 56, r: 20, t: 20, b: 56 },
      },
    };
  }, [datasetSummary]);

  const lengthPlot = useMemo<{ data: Data[]; layout: Partial<Layout> } | null>(() => {
    if (!datasetSummary) return null;
    return {
      data: [
        {
          type: 'bar',
          x: datasetSummary.histogramBins.map((b) => `${b.rangeLabel} w`),
          y: datasetSummary.histogramBins.map((b) => b.count),
          marker: {
            color: '#72B9D8',
            line: { color: '#EAF0FF', width: 1 },
          },
          hovertemplate: 'Range: %{x}<br>Posts: %{y}<extra></extra>',
        },
      ],
      layout: {
        height: 320,
        xaxis: { title: { text: 'Word Count Bin' } },
        yaxis: { title: { text: 'Number of Posts' } },
        margin: { l: 56, r: 20, t: 20, b: 56 },
      },
    };
  }, [datasetSummary]);

  const splitPlot = useMemo<{ data: Data[]; layout: Partial<Layout> } | null>(() => {
    if (!datasetSummary) return null;
    const labels = datasetSummary.splitBreakdown
      ? datasetSummary.splitBreakdown.map((s) => s.split)
      : ['Train (70%)', 'Validation (15%)', 'Test (15%)'];
    const counts = datasetSummary.splitBreakdown
      ? datasetSummary.splitBreakdown.map((s) => s.count)
      : (() => {
          const tr = Math.round(datasetSummary.validRows * 0.7);
          const va = Math.round(datasetSummary.validRows * 0.15);
          return [tr, va, Math.max(0, datasetSummary.validRows - tr - va)];
        })();

    return {
      data: [
        {
          type: 'bar',
          x: labels,
          y: counts,
          text: counts.map((c) => String(c)),
          textposition: 'auto',
          textfont: { color: '#203653', size: 12 },
          marker: {
            color: labels.map((_, i) => BLUE_BAR_COLORS[i % BLUE_BAR_COLORS.length]),
            line: { color: '#EAF0FF', width: 1 },
          },
          hovertemplate: '%{x}: %{y} records<extra></extra>',
        },
      ],
      layout: {
        height: 290,
        xaxis: { title: { text: 'Dataset Split' } },
        yaxis: { title: { text: 'Records' } },
        margin: { l: 56, r: 20, t: 20, b: 56 },
      },
    };
  }, [datasetSummary]);

  const imbalanceSeverity = !datasetSummary
    ? ''
    : datasetSummary.imbalanceRatio < 1.5
    ? 'Balanced (< 1.5x)'
    : datasetSummary.imbalanceRatio <= 3.0
    ? 'Moderate Imbalance (1.5x – 3.0x)'
    : 'Severe Imbalance (> 3.0x)';

  return (
    <div className="space-y-7">
      {!embedded && (
        <PageHeader
          title="Dataset Analytics"
          description="Automatic inspection of the configured benchmark corpus: record counts, class imbalance, train/validation/test splits, text length distributions, missing values, and duplicate diagnostics."
        />
      )}

      {errorMsg && (
        <div
          className="neu-inset p-3.5 text-xs text-[#6B2A3A]"
          style={{ background: '#F6E9ED' }}
        >
          {errorMsg}
        </div>
      )}

      {loadingDefaultDataset && !datasetSummary ? (
        <section className="neu-card p-8 max-w-3xl space-y-2">
          <div className="text-xs font-mono text-[#4D6FAF]">Loading Configured Dataset...</div>
          <p className="text-sm text-[#526783]">
            Reading and analyzing the backend default research dataset automatically.
          </p>
        </section>
      ) : !datasetSummary ? (
        <section className="neu-card p-8 max-w-3xl space-y-4">
          <div className="text-xs font-mono text-[#5C4419]">
            Default Research Dataset Unavailable
          </div>
          <h2 className="font-display text-lg font-semibold text-[#203653]">
            Unable to Load the Configured Backend Corpus
          </h2>
          <p className="text-sm leading-relaxed text-[#526783]">
            {defaultDatasetError ||
              'Place a labeled CSV file at data/default_dataset.csv or verify outbound access to the configured Hugging Face corpus, then click Retry.'}
          </p>
          <button
            type="button"
            onClick={handleRestoreDefault}
            className="neu-btn-primary inline-flex items-center gap-2 px-4 py-2.5 text-xs min-h-[40px]"
          >
            <RefreshCw className="h-3.5 w-3.5 text-white" aria-hidden="true" />
            <span>Retry Default Dataset</span>
          </button>
        </section>
      ) : (
        <>
          {/* Dataset Overview KPIs */}
          <section className="neu-card-elevated p-7 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-[#AFC5F5]/55 text-xs">
              <div>
                <span className="text-[#526783]">Active Corpus: </span>
                <span className="font-semibold text-[#203653]">{datasetSummary.datasetName}</span>
                <span className="mx-2 text-[#91B9E8]" aria-hidden="true">
                  ·
                </span>
                <span className="text-[#526783]">Source: </span>
                <span className="text-[#203653]">{datasetSummary.datasetSource}</span>
              </div>

              <div className="flex flex-wrap items-center gap-2.5">
                {isCustomDatasetActive && (
                  <button
                    type="button"
                    onClick={handleRestoreDefault}
                    className="neu-btn inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium"
                  >
                    <RotateCcw className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
                    <span>Restore Default Dataset</span>
                  </button>
                )}
                {onOpenModelPerformance && (
                  <button
                    type="button"
                    onClick={onOpenModelPerformance}
                    className="neu-btn-primary inline-flex items-center gap-1.5 px-4 py-2 text-xs"
                  >
                    <span>Evaluate or Train Transformers</span>
                    <ArrowRight className="h-3.5 w-3.5 text-white" aria-hidden="true" />
                  </button>
                )}
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
              {[
                { label: 'Total Records', value: String(datasetSummary.totalRows) },
                { label: 'Usable Records', value: String(datasetSummary.validRows) },
                { label: 'Number of Classes', value: String(datasetSummary.numClasses) },
                {
                  label: 'Missing Values',
                  value: String(datasetSummary.missingTextRows + datasetSummary.missingLabelRows),
                },
                { label: 'Duplicate Records', value: String(datasetSummary.duplicateTexts) },
                { label: 'Imbalance Ratio', value: `${datasetSummary.imbalanceRatio.toFixed(2)}x` },
              ].map((stat) => (
                <div key={stat.label} className="neu-inset p-4">
                  <div className="text-xs text-[#526783]">{stat.label}</div>
                  <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-[#203653]">
                    {stat.value}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Interactive Plotly Class Distribution & Text Length Distribution */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-6 neu-card p-7 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Class Distribution
                </h2>
                <span className="font-mono text-xs tabular-nums text-[#526783]">
                  {datasetSummary.numClasses} classes
                </span>
              </div>
              {classPlot && <PlotlyChart data={classPlot.data} layout={classPlot.layout} />}
              <p className="text-xs leading-relaxed text-[#526783]">
                <strong className="text-[#203653]">What this shows:</strong> How many usable posts
                belong to each category. Balanced bars help the Transformer learn every category
                evenly without favouring only the most common label.
              </p>
            </section>

            <section className="lg:col-span-6 neu-card p-7 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Text Length Distribution
                </h2>
                <span className="text-xs font-mono tabular-nums text-[#526783]">
                  Median: {datasetSummary.lengthStats.medianWords.toFixed(1)} w · P95:{' '}
                  {datasetSummary.lengthStats.p95Words.toFixed(0)} w
                </span>
              </div>
              {lengthPlot && <PlotlyChart data={lengthPlot.data} layout={lengthPlot.layout} />}
              <p className="text-xs leading-relaxed text-[#526783]">
                <strong className="text-[#203653]">What this shows:</strong> How long posts are in
                words. Short posts may lack context, while very long posts (over ~200 words) approach
                the Transformer&apos;s 256-token sequence limit and get truncated.
              </p>
            </section>
          </div>

          {/* Dataset Split Chart + Class Imbalance & Quality Audit */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-6 neu-card p-7 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Training, Validation & Testing Split
                </h2>
                <span className="text-xs font-mono text-[#4D6FAF]">
                  {datasetSummary.splitBreakdown ? 'Observed Split' : '70 / 15 / 15 Stratified Plan'}
                </span>
              </div>
              {splitPlot && <PlotlyChart data={splitPlot.data} layout={splitPlot.layout} />}
              <p className="text-xs leading-relaxed text-[#526783]">
                <strong className="text-[#203653]">What this shows:</strong> Records are separated
                after deduplication into a training set (to fit weights), validation set (for early
                stopping), and held-out test set (for unbiased evaluation) with zero text overlap.
              </p>
            </section>

            <section className="lg:col-span-6 neu-card p-7 space-y-5">
              <div>
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Class Imbalance Analysis
                </h2>
                <div className="mt-3 neu-inset p-4 space-y-2 text-xs">
                  <div className="flex justify-between gap-2">
                    <span className="text-[#526783]">Severity Classification</span>
                    <span className="font-semibold text-[#203653]">{imbalanceSeverity}</span>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="text-[#526783]">Majority vs. Minority Class</span>
                    <span className="font-mono text-[#203653]">
                      {datasetSummary.majorityClass} vs. {datasetSummary.minorityClass} (
                      {datasetSummary.imbalanceRatio.toFixed(2)}x)
                    </span>
                  </div>
                  <p className="pt-1 leading-relaxed text-[#526783]">
                    When one category appears much more often than another, standard accuracy can
                    look high even if minority categories are missed. MindTrace AI computes
                    inverse-frequency class weights on the training split and tracks Macro F1 to
                    protect minority-class sensitivity.
                  </p>
                </div>
              </div>

              <div>
                <h3 className="font-display text-sm font-semibold text-[#203653] mb-2">
                  Missing Values & Data Quality Statistics
                </h3>
                <table className="w-full text-left border-collapse text-xs">
                  <tbody className="divide-y divide-[#AFC5F5]/45">
                    {[
                      ['Total Corpus Records', datasetSummary.totalRows],
                      ['Usable Records (valid text & label)', datasetSummary.validRows],
                      ['Missing / Empty Text Records', datasetSummary.missingTextRows],
                      ['Missing Label Records', datasetSummary.missingLabelRows],
                      ['Exact Duplicate Records', datasetSummary.duplicateTexts],
                      ['Short Posts (< 4 words)', datasetSummary.shortTextCount],
                      ['Long Posts (> 200 words)', datasetSummary.longTextCount],
                      ['Mean Characters / Post', datasetSummary.lengthStats.meanChars.toFixed(1)],
                    ].map(([k, v]) => (
                      <tr key={String(k)}>
                        <td className="py-2 text-[#526783]">{k}</td>
                        <td className="py-2 text-right font-mono tabular-nums font-medium text-[#203653]">
                          {v}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>

          {/* Per-class Top Terms */}
          <section className="neu-card p-7 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="font-display text-base font-semibold text-[#203653]">
                  Frequent Vocabulary by Class
                </h2>
                <p className="text-xs text-[#526783]">
                  Most frequent non-stopword terms observed in each category of the active dataset.
                </p>
              </div>
              <div className="neu-inset flex flex-wrap gap-1.5 p-1.5">
                {datasetSummary.classCounts.map((c) => (
                  <button
                    key={c.label}
                    type="button"
                    onClick={() => setSelectedClass(c.label)}
                    className={`rounded-xl px-3 py-1.5 text-xs font-medium transition-all ${
                      selectedClass === c.label
                        ? 'neu-btn-accent text-[#203653]'
                        : 'text-[#526783] hover:text-[#203653]'
                    }`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-5">
              {(datasetSummary.topTermsByClass[selectedClass] ?? []).map((t) => (
                <div
                  key={t.term}
                  className="neu-inset flex items-center justify-between px-3.5 py-2.5 text-xs"
                >
                  <span className="font-mono text-[#203653]">{t.term}</span>
                  <span className="font-mono tabular-nums text-[#526783]">{t.frequency}</span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {/* Optional Advanced Options: Custom CSV Override */}
      <details className="group neu-card">
        <summary className="flex cursor-pointer list-none items-center justify-between px-7 py-5 text-xs font-semibold text-[#203653] select-none">
          <span>Advanced Options (Inspect a Custom Labeled Dataset CSV)</span>
          <ChevronDown className="h-4 w-4 text-[#4D6FAF] transition-transform group-open:rotate-180" />
        </summary>

        <div className="border-t border-[#AFC5F5]/55 p-7 space-y-4 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-[#526783] max-w-2xl">
              By default, Dataset Analytics loads the configured research corpus automatically. If you
              want to audit or train on your own CSV file (<code className="font-mono">text</code>,{' '}
              <code className="font-mono">label</code>, optional <code className="font-mono">split</code>),
              load it here.
            </p>
            <label className="neu-btn inline-flex cursor-pointer items-center gap-2 px-4 py-2 font-medium whitespace-nowrap">
              <Upload className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
              <span>Select Custom Dataset CSV</span>
              <input type="file" accept=".csv,text/csv" onChange={handleFileUpload} className="sr-only" />
            </label>
          </div>

          {parsedCsv && (
            <div className="pt-4 border-t border-[#AFC5F5]/55 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              <div>
                <label htmlFor="dataset-name-input" className="block text-[#526783] mb-1">
                  Dataset Name
                </label>
                <input
                  id="dataset-name-input"
                  type="text"
                  value={datasetNameInput}
                  onChange={(e) => handleMetadataChange(e.target.value, datasetSourceInput)}
                  className="neu-input w-full px-3 py-1.5 text-xs text-[#203653]"
                />
              </div>
              <div>
                <label htmlFor="dataset-source-input" className="block text-[#526783] mb-1">
                  Dataset Source / Citation
                </label>
                <input
                  id="dataset-source-input"
                  type="text"
                  value={datasetSourceInput}
                  onChange={(e) => handleMetadataChange(datasetNameInput, e.target.value)}
                  className="neu-input w-full px-3 py-1.5 text-xs text-[#203653]"
                />
              </div>
              <div>
                <label htmlFor="corpus-text-col" className="block text-[#526783] mb-1">
                  Text Column
                </label>
                <select
                  id="corpus-text-col"
                  value={textCol}
                  onChange={(e) => handleRecompute(e.target.value, labelCol, splitCol)}
                  className="neu-input w-full px-3 py-1.5 text-xs text-[#203653]"
                >
                  {parsedCsv.data.headers.map((h) => (
                    <option key={h} value={h}>
                      {h}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="corpus-label-col" className="block text-[#526783] mb-1">
                  Label Column
                </label>
                <select
                  id="corpus-label-col"
                  value={labelCol}
                  onChange={(e) => handleRecompute(textCol, e.target.value, splitCol)}
                  className="neu-input w-full px-3 py-1.5 text-xs text-[#203653]"
                >
                  {parsedCsv.data.headers.map((h) => (
                    <option key={h} value={h}>
                      {h}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="corpus-split-col" className="block text-[#526783] mb-1">
                  Split Column (Optional)
                </label>
                <select
                  id="corpus-split-col"
                  value={splitCol}
                  onChange={(e) => handleRecompute(textCol, labelCol, e.target.value)}
                  className="neu-input w-full px-3 py-1.5 text-xs text-[#203653]"
                >
                  <option value="">None (70/15/15 stratified plan)</option>
                  {parsedCsv.data.headers.map((h) => (
                    <option key={h} value={h}>
                      {h}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}
