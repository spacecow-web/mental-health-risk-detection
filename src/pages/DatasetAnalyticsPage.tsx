import { useEffect, useState, type ChangeEvent } from 'react';
import { Upload } from 'lucide-react';
import { PageHeader } from '@/src/components/PageHeader';
import type { DatasetSummary } from '@/src/types/mindtrace';
import { parseCsvContent, type ParsedCsv } from '@/src/utils/csvParser';
import { computeDatasetAnalytics } from '@/src/utils/datasetAnalytics';

interface DatasetAnalyticsPageProps {
  datasetSummary: DatasetSummary | null;
  onSetDatasetSummary: (summary: DatasetSummary | null) => void;
}

export function DatasetAnalyticsPage({
  datasetSummary,
  onSetDatasetSummary,
}: DatasetAnalyticsPageProps) {
  const [parsedCsv, setParsedCsv] = useState<{ fileName: string; data: ParsedCsv } | null>(null);
  const [textCol, setTextCol] = useState('text');
  const [labelCol, setLabelCol] = useState('label');
  const [splitCol, setSplitCol] = useState('');
  const [selectedClass, setSelectedClass] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (datasetSummary && datasetSummary.classCounts.length > 0) {
      if (!selectedClass || !datasetSummary.topTermsByClass[selectedClass]) {
        setSelectedClass(datasetSummary.classCounts[0].label);
      }
    }
  }, [datasetSummary, selectedClass]);

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

      setParsedCsv({ fileName: file.name, data: parsed });
      setTextCol(detectedText);
      setLabelCol(detectedLabel);
      setSplitCol(detectedSplit);

      const summary = computeDatasetAnalytics(
        parsed.rows,
        detectedText,
        detectedLabel,
        file.name,
        detectedSplit || undefined
      );
      onSetDatasetSummary(summary);
      setSelectedClass(summary.classCounts[0]?.label ?? '');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unable to parse dataset CSV.');
    } finally {
      e.target.value = '';
    }
  }

  function handleRecompute(nextTextCol: string, nextLabelCol: string, nextSplitCol: string) {
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
        nextSplitCol || undefined
      );
      onSetDatasetSummary(summary);
      setSelectedClass(summary.classCounts[0]?.label ?? '');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Invalid column mapping.');
    }
  }

  const maxClassCount = datasetSummary
    ? Math.max(...datasetSummary.classCounts.map((c) => c.count), 1)
    : 1;
  const maxBinCount = datasetSummary
    ? Math.max(...datasetSummary.histogramBins.map((b) => b.count), 1)
    : 1;

  const projectedTrain = datasetSummary ? Math.round(datasetSummary.validRows * 0.7) : 0;
  const projectedVal = datasetSummary ? Math.round(datasetSummary.validRows * 0.15) : 0;
  const projectedTest = datasetSummary
    ? Math.max(0, datasetSummary.validRows - projectedTrain - projectedVal)
    : 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dataset Analytics"
        description="Inspect corpus size, class balance, text length distributions, train/validation/test split ratios, and data quality diagnostics."
        action={
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-[#0F2942] px-4 py-2.5 text-xs font-medium text-white hover:bg-[#173A5E] transition-colors whitespace-nowrap min-h-[40px]">
            <Upload className="h-3.5 w-3.5" aria-hidden="true" />
            <span>{datasetSummary ? 'Replace Dataset CSV' : 'Upload Dataset CSV'}</span>
            <input type="file" accept=".csv,text/csv" onChange={handleFileUpload} className="sr-only" />
          </label>
        }
      />

      {parsedCsv && (
        <section className="rounded-xl border border-[#E2E8F0] bg-white p-5 grid gap-4 sm:grid-cols-4 text-xs">
          <div>
            <span className="block text-[#64748B] mb-1">Active Corpus</span>
            <span className="font-mono font-medium text-[#0F172A] break-all">
              {parsedCsv.fileName}
            </span>
          </div>
          <div>
            <label htmlFor="corpus-text-col" className="block text-[#64748B] mb-1">
              Text Column
            </label>
            <select
              id="corpus-text-col"
              value={textCol}
              onChange={(e) => handleRecompute(e.target.value, labelCol, splitCol)}
              className="w-full rounded-lg border border-[#CBD5E1] bg-white px-3 py-1.5 text-xs text-[#0F172A]"
            >
              {parsedCsv.data.headers.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="corpus-label-col" className="block text-[#64748B] mb-1">
              Label Column
            </label>
            <select
              id="corpus-label-col"
              value={labelCol}
              onChange={(e) => handleRecompute(textCol, e.target.value, splitCol)}
              className="w-full rounded-lg border border-[#CBD5E1] bg-white px-3 py-1.5 text-xs text-[#0F172A]"
            >
              {parsedCsv.data.headers.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="corpus-split-col" className="block text-[#64748B] mb-1">
              Split Column (Optional)
            </label>
            <select
              id="corpus-split-col"
              value={splitCol}
              onChange={(e) => handleRecompute(textCol, labelCol, e.target.value)}
              className="w-full rounded-lg border border-[#CBD5E1] bg-white px-3 py-1.5 text-xs text-[#0F172A]"
            >
              <option value="">None (use 70/15/15 stratified plan)</option>
              {parsedCsv.data.headers.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </div>
        </section>
      )}

      {errorMsg && (
        <div className="rounded-lg bg-[#FEF2F2] p-3 text-xs text-[#991B1B]">{errorMsg}</div>
      )}

      {!datasetSummary ? (
        <section className="rounded-xl border border-[#E2E8F0] bg-white p-8 max-w-3xl">
          <div className="text-xs font-mono text-[#0D7377]">
            Empty State · No Corpus Supplied
          </div>
          <h2 className="mt-2 font-display text-lg font-semibold text-[#0F172A]">
            Upload a Labeled Social Media Dataset CSV
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-[#475569]">
            MindTrace AI never fabricates dataset sizes, class distributions, or quality reports.
            Upload your labeled CSV file (<code className="font-mono text-xs">text</code>,{' '}
            <code className="font-mono text-xs">label</code>, optional{' '}
            <code className="font-mono text-xs">split</code>) to compute real exploratory analytics.
          </p>
          <pre className="mt-4 overflow-x-auto rounded-lg bg-[#0F172A] p-3.5 text-xs font-mono text-[#E2E8F0]">
{`text,label,split
"I have not slept properly all week and feel overwhelmed by deadlines.",Stress,train
"Finished my morning coffee and looking forward to the weekend hike.",Neutral,test`}
          </pre>
        </section>
      ) : (
        <>
          {/* Dataset Overview KPIs */}
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: 'Valid Posts', value: String(datasetSummary.validRows) },
              { label: 'Distinct Classes', value: String(datasetSummary.numClasses) },
              {
                label: 'Median / P95 Words',
                value: `${datasetSummary.lengthStats.medianWords.toFixed(1)} / ${datasetSummary.lengthStats.p95Words.toFixed(0)}`,
              },
              {
                label: 'Duplicate Texts',
                value: String(datasetSummary.duplicateTexts),
              },
            ].map((stat) => (
              <div key={stat.label} className="rounded-xl border border-[#E2E8F0] bg-white p-5">
                <div className="text-xs text-[#64748B]">{stat.label}</div>
                <div className="mt-1.5 font-mono text-xl font-semibold tabular-nums text-[#0F2942]">
                  {stat.value}
                </div>
              </div>
            ))}
          </section>

          {/* Class Distribution & Text Length Distribution */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-semibold text-[#0F172A]">
                  Class Distribution
                </h2>
                <span className="font-mono text-xs tabular-nums text-[#64748B]">
                  N={datasetSummary.validRows}
                </span>
              </div>
              <div className="space-y-3">
                {datasetSummary.classCounts.map((item) => (
                  <div key={item.label} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-[#0F172A]">{item.label}</span>
                      <span className="font-mono tabular-nums text-[#475569]">
                        {item.count} ({item.sharePercent.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="h-2.5 w-full rounded-full bg-[#F1F5F9] overflow-hidden">
                      <div
                        className="h-full rounded-full bg-[#0F2942]"
                        style={{ width: `${(item.count / maxClassCount) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-semibold text-[#0F172A]">
                  Text Length Distribution
                </h2>
                <span className="text-xs font-mono tabular-nums text-[#64748B]">
                  Mean: {datasetSummary.lengthStats.meanWords.toFixed(1)} words
                </span>
              </div>
              <div className="space-y-2.5">
                {datasetSummary.histogramBins.map((bin) => (
                  <div key={bin.rangeLabel} className="grid grid-cols-12 items-center gap-2 text-xs">
                    <div className="col-span-3 font-mono tabular-nums text-[#475569]">
                      {bin.rangeLabel} w
                    </div>
                    <div className="col-span-7 h-4 rounded bg-[#F1F5F9] overflow-hidden">
                      <div
                        className="h-full rounded bg-[#0D7377]"
                        style={{ width: `${(bin.count / maxBinCount) * 100}%` }}
                      />
                    </div>
                    <div className="col-span-2 text-right font-mono tabular-nums text-[#0F172A]">
                      {bin.count}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* Dataset Splitting Information + Data Quality Statistics */}
          <div className="grid gap-6 lg:grid-cols-12">
            <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="font-display text-base font-semibold text-[#0F172A]">
                  Dataset Splitting Information
                </h2>
                <span className="text-xs font-mono text-[#0D7377]">
                  {datasetSummary.splitBreakdown ? 'Observed Column' : '70 / 15 / 15 Stratified Plan'}
                </span>
              </div>

              {datasetSummary.splitBreakdown ? (
                <div className="space-y-3">
                  {datasetSummary.splitBreakdown.map((item) => (
                    <div key={item.split} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="font-medium capitalize text-[#0F172A]">{item.split}</span>
                        <span className="font-mono tabular-nums text-[#475569]">
                          {item.count} ({item.sharePercent.toFixed(1)}%)
                        </span>
                      </div>
                      <div className="h-2.5 w-full rounded-full bg-[#F1F5F9] overflow-hidden">
                        <div
                          className="h-full rounded-full bg-[#2563EB]"
                          style={{ width: `${item.sharePercent}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-3">
                  <p className="text-xs leading-relaxed text-[#475569]">
                    No explicit <code className="font-mono">split</code> column was detected in this
                    CSV. When passed to <code className="font-mono">src/train.py</code>, stratified
                    splitting allocates samples as follows:
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { name: 'Train (70%)', count: projectedTrain },
                      { name: 'Validation (15%)', count: projectedVal },
                      { name: 'Held-Out Test (15%)', count: projectedTest },
                    ].map((s) => (
                      <div key={s.name} className="rounded-lg bg-[#F8FAFC] p-3">
                        <div className="text-[11px] text-[#64748B]">{s.name}</div>
                        <div className="mt-0.5 font-mono text-base font-semibold tabular-nums text-[#0F2942]">
                          {s.count}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6">
              <h2 className="font-display text-base font-semibold text-[#0F172A] mb-3">
                Data Quality Statistics
              </h2>
              <table className="w-full text-left border-collapse text-xs">
                <tbody className="divide-y divide-[#E2E8F0]">
                  {[
                    ['Total CSV Rows', datasetSummary.totalRows],
                    ['Valid Non-Empty Rows', datasetSummary.validRows],
                    ['Missing / Empty Text Rows', datasetSummary.missingTextRows],
                    ['Missing Label Rows', datasetSummary.missingLabelRows],
                    ['Exact Duplicate Posts', datasetSummary.duplicateTexts],
                    ['Short Posts (< 4 words)', datasetSummary.shortTextCount],
                    ['Long Posts (> 200 words, near 256-token cap)', datasetSummary.longTextCount],
                    ['Mean Characters / Post', datasetSummary.lengthStats.meanChars.toFixed(1)],
                  ].map(([k, v]) => (
                    <tr key={String(k)}>
                      <td className="py-2 text-[#475569]">{k}</td>
                      <td className="py-2 text-right font-mono tabular-nums font-medium text-[#0F172A]">
                        {v}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          </div>

          {/* Per-class Top Terms */}
          <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="font-display text-base font-semibold text-[#0F172A]">
                Frequent Vocabulary by Class
              </h2>
              <div className="flex flex-wrap gap-1 rounded-lg bg-[#F1F5F9] p-1">
                {datasetSummary.classCounts.map((c) => (
                  <button
                    key={c.label}
                    type="button"
                    onClick={() => setSelectedClass(c.label)}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                      selectedClass === c.label
                        ? 'bg-white text-[#0F2942] shadow-xs'
                        : 'text-[#64748B] hover:text-[#0F172A]'
                    }`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
              {(datasetSummary.topTermsByClass[selectedClass] ?? []).map((t) => (
                <div
                  key={t.term}
                  className="flex items-center justify-between rounded-lg bg-[#F8FAFC] px-3 py-2 text-xs"
                >
                  <span className="font-mono text-[#0F172A]">{t.term}</span>
                  <span className="font-mono tabular-nums text-[#64748B]">{t.frequency}</span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
