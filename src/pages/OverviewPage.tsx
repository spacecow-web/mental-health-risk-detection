import { ArrowRight } from 'lucide-react';
import { PageHeader } from '@/src/components/PageHeader';
import type {
  DatasetSummary,
  EndpointConfig,
  InferenceExplanationBundle,
  NamedEvaluationRun,
  PageId,
} from '@/src/types/mindtrace';

interface OverviewPageProps {
  onNavigate: (page: PageId) => void;
  endpointConfig: EndpointConfig;
  activeEvalRun: NamedEvaluationRun | null;
  datasetSummary: DatasetSummary | null;
  activeBundle: InferenceExplanationBundle | null;
}

const WORKFLOW_STEPS = [
  {
    step: '01',
    title: 'Validation & Anonymization',
    detail: 'Validates text length and masks URLs ([URL]) and handles ([USER]) before encoding.',
  },
  {
    step: '02',
    title: 'WordPiece Tokenization',
    detail: 'Segments cleaned text into subword units bounded at 256 tokens with attention masks.',
  },
  {
    step: '03',
    title: 'Transformer Inference',
    detail: 'Computes calibrated class probabilities and normalized Shannon entropy uncertainty.',
  },
  {
    step: '04',
    title: 'SHAP & LIME Attribution',
    detail: 'Measures positive and negative token sensitivity for transparent model explanation.',
  },
  {
    step: '05',
    title: 'Held-Out Evaluation',
    detail: 'Audits Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC on real splits.',
  },
];

export function OverviewPage({
  onNavigate,
  endpointConfig,
  activeEvalRun,
  datasetSummary,
  activeBundle,
}: OverviewPageProps) {
  const maxClassCount = datasetSummary
    ? Math.max(...datasetSummary.classCounts.map((c) => c.count), 1)
    : 1;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Overview"
        description="Project summary, Transformer model architecture, dataset readiness, analysis workflow, and verified evaluation status."
        action={
          <button
            type="button"
            onClick={() => onNavigate('text-analysis')}
            className="inline-flex items-center gap-2 rounded-lg bg-[#0F2942] px-4 py-2.5 text-xs font-medium text-white hover:bg-[#173A5E] transition-colors whitespace-nowrap min-h-[40px]"
          >
            <span>Open Text Analysis</span>
            <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        }
      />

      {/* Project Title & Description */}
      <section className="rounded-xl border border-[#E2E8F0] bg-white p-6">
        <div className="text-xs font-medium text-[#0D7377]">
          Undergraduate & Academic NLP Research Prototype · Non-Clinical
        </div>
        <h2 className="mt-1.5 font-display text-xl font-bold text-[#0F2942]">
          Explainable Mental Health Risk Detection from Social Media Text Using Transformer Models
        </h2>
        <p className="mt-2.5 text-sm leading-relaxed text-[#475569] max-w-4xl">
          MindTrace AI investigates how encoder-based Transformer models identify linguistic cues
          associated with psychological distress in social media posts while surfacing token-level
          explanations via SHAP and LIME. The system separates statistical classifier behaviour from
          clinical diagnosis and reports metrics exclusively from user-supplied datasets and held-out
          test splits.
        </p>
      </section>

      {/* Model Architecture + Dataset Summary */}
      <div className="grid gap-6 lg:grid-cols-12">
        <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Model Architecture Information
            </h2>
            <span className="font-mono text-xs text-[#64748B]">
              {endpointConfig.status === 'connected' ? 'Bridge Connected' : 'Setup Mode'}
            </span>
          </div>

          <table className="w-full text-left border-collapse text-xs">
            <tbody className="divide-y divide-[#E2E8F0]">
              <tr>
                <td className="py-2.5 text-[#64748B]">Primary Backbone</td>
                <td className="py-2.5 text-right font-mono font-medium text-[#0F172A]">
                  distilbert-base-uncased (66M params, 6 layers)
                </td>
              </tr>
              <tr>
                <td className="py-2.5 text-[#64748B]">Secondary Backbone</td>
                <td className="py-2.5 text-right font-mono font-medium text-[#0F172A]">
                  bert-base-uncased (110M params, 12 layers)
                </td>
              </tr>
              <tr>
                <td className="py-2.5 text-[#64748B]">Tokenization & Context</td>
                <td className="py-2.5 text-right font-mono tabular-nums text-[#0F172A]">
                  WordPiece · Max 256 tokens
                </td>
              </tr>
              <tr>
                <td className="py-2.5 text-[#64748B]">Explainability Methods</td>
                <td className="py-2.5 text-right font-mono text-[#0F172A]">
                  Partition SHAP & LIME Surrogate
                </td>
              </tr>
              <tr>
                <td className="py-2.5 text-[#64748B]">Uncertainty Metric</td>
                <td className="py-2.5 text-right font-mono text-[#0F172A]">
                  Normalized Shannon Entropy [0, 1]
                </td>
              </tr>
              <tr>
                <td className="py-2.5 text-[#64748B]">Active Session Sample</td>
                <td className="py-2.5 text-right font-mono text-[#0F172A]">
                  {activeBundle ? `${activeBundle.predictedLabel} (${activeBundle.method})` : 'None'}
                </td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 flex flex-col justify-between space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Dataset Summary
            </h2>
            <button
              type="button"
              onClick={() => onNavigate('dataset-analytics')}
              className="text-xs font-medium text-[#2563EB] hover:underline"
            >
              {datasetSummary ? 'Inspect Corpus' : 'Upload CSV'}
            </button>
          </div>

          {!datasetSummary ? (
            <div className="rounded-lg bg-[#F8FAFC] p-5 text-xs leading-relaxed text-[#475569]">
              <div className="font-semibold text-[#0F172A]">No Dataset Loaded</div>
              <p className="mt-1">
                Dataset size, class counts, and post length statistics are never simulated. Upload
                a labeled CSV corpus on the Dataset Analytics page to populate this summary.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3 bg-[#F8FAFC] rounded-lg p-3.5">
                <div>
                  <div className="text-[11px] text-[#64748B]">Valid Posts</div>
                  <div className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-[#0F2942]">
                    {datasetSummary.validRows}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-[#64748B]">Classes</div>
                  <div className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-[#0F2942]">
                    {datasetSummary.numClasses}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-[#64748B]">Median Words</div>
                  <div className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-[#0F2942]">
                    {datasetSummary.lengthStats.medianWords.toFixed(1)}
                  </div>
                </div>
              </div>
              <div className="text-xs text-[#64748B] flex justify-between">
                <span>Source file: {datasetSummary.fileName}</span>
                <span className="font-mono tabular-nums">
                  Duplicates: {datasetSummary.duplicateTexts}
                </span>
              </div>
            </div>
          )}
        </section>
      </div>

      {/* Analysis Workflow */}
      <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
        <div>
          <h2 className="font-display text-base font-semibold text-[#0F172A]">
            Analysis Workflow
          </h2>
          <p className="text-xs text-[#64748B]">
            End-to-end NLP processing sequence implemented across the modular Python and web pipeline.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {WORKFLOW_STEPS.map((item) => (
            <div key={item.step} className="rounded-lg bg-[#F8FAFC] p-4 space-y-1.5">
              <div className="font-mono text-xs font-semibold text-[#0D7377]">{item.step}</div>
              <div className="text-sm font-semibold text-[#0F172A]">{item.title}</div>
              <p className="text-xs leading-relaxed text-[#475569]">{item.detail}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Classification Distribution + Model Evaluation Summary */}
      <div className="grid gap-6 lg:grid-cols-12">
        <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Classification Distribution
            </h2>
            {datasetSummary && (
              <span className="font-mono text-xs tabular-nums text-[#64748B]">
                N={datasetSummary.validRows}
              </span>
            )}
          </div>

          {!datasetSummary ? (
            <div className="rounded-lg bg-[#F8FAFC] p-5 text-xs leading-relaxed text-[#475569]">
              <div className="font-semibold text-[#0F172A]">Awaiting Labeled Corpus</div>
              <p className="mt-1">
                Class proportions will render here once a labeled dataset CSV is uploaded in{' '}
                <button
                  type="button"
                  onClick={() => onNavigate('dataset-analytics')}
                  className="font-medium text-[#2563EB] hover:underline"
                >
                  Dataset Analytics
                </button>
                .
              </p>
            </div>
          ) : (
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
                      className="h-full rounded-full bg-[#0D7377]"
                      style={{ width: `${(item.count / maxClassCount) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="lg:col-span-6 rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Model Evaluation Summary
            </h2>
            <button
              type="button"
              onClick={() => onNavigate('model-performance')}
              className="text-xs font-medium text-[#2563EB] hover:underline"
            >
              {activeEvalRun ? 'Full Audit' : 'Upload Predictions'}
            </button>
          </div>

          {!activeEvalRun ? (
            <div className="rounded-lg bg-[#F8FAFC] p-5 text-xs leading-relaxed text-[#475569]">
              <div className="font-semibold text-[#0F172A]">No Evaluation Results Available</div>
              <p className="mt-1">
                Held-out metrics are populated strictly from real test-set predictions. Upload a
                held-out evaluation CSV on the{' '}
                <button
                  type="button"
                  onClick={() => onNavigate('model-performance')}
                  className="font-medium text-[#2563EB] hover:underline"
                >
                  Model Performance
                </button>{' '}
                page to view Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="text-xs text-[#64748B] flex justify-between">
                <span>Run: {activeEvalRun.modelName}</span>
                <span className="font-mono tabular-nums">
                  Held-out N={activeEvalRun.summary.sampleCount}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: 'Accuracy', val: `${(activeEvalRun.summary.accuracy * 100).toFixed(1)}%` },
                  { label: 'Precision', val: activeEvalRun.summary.macroPrecision.toFixed(3) },
                  { label: 'Recall', val: activeEvalRun.summary.macroRecall.toFixed(3) },
                  { label: 'Macro F1', val: activeEvalRun.summary.macroF1.toFixed(3) },
                ].map((m) => (
                  <div key={m.label} className="rounded-lg bg-[#F8FAFC] p-3">
                    <div className="text-[11px] text-[#64748B]">{m.label}</div>
                    <div className="mt-0.5 font-mono text-base font-semibold tabular-nums text-[#0F2942]">
                      {m.val}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
