import { useState } from 'react';
import { Check, Copy, Download } from 'lucide-react';
import { SupportiveGuidancePanel } from '@/src/components/EthicsBanner';
import { PageHeader } from '@/src/components/PageHeader';
import { PYTHON_PROJECT_FILES } from '@/src/utils/pythonKitRegistry';

interface AboutMethodologyPageProps {
  onDownloadPythonKit: () => void;
  downloadingKit: boolean;
}

export function AboutMethodologyPage({
  onDownloadPythonKit,
  downloadingKit,
}: AboutMethodologyPageProps) {
  const [selectedPath, setSelectedPath] = useState<string>(PYTHON_PROJECT_FILES[0].path);
  const [copied, setCopied] = useState(false);

  const selectedFile =
    PYTHON_PROJECT_FILES.find((f) => f.path === selectedPath) ?? PYTHON_PROJECT_FILES[0];

  async function handleCopyFile() {
    await navigator.clipboard.writeText(selectedFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="About & Methodology"
        description="Transformer modeling rationale, SHAP and LIME mathematical foundations, ethical safeguards, dataset limitations, and the complete Python/Streamlit source repository."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-3">
          <h2 className="font-display text-base font-semibold text-[#0F172A]">
            Transformer & Explainability Methodology
          </h2>
          <p className="text-sm leading-relaxed text-[#475569]">
            Social media mental health expression relies heavily on subtle contextual cues such as
            negation (&ldquo;not sleeping anymore&rdquo;), self-referential pronouns, and temporal
            markers. Bidirectional self-attention in <code className="font-mono text-xs">DistilBERT</code>{' '}
            (6 layers, 66M parameters) and <code className="font-mono text-xs">BERT-base</code>{' '}
            (12 layers, 110M parameters) models long-range dependencies that bag-of-words baselines miss.
          </p>
          <ul className="space-y-2 text-sm text-[#334155] pt-1">
            <li>
              <strong className="text-[#0F172A]">Partition SHAP:</strong> Estimates Shapley values
              over hierarchical text token coalitions using Hugging Face tokenizer maskers,
              producing additive supporting (+) and opposing (-) word attributions.
            </li>
            <li>
              <strong className="text-[#0F172A]">LIME Text Explainer:</strong> Samples local
              token-removal perturbations around an individual post and fits a sparse linear
              surrogate model to identify influential phrases.
            </li>
            <li>
              <strong className="text-[#0F172A]">Predictive Uncertainty:</strong> Normalized
              Shannon entropy flags predictions where probability mass is spread across categories.
            </li>
          </ul>
        </section>

        <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-3">
          <h2 className="font-display text-base font-semibold text-[#0F172A]">
            Ethical Framework, Bias & Limitations
          </h2>
          <ul className="space-y-2.5 text-sm leading-relaxed text-[#475569]">
            <li>
              <strong className="text-[#0F172A]">Non-Diagnostic Scope:</strong> Linguistic distress
              detection cannot establish clinical diagnoses or prove an individual&apos;s mental state.
            </li>
            <li>
              <strong className="text-[#0F172A]">Privacy-by-Default:</strong> User inputs remain
              ephemeral in session memory and are masked for URLs and @handles before tokenization.
            </li>
            <li>
              <strong className="text-[#0F172A]">Dataset & Platform Bias:</strong> Public corpora
              skew toward specific demographics, English dialects, and self-selected communities,
              creating domain-shift risk when applied elsewhere.
            </li>
            <li>
              <strong className="text-[#0F172A]">Attribution ≠ Causality:</strong> High SHAP or
              LIME importance explains model sensitivity, not psychological evidence.
            </li>
          </ul>
        </section>
      </div>

      {/* Modular Python & Streamlit Codebase Explorer */}
      <section className="rounded-xl border border-[#E2E8F0] bg-white p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-4 border-b border-[#E2E8F0]">
          <div>
            <h2 className="font-display text-base font-semibold text-[#0F172A]">
              Modular Python & Streamlit Codebase
            </h2>
            <p className="text-xs text-[#64748B]">
              Inspect every module of the local PyTorch, Hugging Face, SHAP/LIME, and Streamlit
              application, or download the complete ZIP archive.
            </p>
          </div>
          <button
            type="button"
            onClick={onDownloadPythonKit}
            disabled={downloadingKit}
            className="inline-flex items-center gap-2 rounded-lg bg-[#0F2942] px-4 py-2 text-xs font-medium text-white hover:bg-[#173A5E] disabled:opacity-60 whitespace-nowrap shrink-0 min-h-[40px]"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            <span>{downloadingKit ? 'Packaging ZIP...' : 'Download Full Python ZIP'}</span>
          </button>
        </div>

        <div className="grid gap-6 lg:grid-cols-12">
          <div className="lg:col-span-4 space-y-1 max-h-[480px] overflow-y-auto pr-1">
            {PYTHON_PROJECT_FILES.map((file) => {
              const active = file.path === selectedFile.path;
              return (
                <button
                  key={file.path}
                  type="button"
                  onClick={() => setSelectedPath(file.path)}
                  className={`w-full text-left rounded-lg px-3 py-2.5 transition-colors ${
                    active
                      ? 'bg-[#0F2942] text-white'
                      : 'bg-[#F8FAFC] text-[#0F172A] hover:bg-[#F1F5F9]'
                  }`}
                >
                  <div className="font-mono text-xs font-medium">{file.path}</div>
                  <div
                    className={`mt-0.5 text-[11px] leading-snug ${
                      active ? 'text-[#CBD5E1]' : 'text-[#64748B]'
                    }`}
                  >
                    {file.role}
                  </div>
                </button>
              );
            })}
          </div>

          <div className="lg:col-span-8 flex flex-col rounded-lg border border-[#E2E8F0] overflow-hidden">
            <div className="flex items-center justify-between bg-[#F8FAFC] px-4 py-2.5 border-b border-[#E2E8F0]">
              <span className="font-mono text-xs font-medium text-[#0F172A]">
                {selectedFile.path}
              </span>
              <button
                type="button"
                onClick={handleCopyFile}
                className="inline-flex items-center gap-1.5 rounded-md border border-[#CBD5E1] bg-white px-2.5 py-1 text-xs font-medium text-[#334155] hover:text-[#0F172A]"
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-[#0D7377]" aria-hidden="true" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="p-4 bg-[#0F172A] text-[#E2E8F0] font-mono text-xs leading-relaxed overflow-auto max-h-[430px]">
              <code>{selectedFile.content}</code>
            </pre>
          </div>
        </div>
      </section>

      <SupportiveGuidancePanel />
    </div>
  );
}
