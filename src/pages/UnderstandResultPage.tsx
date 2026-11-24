import { ArrowLeft, ArrowRight } from 'lucide-react';
import {
  SimpleSupportiveDisclaimer,
  SupportiveGuidancePanel,
} from '@/src/components/EthicsBanner';
import type { InferenceExplanationBundle, PageId } from '@/src/types/mindtrace';
import { detectImmediateSelfHarmCues, detectSupportiveCues } from '@/src/utils/nlpPipeline';
import {
  buildGentleHighlightSegments,
  createPlainLanguageInterpretation,
} from '@/src/utils/plainLanguage';

interface UnderstandResultPageProps {
  activeBundle: InferenceExplanationBundle | null;
  onNavigate: (page: PageId) => void;
}

export function UnderstandResultPage({
  activeBundle,
  onNavigate,
}: UnderstandResultPageProps) {
  if (!activeBundle) {
    return (
      <div className="mx-auto max-w-3xl space-y-6 py-4">
        <section className="neu-card-elevated p-8 space-y-4">
          <h2 className="font-display text-xl font-bold text-[#203653]">
            Nothing checked yet
          </h2>
          <p className="text-sm leading-relaxed text-[#526783]">
            Once you enter a message on the Check Your Text page, this screen will walk you through
            what the AI noticed and why—in simple everyday language.
          </p>
          <div className="pt-1">
            <button
              type="button"
              onClick={() => onNavigate('check-text')}
              className="neu-btn-primary inline-flex items-center gap-2 px-6 py-3 text-xs min-h-[44px]"
            >
              <ArrowLeft className="h-3.5 w-3.5 text-white" aria-hidden="true" />
              <span>Go to Check Your Text</span>
            </button>
          </div>
        </section>

        <SimpleSupportiveDisclaimer />
      </div>
    );
  }

  const submittedText = activeBundle.originalText || activeBundle.cleanedText || '';
  const analyzedText = activeBundle.cleanedText || submittedText;
  const plain = createPlainLanguageInterpretation(activeBundle);

  const showImmediateSafety = detectImmediateSelfHarmCues(submittedText);
  const showSupportive =
    showImmediateSafety || detectSupportiveCues(submittedText, activeBundle.predictedLabel);

  return (
    <div className="mx-auto max-w-3xl space-y-6 py-2">
      {/* A. What the analysis noticed */}
      <section className="neu-card-elevated p-6 sm:p-8 space-y-4">
        <div className="text-xs font-medium text-[#4D6FAF]">A · What the analysis noticed</div>
        <h2 className="font-display text-xl font-bold text-[#203653]">{plain.headline}</h2>
        <p className="text-sm leading-relaxed text-[#526783]">{plain.noticedSummary}</p>

        <div className="space-y-2 pt-2">
          <div className="text-xs font-semibold text-[#203653]">
            Your words with standout phrases softly highlighted
          </div>
          <div className="neu-inset p-4 text-sm leading-loose whitespace-pre-wrap break-words text-[#203653]">
            {buildGentleHighlightSegments(
              analyzedText,
              activeBundle.attributions,
              activeBundle.phraseAttributions
            ).map((seg, idx) =>
              seg.highlighted ? (
                <mark
                  key={`seg-${idx}`}
                  style={{
                    backgroundColor: `rgba(119, 153, 216, ${seg.alpha.toFixed(2)})`,
                    borderBottom: '2px solid #4D6FAF',
                  }}
                  className="rounded-md px-1.5 py-0.5 text-[#172840]"
                >
                  {seg.text}
                </mark>
              ) : (
                <span key={`seg-${idx}`}>{seg.text}</span>
              )
            )}
          </div>
        </div>
      </section>

      {/* B. Why this result appeared */}
      <section className="neu-card p-6 sm:p-8 space-y-4">
        <div className="text-xs font-medium text-[#4D6FAF]">B · Why this result appeared</div>
        <h2 className="font-display text-lg font-semibold text-[#203653]">
          Which parts of your text mattered most
        </h2>

        {plain.supportingWords.length > 0 && (
          <div className="neu-inset p-4 space-y-2.5">
            <div className="text-xs font-medium text-[#203653]">Words that stood out</div>
            <div className="flex flex-wrap gap-2">
              {plain.supportingWords.map((word) => (
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

        <ul className="space-y-2.5 text-sm leading-relaxed text-[#526783] list-disc pl-5">
          {plain.whyItAppeared.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </section>

      {/* C. What this result means */}
      <section className="neu-card p-6 sm:p-8 space-y-4">
        <div className="text-xs font-medium text-[#4D6FAF]">C · What this result means</div>
        <h2 className="font-display text-lg font-semibold text-[#203653]">
          Keeping this in perspective
        </h2>
        <ul className="space-y-2.5 text-sm leading-relaxed text-[#526783] list-disc pl-5">
          {plain.whatItMeans.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>

        <div className="pt-3 border-t border-[#AFC5F5]/55 flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => onNavigate('check-text')}
            className="neu-btn inline-flex items-center gap-2 px-5 py-2.5 text-xs font-medium min-h-[42px]"
          >
            <ArrowLeft className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
            <span>Check Another Message</span>
          </button>

          <button
            type="button"
            onClick={() => onNavigate('project-details')}
            className="text-xs font-medium text-[#4D6FAF] hover:text-[#203653] inline-flex items-center gap-1.5"
          >
            <span>Inspect technical details &amp; charts</span>
            <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>
      </section>

      {showSupportive && <SupportiveGuidancePanel immediateSafety={showImmediateSafety} />}

      <SimpleSupportiveDisclaimer />
    </div>
  );
}
