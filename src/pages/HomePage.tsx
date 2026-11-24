import { ArrowRight, CheckCircle2 } from 'lucide-react';
import { SimpleSupportiveDisclaimer } from '@/src/components/EthicsBanner';
import type { PageId } from '@/src/types/mindtrace';

interface HomePageProps {
  onNavigate: (page: PageId) => void;
}

const KEY_POINTS = [
  {
    title: 'Simple text-based analysis',
    detail: 'Write a few sentences at your own pace—no forms, uploads, or setup needed.',
  },
  {
    title: 'Easy-to-understand explanations',
    detail: 'See which words stood out most in plain everyday language.',
  },
  {
    title: 'Designed for awareness, not diagnosis',
    detail: 'Built as a calm educational tool, never a medical or clinical label.',
  },
];

export function HomePage({ onNavigate }: HomePageProps) {
  return (
    <div className="mx-auto max-w-3xl space-y-8 py-4 sm:py-8">
      <section className="neu-card-elevated p-8 sm:p-11 space-y-6">
        <div className="space-y-4">
          <h2 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-[#203653] leading-snug">
            Take a moment to understand what&apos;s in your words.
          </h2>
          <p className="text-base leading-relaxed text-[#526783] max-w-2xl">
            MindTrace AI looks at patterns in written text that may be associated with emotional
            distress. It uses AI to help explain these patterns in a simple way.
          </p>
        </div>

        <div className="pt-1">
          <button
            type="button"
            onClick={() => onNavigate('check-text')}
            className="neu-btn-primary inline-flex items-center gap-2.5 px-7 py-3.5 text-sm min-h-[48px]"
          >
            <span>Analyse Your Text</span>
            <ArrowRight className="h-4 w-4 text-white" aria-hidden="true" />
          </button>
        </div>
      </section>

      <section aria-label="What MindTrace AI offers" className="grid gap-4 sm:grid-cols-3">
        {KEY_POINTS.map((point) => (
          <div key={point.title} className="neu-card p-6 space-y-2">
            <CheckCircle2 className="h-5 w-5 text-[#4D6FAF]" aria-hidden="true" />
            <h3 className="font-display text-sm font-semibold text-[#203653]">{point.title}</h3>
            <p className="text-xs leading-relaxed text-[#526783]">{point.detail}</p>
          </div>
        ))}
      </section>

      <SimpleSupportiveDisclaimer />
    </div>
  );
}
