import { ArrowRight, ShieldCheck } from 'lucide-react';
import {
  PRIVACY_NOTICE_TEXT,
  SimpleSupportiveDisclaimer,
  SupportiveGuidancePanel,
} from '@/src/components/EthicsBanner';
import { WorkflowFlowchart } from '@/src/components/WorkflowFlowchart';
import type { PageId } from '@/src/types/mindtrace';

interface AboutProjectPageProps {
  onNavigate: (page: PageId) => void;
}

export function AboutProjectPage({ onNavigate }: AboutProjectPageProps) {
  return (
    <div className="mx-auto max-w-4xl space-y-6 py-2">
      <section className="neu-card-elevated p-6 sm:p-8 space-y-4">
        <h2 className="font-display text-2xl font-bold text-[#203653]">What MindTrace AI Does</h2>
        <p className="text-sm sm:text-base leading-relaxed text-[#526783]">
          MindTrace AI is an undergraduate computer science research project designed to show how
          modern language AI reads written text for signs of emotional strain—and, just as
          importantly, how explainable AI can make those readings transparent and easy to understand.
        </p>
        <p className="text-sm leading-relaxed text-[#526783]">
          Instead of giving an unexplained label, MindTrace AI highlights the words and phrases that
          influenced its reading most, keeps language calm and non-clinical, and offers immediate
          supportive resources whenever someone may be in distress.
        </p>
      </section>

      <section className="neu-card p-6 sm:p-8 space-y-4">
        <div className="space-y-1">
          <h2 className="font-display text-lg font-semibold text-[#203653]">
            How It Works (Step by Step)
          </h2>
          <p className="text-xs text-[#526783]">
            A simple overview of what happens when you share text in MindTrace AI.
          </p>
        </div>
        <WorkflowFlowchart />
      </section>

      <section className="neu-card p-6 sm:p-8 space-y-3">
        <div className="flex items-center gap-2.5">
          <ShieldCheck className="h-5 w-5 text-[#4D6FAF]" aria-hidden="true" />
          <h2 className="font-display text-lg font-semibold text-[#203653]">
            Privacy &amp; Safe Boundaries
          </h2>
        </div>
        <p className="text-sm leading-relaxed text-[#526783]">{PRIVACY_NOTICE_TEXT}</p>
        <SimpleSupportiveDisclaimer />
      </section>

      <section className="neu-card p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="space-y-1">
          <h2 className="font-display text-base font-semibold text-[#203653]">
            Reviewing this as an academic project?
          </h2>
          <p className="text-xs leading-relaxed text-[#526783]">
            Explore full Transformer details, Explainable AI visualizations, dataset analytics,
            and evaluation metrics in Project Details.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onNavigate('project-details')}
          className="neu-btn-primary inline-flex items-center gap-2 px-5 py-3 text-xs shrink-0 min-h-[44px]"
        >
          <span>Open Project Details</span>
          <ArrowRight className="h-3.5 w-3.5 text-white" aria-hidden="true" />
        </button>
      </section>

      <SupportiveGuidancePanel />
    </div>
  );
}
