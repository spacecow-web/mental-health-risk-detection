import { AlertCircle, HeartHandshake } from 'lucide-react';

export function EthicsDisclaimerBanner() {
  return (
    <div
      role="note"
      aria-label="Research purpose disclaimer"
      className="rounded-xl border border-[#E2E8F0] bg-[#F1F5F9] px-4 py-3 text-xs leading-relaxed text-[#334155] flex items-start gap-3"
    >
      <AlertCircle className="h-4 w-4 text-[#0F2942] shrink-0 mt-0.5" aria-hidden="true" />
      <div>
        <span className="font-semibold text-[#0F172A]">Research Prototype Disclaimer: </span>
        MindTrace AI is an educational and academic NLP prototype, not a clinical diagnostic system.
        Outputs reflect statistical linguistic patterns in text and do not prove a person&apos;s mental
        health condition. Submitted text stays strictly in ephemeral browser memory.
      </div>
    </div>
  );
}

export function SupportiveGuidancePanel() {
  return (
    <section
      aria-label="Supportive mental health resources"
      className="rounded-xl border border-[#99F6E4] bg-[#F0FDFA] p-5 text-sm text-[#0F172A]"
    >
      <div className="flex items-start gap-3">
        <HeartHandshake className="h-5 w-5 text-[#0D7377] shrink-0 mt-0.5" aria-hidden="true" />
        <div className="space-y-3">
          <div>
            <h3 className="font-display text-sm font-semibold text-[#0F2942]">
              Supportive Guidance & Confidential Resources
            </h3>
            <p className="mt-1 text-xs leading-relaxed text-[#334155]">
              Distress-associated language or elevated risk terminology was observed in the current
              text or prediction. This tool cannot provide clinical support. If you or someone you
              know is experiencing distress, free confidential help is available:
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 text-xs">
            <div className="rounded-lg border border-[#CCFBF1] bg-white p-3">
              <div className="font-semibold text-[#0F172A]">US & Canada — 988 Lifeline</div>
              <div className="mt-0.5 text-[#475569]">Call or text 988 (24/7, free, confidential)</div>
            </div>
            <div className="rounded-lg border border-[#CCFBF1] bg-white p-3">
              <div className="font-semibold text-[#0F172A]">Crisis Text Line</div>
              <div className="mt-0.5 text-[#475569]">Text HOME to 741741 (US/CA) or 85258 (UK)</div>
            </div>
            <div className="rounded-lg border border-[#CCFBF1] bg-white p-3">
              <div className="font-semibold text-[#0F172A]">United Kingdom — NHS / Samaritans</div>
              <div className="mt-0.5 text-[#475569]">Call 111 (NHS) or 116 123 (Samaritans)</div>
            </div>
            <div className="rounded-lg border border-[#CCFBF1] bg-white p-3">
              <div className="font-semibold text-[#0F172A]">International Directory</div>
              <div className="mt-0.5 text-[#475569]">findahelpline.com (local crisis lines globally)</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
