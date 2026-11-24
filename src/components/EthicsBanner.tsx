import { HeartHandshake, Info } from 'lucide-react';

export const SIMPLE_SUPPORTIVE_DISCLAIMER =
  'MindTrace AI is an educational research tool. It looks for patterns in text that may be associated with emotional distress. It cannot diagnose mental health conditions or replace professional support.';

export const EXACT_RESEARCH_DISCLAIMER =
  'This application is an educational research prototype. Its predictions indicate patterns in text learned from a dataset and are not medical diagnoses or validated clinical assessments. The results should not be used to label, monitor or make decisions about individuals.';

export const PRIVACY_NOTICE_TEXT =
  'Privacy Notice: Submitted text is processed strictly in ephemeral session memory and is not stored, logged, or saved to disk or external databases by default.';

export function SimpleSupportiveDisclaimer() {
  return (
    <div
      role="note"
      aria-label="Educational research notice"
      className="neu-inset px-5 py-4 text-xs leading-relaxed text-[#526783] flex items-start gap-3"
    >
      <Info className="h-4 w-4 text-[#4D6FAF] shrink-0 mt-0.5" aria-hidden="true" />
      <p>{SIMPLE_SUPPORTIVE_DISCLAIMER}</p>
    </div>
  );
}

interface SupportiveGuidancePanelProps {
  immediateSafety?: boolean;
}

export function SupportiveGuidancePanel({
  immediateSafety = false,
}: SupportiveGuidancePanelProps) {
  return (
    <section
      aria-label="Supportive guidance and confidential India helpline resources"
      className="neu-card-elevated p-6 sm:p-7 text-sm text-[#203653]"
      style={{
        background: immediateSafety
          ? 'linear-gradient(145deg, #F4ECEF 0%, #EADCE2 100%)'
          : 'linear-gradient(145deg, #EAF0FF 0%, #DCE6FC 100%)',
        border: immediateSafety ? '1px solid #F7EEF2' : '1px solid #F2F6FF',
      }}
    >
      <div className="flex items-start gap-3.5">
        <HeartHandshake
          className={`h-5 w-5 shrink-0 mt-0.5 ${
            immediateSafety ? 'text-[#9B5064]' : 'text-[#4D6FAF]'
          }`}
          aria-hidden="true"
        />
        <div className="space-y-3.5">
          <div>
            <h3 className="font-display text-base font-semibold text-[#203653]">
              {immediateSafety
                ? 'You do not have to carry this alone'
                : 'Support is always available'}
            </h3>
            {immediateSafety ? (
              <p className="mt-1.5 text-sm leading-relaxed text-[#4E3540]">
                If you are going through a painful moment or having thoughts of harming yourself,
                please consider reaching out right now to someone you trust—a friend, family member,
                or doctor—or contact emergency services in India on <strong>112</strong>. Free,
                gentle, and confidential support is also available 24/7 below.
              </p>
            ) : (
              <p className="mt-1.5 text-xs leading-relaxed text-[#526783]">
                If things feel heavy right now, talking with someone you trust or connecting with a
                free, confidential mental health helpline in India can make a real difference:
              </p>
            )}
          </div>
          <div className="grid gap-3 sm:grid-cols-2 text-xs">
            <div className="neu-inset p-3.5">
              <div className="font-semibold text-[#203653]">National Emergency Services (India)</div>
              <div className="mt-0.5 text-[#526783]">
                Call 112 (24/7 pan-India emergency support) or visit the nearest hospital
              </div>
            </div>
            <div className="neu-inset p-3.5">
              <div className="font-semibold text-[#203653]">Tele-MANAS (Govt. of India)</div>
              <div className="mt-0.5 text-[#526783]">
                Call 14416 or 1800-891-4416 (24/7, free, confidential, multilingual)
              </div>
            </div>
            <div className="neu-inset p-3.5">
              <div className="font-semibold text-[#203653]">KIRAN Mental Health Helpline</div>
              <div className="mt-0.5 text-[#526783]">
                Call 1800-599-0019 (24/7 toll-free Government of India helpline)
              </div>
            </div>
            <div className="neu-inset p-3.5">
              <div className="font-semibold text-[#203653]">Vandrevala Foundation &amp; iCall</div>
              <div className="mt-0.5 text-[#526783]">
                Vandrevala: +91 9999 666 555 (24/7) · iCall (TISS): +91 91529 87821 (Mon–Sat, 10
                AM–8 PM IST)
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
