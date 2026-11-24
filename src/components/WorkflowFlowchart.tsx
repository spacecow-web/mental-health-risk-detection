import { ArrowRight } from 'lucide-react';

const WORKFLOW_STEPS = [
  {
    step: '1',
    title: 'User enters text',
    detail: 'You type or paste a short message in private session memory.',
  },
  {
    step: '2',
    title: 'Text preprocessing',
    detail: 'Links and usernames are masked and spacing is cleaned.',
  },
  {
    step: '3',
    title: 'Transformer model analyses text',
    detail: 'The language model reads how all words connect in context.',
  },
  {
    step: '4',
    title: 'AI identifies relevant patterns',
    detail: 'It checks which learned language tone matches most closely.',
  },
  {
    step: '5',
    title: 'Explainable AI highlights influential words',
    detail: 'SHAP and LIME measure which words shaped that reading.',
  },
  {
    step: '6',
    title: 'Simple result is displayed',
    detail: 'A clear, supportive summary appears with key standout words.',
  },
];

export function WorkflowFlowchart() {
  return (
    <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
      {WORKFLOW_STEPS.map((item, idx) => (
        <div key={item.step} className="neu-inset p-4 space-y-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-[#DCE6FC] border border-[#F2F6FF] font-mono text-xs font-semibold text-[#203653]">
              {item.step}
            </span>
            {idx < WORKFLOW_STEPS.length - 1 && (
              <ArrowRight className="h-3.5 w-3.5 text-[#7799D8]" aria-hidden="true" />
            )}
          </div>
          <h3 className="font-display text-sm font-semibold text-[#203653]">{item.title}</h3>
          <p className="text-xs leading-relaxed text-[#526783]">{item.detail}</p>
        </div>
      ))}
    </div>
  );
}
