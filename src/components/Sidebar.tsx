import {
  BarChart3,
  BookOpen,
  FileSearch,
  LayoutDashboard,
  PieChart,
  SlidersHorizontal,
  X,
} from 'lucide-react';
import type {
  DatasetSummary,
  EndpointConfig,
  NamedEvaluationRun,
  PageId,
} from '@/src/types/mindtrace';

interface SidebarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  endpointConfig: EndpointConfig;
  evaluationRuns: NamedEvaluationRun[];
  datasetSummary: DatasetSummary | null;
}

const NAV_ITEMS: { id: PageId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'text-analysis', label: 'Text Analysis', icon: FileSearch },
  { id: 'explainable-ai', label: 'Explainable AI', icon: SlidersHorizontal },
  { id: 'model-performance', label: 'Model Performance', icon: BarChart3 },
  { id: 'dataset-analytics', label: 'Dataset Analytics', icon: PieChart },
  { id: 'about-methodology', label: 'About & Methodology', icon: BookOpen },
];

function MindTraceLogo() {
  return (
    <svg
      width="28"
      height="28"
      viewBox="0 0 28 28"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className="shrink-0"
    >
      <rect width="28" height="28" rx="6" fill="#0F2942" />
      <path
        d="M6.5 17.5L10.5 11.5L14 15.5L17.5 9.5L21.5 17.5"
        stroke="#FFFFFF"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="17.5" cy="9.5" r="2" fill="#2DD4BF" />
    </svg>
  );
}

export function Sidebar({
  activePage,
  onSelectPage,
  mobileOpen,
  onCloseMobile,
  endpointConfig,
  evaluationRuns,
  datasetSummary,
}: SidebarProps) {
  const modelStatusText =
    endpointConfig.status === 'connected'
      ? `Connected (${endpointConfig.backbone})`
      : 'Not Connected';

  const navContent = (
    <div className="flex h-full flex-col justify-between bg-white border-r border-[#E2E8F0] px-4 py-6">
      <div>
        <div className="px-2 pb-5 border-b border-[#E2E8F0]">
          <div className="flex items-center gap-2.5">
            <MindTraceLogo />
            <div>
              <div className="font-display text-base font-bold tracking-tight text-[#0F2942] leading-tight">
                MindTrace AI
              </div>
              <div className="text-[11px] text-[#64748B]">NLP Research Portal</div>
            </div>
          </div>
        </div>

        <nav aria-label="Primary navigation" className="mt-5 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  onSelectPage(item.id);
                  onCloseMobile();
                }}
                aria-current={isActive ? 'page' : undefined}
                className={`w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB] ${
                  isActive
                    ? 'bg-[#0F2942] text-white'
                    : 'text-[#475569] hover:bg-[#F1F5F9] hover:text-[#0F172A]'
                }`}
              >
                <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="border-t border-[#E2E8F0] pt-4 px-2 space-y-2.5 text-xs text-[#64748B]">
        <div className="font-semibold text-[#0F172A]">Session Readiness</div>
        <div className="flex items-center justify-between gap-2">
          <span>Model Bridge</span>
          <span className="font-mono tabular-nums text-[#0F172A] truncate max-w-[130px]">
            {modelStatusText}
          </span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span>Evaluations</span>
          <span className="font-mono tabular-nums text-[#0F172A]">
            {evaluationRuns.length > 0
              ? `${evaluationRuns.length} run${evaluationRuns.length > 1 ? 's' : ''}`
              : 'No CSV'}
          </span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span>Dataset</span>
          <span className="font-mono tabular-nums text-[#0F172A]">
            {datasetSummary ? `N=${datasetSummary.validRows}` : 'No CSV'}
          </span>
        </div>
        <div className="pt-1 text-[11px] leading-snug text-[#64748B]">
          Ephemeral memory only. Submitted text is never saved by default.
        </div>
      </div>
    </div>
  );

  return (
    <>
      <aside className="hidden lg:block lg:sticky lg:top-0 lg:h-screen w-[260px] shrink-0">
        {navContent}
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/35"
            aria-hidden="true"
            onClick={onCloseMobile}
          />
          <div className="relative z-10 w-[272px] max-w-[85vw] h-full bg-white shadow-xl">
            <button
              type="button"
              onClick={onCloseMobile}
              aria-label="Close navigation menu"
              className="absolute top-4 right-3 rounded-md p-2 text-[#475569] hover:bg-[#F1F5F9] hover:text-[#0F172A]"
            >
              <X className="h-4 w-4" />
            </button>
            {navContent}
          </div>
        </div>
      )}
    </>
  );
}
