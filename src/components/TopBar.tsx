import { Download, Menu, RotateCcw } from 'lucide-react';
import type { PageId } from '@/src/types/mindtrace';

interface TopBarProps {
  activePage: PageId;
  onOpenMobileMenu: () => void;
  onDownloadPythonKit: () => void;
  downloadingKit: boolean;
  hasSessionData: boolean;
  onClearSession: () => void;
}

const PAGE_TITLES: Record<PageId, string> = {
  overview: 'Overview',
  'text-analysis': 'Text Analysis',
  'explainable-ai': 'Explainable AI',
  'model-performance': 'Model Performance',
  'dataset-analytics': 'Dataset Analytics',
  'about-methodology': 'About & Methodology',
};

export function TopBar({
  activePage,
  onOpenMobileMenu,
  onDownloadPythonKit,
  downloadingKit,
  hasSessionData,
  onClearSession,
}: TopBarProps) {
  return (
    <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-[#E2E8F0]">
      <div className="mx-auto flex items-center justify-between gap-4 px-5 py-3.5 lg:px-8">
        <div className="flex items-center gap-3 min-w-0">
          <button
            type="button"
            onClick={onOpenMobileMenu}
            aria-label="Open navigation menu"
            className="lg:hidden rounded-md p-2 text-[#475569] hover:bg-[#F1F5F9] hover:text-[#0F172A] min-h-[40px] min-w-[40px] flex items-center justify-center"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2 text-sm min-w-0">
            <span className="hidden sm:inline text-[#64748B] whitespace-nowrap">MindTrace AI</span>
            <span className="hidden sm:inline text-[#CBD5E1]" aria-hidden="true">
              /
            </span>
            <h1 className="font-display text-base font-semibold text-[#0F172A] truncate">
              {PAGE_TITLES[activePage]}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {hasSessionData && (
            <button
              type="button"
              onClick={onClearSession}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E8F0] bg-white px-3.5 py-2 text-xs font-medium text-[#475569] hover:bg-[#F1F5F9] hover:text-[#0F172A] transition-colors whitespace-nowrap shrink-0 min-h-[40px]"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Clear Session</span>
            </button>
          )}
          <button
            type="button"
            onClick={onDownloadPythonKit}
            disabled={downloadingKit}
            className="inline-flex items-center gap-2 rounded-lg bg-[#0F2942] px-4 py-2 text-xs font-medium text-white hover:bg-[#173A5E] disabled:opacity-60 transition-colors whitespace-nowrap shrink-0 min-h-[40px]"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            <span>{downloadingKit ? 'Packaging ZIP...' : 'Download Python Kit'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
