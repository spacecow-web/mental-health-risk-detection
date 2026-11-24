import { Menu, RotateCcw } from 'lucide-react';
import type { PageId } from '@/src/types/mindtrace';

interface TopBarProps {
  activePage: PageId;
  onOpenMobileMenu: () => void;
  hasSessionData: boolean;
  onClearSession: () => void;
}

const PAGE_TITLES: Record<PageId, string> = {
  home: 'Home',
  'check-text': 'Check Your Text',
  'understand-result': 'Understand Your Result',
  'about-project': 'About This Project',
  'project-details': 'Project Details',
};

export function TopBar({
  activePage,
  onOpenMobileMenu,
  hasSessionData,
  onClearSession,
}: TopBarProps) {
  return (
    <header
      className="sticky top-0 z-30 backdrop-blur-md border-b border-[#D2DFFF]/90"
      style={{
        background: 'rgba(234, 240, 255, 0.84)',
        boxShadow: '0 4px 18px rgba(96, 123, 178, 0.1)',
      }}
    >
      <div className="mx-auto flex items-center justify-between gap-4 px-5 py-3.5 lg:px-8">
        <div className="flex items-center gap-3 min-w-0">
          <button
            type="button"
            onClick={onOpenMobileMenu}
            aria-label="Open navigation menu"
            className="neu-btn lg:hidden p-2 text-[#203653] min-h-[40px] min-w-[40px] flex items-center justify-center"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2 text-sm min-w-0">
            <span className="hidden sm:inline text-[#526783] whitespace-nowrap">MindTrace AI</span>
            <span className="hidden sm:inline text-[#91B9E8]" aria-hidden="true">
              /
            </span>
            <h1 className="font-display text-base font-semibold text-[#203653] truncate">
              {PAGE_TITLES[activePage]}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {hasSessionData && (
            <button
              type="button"
              onClick={onClearSession}
              className="neu-btn inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium whitespace-nowrap shrink-0 min-h-[40px]"
            >
              <RotateCcw className="h-3.5 w-3.5 text-[#4D6FAF]" aria-hidden="true" />
              <span>Start Fresh</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
