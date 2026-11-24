import {
  BookOpen,
  FileSearch,
  FlaskConical,
  Home,
  MessageSquareHeart,
  X,
} from 'lucide-react';
import type { PageId } from '@/src/types/mindtrace';

interface SidebarProps {
  activePage: PageId;
  onSelectPage: (page: PageId) => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

const PRIMARY_NAV_ITEMS: { id: PageId; label: string; icon: typeof Home }[] = [
  { id: 'home', label: 'Home', icon: Home },
  { id: 'check-text', label: 'Check Your Text', icon: FileSearch },
  { id: 'understand-result', label: 'Understand Your Result', icon: MessageSquareHeart },
  { id: 'about-project', label: 'About This Project', icon: BookOpen },
];

function MindTraceLogo() {
  return (
    <div
      className="flex h-10 w-10 items-center justify-center rounded-2xl shrink-0"
      style={{
        background: 'linear-gradient(145deg, #EAF0FF, #C9D8FF)',
        border: '1px solid #F2F6FF',
        boxShadow:
          '4px 4px 10px rgba(102, 128, 181, 0.24), -3px -3px 8px rgba(246, 250, 255, 0.92)',
      }}
      aria-hidden="true"
    >
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <path
          d="M4 15.5L8.5 9.5L12.5 13.5L16.5 7.5L20 15.5"
          stroke="#203653"
          strokeWidth="2.1"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="16.5" cy="7.5" r="2.2" fill="#72B9D8" />
      </svg>
    </div>
  );
}

export function Sidebar({
  activePage,
  onSelectPage,
  mobileOpen,
  onCloseMobile,
}: SidebarProps) {
  const isDetailsActive = activePage === 'project-details';

  const navContent = (
    <div
      className="flex h-full flex-col justify-between px-4 py-6"
      style={{
        background: 'linear-gradient(180deg, #EAF0FF 0%, #DCE6FF 100%)',
        borderRight: '1px solid rgba(210, 223, 255, 0.9)',
        boxShadow: '6px 0 24px rgba(102, 128, 181, 0.14)',
      }}
    >
      <div>
        <div className="px-2 pb-5 border-b border-[#D2DFFF]/80">
          <button
            type="button"
            onClick={() => {
              onSelectPage('home');
              onCloseMobile();
            }}
            className="flex items-center gap-3 text-left w-full rounded-xl focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4D6FAF]"
          >
            <MindTraceLogo />
            <div>
              <div className="font-display text-base font-bold tracking-tight text-[#203653] leading-tight">
                MindTrace AI
              </div>
              <div className="text-[11px] text-[#526783]">Calm Text Reflection</div>
            </div>
          </button>
        </div>

        <nav aria-label="Primary navigation" className="mt-6 space-y-2">
          {PRIMARY_NAV_ITEMS.map((item) => {
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
                className={`w-full flex items-center gap-3 rounded-[16px] px-3.5 py-3 text-sm font-medium transition-all whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4D6FAF] ${
                  isActive
                    ? 'text-[#203653] font-semibold'
                    : 'text-[#526783] hover:text-[#203653] hover:bg-[#E5ECFF]/80'
                }`}
                style={
                  isActive
                    ? {
                        background: 'linear-gradient(145deg, #D5E2FF, #AFC5F5)',
                        border: '1px solid #EAF0FF',
                        boxShadow:
                          '5px 5px 12px rgba(96, 123, 178, 0.24), -4px -4px 10px rgba(246, 250, 255, 0.9)',
                      }
                    : undefined
                }
              >
                <Icon
                  className={`h-4 w-4 shrink-0 ${
                    isActive ? 'text-[#203653]' : 'text-[#4D6FAF]'
                  }`}
                  aria-hidden="true"
                />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="mt-6 pt-5 border-t border-[#D2DFFF]/80">
          <div className="px-3 mb-2 text-[11px] font-medium text-[#526783]">
            Academic &amp; Technical Reference
          </div>
          <button
            type="button"
            onClick={() => {
              onSelectPage('project-details');
              onCloseMobile();
            }}
            aria-current={isDetailsActive ? 'page' : undefined}
            className={`w-full flex items-center gap-3 rounded-[16px] px-3.5 py-3 text-sm font-medium transition-all whitespace-nowrap focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#4D6FAF] ${
              isDetailsActive
                ? 'text-[#203653] font-semibold'
                : 'text-[#526783] hover:text-[#203653] hover:bg-[#E5ECFF]/80'
            }`}
            style={
              isDetailsActive
                ? {
                    background: 'linear-gradient(145deg, #D5E2FF, #AFC5F5)',
                    border: '1px solid #EAF0FF',
                    boxShadow:
                      '5px 5px 12px rgba(96, 123, 178, 0.24), -4px -4px 10px rgba(246, 250, 255, 0.9)',
                  }
                : undefined
            }
          >
            <FlaskConical
              className={`h-4 w-4 shrink-0 ${
                isDetailsActive ? 'text-[#203653]' : 'text-[#4D6FAF]'
              }`}
              aria-hidden="true"
            />
            <span className="truncate">Project Details</span>
          </button>
        </div>
      </div>

      <div className="neu-inset p-4 space-y-1.5 text-xs text-[#526783]">
        <div className="font-display font-semibold text-[#203653]">Your Privacy</div>
        <p className="text-[11px] leading-relaxed text-[#526783]">
          Anything you write stays in this temporary session only and is never saved by default.
        </p>
      </div>
    </div>
  );

  return (
    <>
      <aside className="hidden lg:block lg:sticky lg:top-0 lg:h-screen w-[264px] shrink-0">
        {navContent}
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-[#203653]/25 backdrop-blur-xs"
            aria-hidden="true"
            onClick={onCloseMobile}
          />
          <div className="relative z-10 w-[272px] max-w-[85vw] h-full">
            <button
              type="button"
              onClick={onCloseMobile}
              aria-label="Close navigation menu"
              className="neu-btn absolute top-4 right-3 p-2 text-[#203653]"
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
