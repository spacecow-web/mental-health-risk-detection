import type { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  description: string;
  action?: ReactNode;
}

export function PageHeader({ title, description, action }: PageHeaderProps) {
  return (
    <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 pb-5 border-b border-[#AFC5F5]/60">
      <div className="max-w-3xl">
        <h1 className="font-display text-2xl sm:text-[28px] font-bold tracking-tight text-[#203653]">
          {title}
        </h1>
        <p className="mt-1.5 text-sm leading-relaxed text-[#526783]">{description}</p>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}
