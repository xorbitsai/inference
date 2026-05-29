'use client';

import { usePathname } from 'next/navigation';
import { Sidebar } from '@/components/layout/sidebar';
import { HIDE_SIDEBAR_PATHS } from '@/constants';
import { cn } from '@/lib/utils';
interface LayoutContentProps {
  children: React.ReactNode;
}

export function LayoutContent({ children }: LayoutContentProps) {
  const pathname = usePathname();
  const hiddenSidebar = HIDE_SIDEBAR_PATHS.includes(pathname);
  return (
    <div className="flex h-screen bg-background relative overflow-hidden">
      {!hiddenSidebar && <Sidebar />}
      <main
        className={cn(
          'flex-1 flex flex-col overflow-hidden bg-background overflow-y-auto',
          hiddenSidebar ? '' : 'p-8'
        )}
      >
        {children}
      </main>
    </div>
  );
}
