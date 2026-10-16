'use client';

import { usePathname } from 'next/navigation';
import { Sidebar } from '@/components/layout/sidebar';
import { HIDE_SIDEBAR_PATHS } from '@/constants';
import { cn } from '@/lib/utils';
import { useMenuAuth } from '@/hooks/use-menu-auth';
import { useI18n } from '@/contexts/i18n-context';
import { useGlobal } from '@/contexts/global-context';
interface LayoutContentProps {
  children: React.ReactNode;
}

export function LayoutContent({ children }: LayoutContentProps) {
  const pathname = usePathname();
  const hiddenSidebar = HIDE_SIDEBAR_PATHS.includes(pathname);
  const auth = useMenuAuth();
  const { t } = useI18n();
  const { globalReady } = useGlobal();
  return (
    <div className="flex h-screen bg-background relative overflow-hidden">
      {!hiddenSidebar && <Sidebar />}
      <main
        className={cn(
          'flex-1 flex flex-col overflow-hidden bg-background overflow-y-auto',
          hiddenSidebar ? '' : 'p-8'
        )}
      >
        {hiddenSidebar || (globalReady && auth.permissionsReady && auth.hasPageAccess(pathname)) ? (
          children
        ) : (
          <p role="status">
            {auth.permissionsReady ? t('common.accessDenied') : t('common.loading')}
          </p>
        )}
      </main>
    </div>
  );
}
