'use client';

import { Loader2, ShieldAlert } from 'lucide-react';
import { useI18n } from '@/contexts/i18n-context';
import { useGlobal } from '@/contexts/global-context';
import { useMenuAuth } from '@/hooks/use-menu-auth';

interface PermissionGuardProps {
  scope: 'monitor:view' | 'logs:list' | 'models:register';
  children: React.ReactNode;
}

export function PermissionGuard({ scope, children }: PermissionGuardProps) {
  const { t } = useI18n();
  const { clusterUIConfig, globalReady } = useGlobal();
  const auth = useMenuAuth();

  // Wait for the global configuration (cluster auth, ui_config) to load
  // before making any visibility decision.  clusterUIConfig starts as {},
  // so checking auth_advanced before globalReady would render children
  // for unauthorized users during the initial load window.
  if (!globalReady) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // When auth_advanced is disabled, all pages are accessible
  if (!clusterUIConfig?.auth_advanced) {
    return <>{children}</>;
  }

  // Admin has all scopes
  if (auth.isAdmin) {
    return <>{children}</>;
  }

  const scopeMap: Record<string, boolean> = {
    'monitor:view': auth.hasMonitorView,
    'logs:list': auth.hasLogsList,
    'models:register': auth.canRegisterModel,
  };

  if (scopeMap[scope]) {
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 text-muted-foreground">
      <ShieldAlert className="h-16 w-16" />
      <h2 className="text-xl font-semibold">{t('common.accessDenied')}</h2>
      <p className="text-sm">{t('common.accessDeniedDescription', { scope })}</p>
    </div>
  );
}
