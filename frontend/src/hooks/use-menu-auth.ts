'use client';

import { useEffect, useMemo, useState } from 'react';

import { NO_AUTH } from '@/constants';
import { getAccessToken } from '@/lib/auth-storage';
import { decodeJwtScopes } from '@/lib/utils';
import { useGlobal } from '@/contexts/global-context';
import { pageAccessExtensions } from '@/lib/ui-extensions';

export function useMenuAuth() {
  const { accountPermissions, clusterUIConfig, clusterAuth, globalReady } = useGlobal();
  const [token, setToken] = useState<string | undefined>();
  const jwtScopes = useMemo(() => decodeJwtScopes(token === NO_AUTH ? undefined : token), [token]);

  useEffect(() => {
    setToken(getAccessToken());
  }, []);

  const permissions = !globalReady
    ? []
    : clusterUIConfig?.auth_advanced
      ? (accountPermissions ?? [])
      : jwtScopes;
  const hasScope = (...scopes: string[]) => permissions.some((item) => scopes.includes(item));
  const hasPageAccess = (path: string) => {
    if (clusterAuth?.auth === false || hasScope('admin')) return true;
    return pageAccessExtensions.every((rule) => {
      if (rule.appliesWhen && !hasScope(rule.appliesWhen)) return true;
      const matches = path === rule.path || (rule.path !== '/' && path.startsWith(rule.path + '/'));
      return !matches || hasScope(rule.scope);
    });
  };

  return {
    permissionsReady:
      !clusterUIConfig?.auth_advanced || clusterAuth?.auth === false || accountPermissions !== null,
    hasPageAccess,
    isAdmin: hasScope('admin'),
    usersManagePage: hasScope('admin', 'users:manage'),
    canCreateKeys: hasScope('admin', 'keys:create'),
    canManageKeys: hasScope('admin', 'keys:manage'),
    canAccessKeysPage: hasScope('admin', 'keys:create', 'keys:manage'),
    hasLogsList: hasScope('admin', 'logs:list'),
    canReadModelRequestBody: hasScope('admin', 'model_requests:read_body'),
    hasMonitorView: hasScope('admin', 'monitor:view'),
    hasSettingsRead: hasScope('admin', 'settings:read'),
    canWriteSettings: hasScope('admin', 'settings:write'),
    hasModelsRead: hasScope('admin', 'models:read'),
    canWriteModels: hasScope('admin', 'models:write'),
    hasCacheList: hasScope('admin', 'cache:list'),
    canDeleteCache: hasScope('admin', 'cache:delete'),
    hasVirtualEnvList: hasScope('admin', 'virtualenv:list'),
    canDeleteVirtualEnv: hasScope('admin', 'virtualenv:delete'),
    canAccessCacheManagement: hasScope('admin', 'models:read', 'cache:list', 'virtualenv:list'),
    canRegisterModel: hasScope('admin', 'models:register', 'models:add', 'models:unregister'),
    hasRouterList: hasScope('admin', 'routers:list'),
    hasRouterRead: hasScope('admin', 'routers:read'),
    canWriteRouters: hasScope('admin', 'routers:write'),
    canOperateRouters: hasScope('admin', 'routers:operate'),
    canAccessRouterPage: hasScope('admin', 'routers:list', 'routers:read'),
  };
}
