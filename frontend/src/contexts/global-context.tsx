'use client';

import {
  createContext,
  PropsWithChildren,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import request from '@/lib/request';
import { getAccessToken } from '@/lib/auth-storage';
import { decodeJwtPayload } from '@/lib/utils';
import { NO_AUTH } from '@/constants';
import type { ClusterAuth, ClusterUIConfig, ClusterVersion } from '@/types/services';

export interface GlobalState {
  /**
   * global APIs is ready（/cluster/auth）
   */
  globalReady: boolean;
  clusterAuth: ClusterAuth | null;
  setClusterAuth: (value: ClusterAuth | null) => void;
  clusterVersion: ClusterVersion;
  setClusterVersion: (value: ClusterVersion) => void;
  clusterUIConfig: ClusterUIConfig;
  setClusterUIConfig: (value: ClusterUIConfig) => void;
  fetchGlobalAfterAuth: () => void;
  accountPermissions: string[] | null;
  accountSource: string | null;
  refreshAccountPermissions: () => Promise<void>;
}

const GlobalContext = createContext<GlobalState | null>(null);

function accountIdentity(token: string | undefined): string | undefined {
  if (!token || token === NO_AUTH) return undefined;
  const payload = decodeJwtPayload(token);
  // Only a UI cache key: request authorization remains server-side. Unknown
  // token formats never carry permissions over to a different credential.
  return payload?.user_id != null ? JSON.stringify([payload.user_id, payload.sub]) : token;
}

interface Props extends PropsWithChildren {
  initClusterAuth?: ClusterAuth | null;
}

export function GlobalProvider({ children, initClusterAuth = null }: Props) {
  const [clusterAuth, setClusterAuth] = useState(initClusterAuth);
  const [clusterVersion, setClusterVersion] = useState({} as ClusterVersion);
  const [clusterUIConfig, setClusterUIConfig] = useState({} as ClusterUIConfig);
  const [globalReady, setGlobalReady] = useState(false);
  const [accountPermissions, setAccountPermissions] = useState<string[] | null>(null);
  const [accountSource, setAccountSource] = useState<string | null>(null);
  const permissionRequest = useRef(0);
  const permissionIdentity = useRef<string | undefined>(undefined);
  const refreshAccountPermissions = useCallback(async () => {
    const requestId = ++permissionRequest.current;
    const token = getAccessToken();
    const identity = accountIdentity(token);
    if (permissionIdentity.current !== identity) {
      permissionIdentity.current = identity;
      setAccountPermissions(null);
      setAccountSource(null);
    }
    if (!clusterUIConfig?.auth_advanced || !token || token === NO_AUTH) {
      setAccountPermissions(null);
      setAccountSource(null);
      return;
    }
    try {
      const account = await request.get<{ permissions: string[]; source: string }>('/v1/auth/me', {
        suppressGlobalError: true,
      });
      // Do not publish a response belonging to a logged-out/replaced session.
      if (
        accountIdentity(getAccessToken()) === identity &&
        requestId === permissionRequest.current
      ) {
        setAccountPermissions(account.permissions);
        setAccountSource(account.source);
      }
    } catch {
      // A transient failure must not unmount the page or discard user input.
      // Identity changes/logout already clear this cache above. Successful
      // revocations still apply immediately; the backend authorizes every API.
    }
  }, [clusterUIConfig?.auth_advanced]);

  useEffect(() => {
    void refreshAccountPermissions();
    const refresh = () => {
      void refreshAccountPermissions();
    };
    window.addEventListener('focus', refresh);
    window.addEventListener('xinference-auth-change', refresh);
    const timer = window.setInterval(refresh, 15000);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('focus', refresh);
      window.removeEventListener('xinference-auth-change', refresh);
    };
  }, [refreshAccountPermissions]);

  const fetchGlobalAfterAuth = useCallback(async () => {
    // Extend global APIs here as needed
    const [versionRes, uiConfigRes] = await Promise.allSettled([
      request.get('/v1/cluster/version', { suppressGlobalError: true }),
      request.get('/v1/cluster/ui_config'),
    ]);

    if (versionRes.status === 'fulfilled') {
      setClusterVersion(versionRes.value);
    }

    if (uiConfigRes.status === 'fulfilled') {
      setClusterUIConfig(uiConfigRes.value);
      setGlobalReady(true);
    }
  }, []);
  return (
    <GlobalContext.Provider
      value={{
        globalReady,
        clusterAuth,
        setClusterAuth,
        clusterVersion,
        setClusterVersion,
        clusterUIConfig,
        setClusterUIConfig,
        fetchGlobalAfterAuth,
        accountPermissions,
        accountSource,
        refreshAccountPermissions,
      }}
    >
      {children}
    </GlobalContext.Provider>
  );
}

export function useGlobal() {
  const context = useContext(GlobalContext);

  if (!context) {
    throw new Error('useGlobal must be used within GlobalProvider');
  }

  return context;
}
