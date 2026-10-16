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
  refreshAccountPermissions: () => Promise<void>;
}

const GlobalContext = createContext<GlobalState | null>(null);

interface Props extends PropsWithChildren {
  initClusterAuth?: ClusterAuth | null;
}

export function GlobalProvider({ children, initClusterAuth = null }: Props) {
  const [clusterAuth, setClusterAuth] = useState(initClusterAuth);
  const [clusterVersion, setClusterVersion] = useState({} as ClusterVersion);
  const [clusterUIConfig, setClusterUIConfig] = useState({} as ClusterUIConfig);
  const [globalReady, setGlobalReady] = useState(false);
  const [accountPermissions, setAccountPermissions] = useState<string[] | null>(null);
  const permissionRequest = useRef(0);
  const permissionToken = useRef<string | undefined>(undefined);
  const refreshAccountPermissions = useCallback(async () => {
    const requestId = ++permissionRequest.current;
    const token = getAccessToken();
    if (permissionToken.current !== token) {
      permissionToken.current = token;
      setAccountPermissions(null);
    }
    if (!clusterUIConfig?.auth_advanced || !token || token === NO_AUTH) {
      setAccountPermissions(null);
      return;
    }
    try {
      const account = await request.get<{ permissions: string[] }>('/v1/auth/me', {
        suppressGlobalError: true,
      });
      // Do not publish a response belonging to a logged-out/replaced session.
      if (getAccessToken() === token && requestId === permissionRequest.current) {
        setAccountPermissions(account.permissions);
      }
    } catch {
      if (requestId === permissionRequest.current) setAccountPermissions(null);
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

  const fetchGlobalAfterAuth = async () => {
    setGlobalReady(false);

    try {
      // Extend global APIs here as needed
      const [versionRes, uiConfigRes] = await Promise.allSettled([
        request.get('/v1/cluster/version'),
        request.get('/v1/cluster/ui_config'),
      ]);

      if (versionRes.status === 'fulfilled') {
        setClusterVersion(versionRes.value);
      }

      if (uiConfigRes.status === 'fulfilled') {
        setClusterUIConfig(uiConfigRes.value);
      }
    } finally {
      setGlobalReady(true);
    }
  };
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
