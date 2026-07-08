'use client';

import { createContext, PropsWithChildren, useContext, useState } from 'react';
import request from '@/lib/request';
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
