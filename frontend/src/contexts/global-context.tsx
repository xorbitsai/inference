'use client';

import { createContext, PropsWithChildren, useContext, useState } from 'react';
import request from '@/lib/request';
import type { ClusterAuth, ClusterVersion } from '@/types/services';

export interface GlobalState {
  /**
   * 全局接口是否初始化完成（/cluster/auth）
   */
  globalReady: boolean;
  clusterAuth: ClusterAuth;
  setClusterAuth: (value: ClusterAuth) => void;
  clusterVersion: ClusterVersion;
  setClusterVersion: (value: ClusterVersion) => void;
  fetchGlobalAfterAuth: () => void;
}

const GlobalContext = createContext<GlobalState | null>(null);

interface Props extends PropsWithChildren {
  initClusterAuth: ClusterAuth;
}

export function GlobalProvider({ children, initClusterAuth }: Props) {
  const [clusterAuth, setClusterAuth] = useState(initClusterAuth);
  const [clusterVersion, setClusterVersion] = useState({} as ClusterVersion);
  const [globalReady, setGlobalReady] = useState(false);

  const fetchGlobalAfterAuth = async () => {
    setGlobalReady(false);
  
    try {
      // 后续全局接口，在此处扩展
      const [versionRes] = await Promise.allSettled([
        request.get('/v1/cluster/version'),
      ]);
  
      if (versionRes.status === 'fulfilled') {
        setClusterVersion(versionRes.value);
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
