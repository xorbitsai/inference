'use client';

import {
  createContext,
  PropsWithChildren,
  useContext,
  useState,
} from 'react';

import type { ClusterAuth, ClusterVersion } from '@/types/services';

export interface GlobalState {
  /**
   * 全局接口是否初始化完成（/cluster/auth）
   */
  // globalReady: boolean;
  clusterAuth: ClusterAuth;
  setClusterAuth: (value: ClusterAuth)=> void;
  clusterVersion: ClusterVersion;
  setClusterVersion: (value: ClusterVersion)=> void;
}

const GlobalContext =
  createContext<GlobalState | null>(null);

interface Props extends PropsWithChildren {
  initClusterAuth: ClusterAuth;
}

export function GlobalProvider({
  children,
  initClusterAuth
}: Props) {
  const [clusterAuth, setClusterAuth] = useState(initClusterAuth);
  const [clusterVersion, setClusterVersion] = useState({} as ClusterVersion);
  return (
    <GlobalContext.Provider value={{
      clusterAuth,
      setClusterAuth,
      clusterVersion,
      setClusterVersion,
    }}>
      {children}
    </GlobalContext.Provider>
  );
}

export function useGlobal() {
  const context = useContext(GlobalContext);

  if (!context) {
    throw new Error(
      'useGlobal must be used within GlobalProvider'
    );
  }

  return context;
}