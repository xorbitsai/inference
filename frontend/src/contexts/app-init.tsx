'use client';

import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { authStore } from '@/lib/auth-store';
import request from '@/lib/request';
import { useGlobal } from '@/contexts/global-context';

interface ClusterAuthResponse {
  auth: false | string;
}

interface AppInitProps {
  clusterAuth: ClusterAuthResponse | null;
  clusterAuthError: string | null;
}

export default function AppInit({ clusterAuth, clusterAuthError }: AppInitProps) {
  const initialized = useRef(false);
  const { setClusterVersion } = useGlobal();
  useEffect(() => {
    if (initialized.current) return;

    initialized.current = true;
    const init = async () => {
      // 接口错误
      if (clusterAuthError) {
        // 解决初始化时 sonner 未挂载
        queueMicrotask(() => {
          toast.error(clusterAuthError);
        });
        return;
      }

      if (!clusterAuth) {
        return;
      }

      // 匿名模式
      if (clusterAuth.auth === false) {
        authStore.set({
          type: 'anonymous',
        });

        document.cookie = 'token=no_auth; path=/';
      } else {
        // token 模式
        authStore.set({
          type: 'token',
          token: clusterAuth.auth,
        });
        document.cookie = 'token=authenticated; path=/';
      }
      const versionRes = await request.get('/v1/cluster/version');
      setClusterVersion(versionRes);
    };
    init();
  }, [clusterAuth, clusterAuthError, setClusterVersion]);

  return null;
}
