'use client';

import { useEffect, useRef } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { toast } from 'sonner';
import { NO_AUTH, LOGIN_PATH } from '@/constants';
import { useGlobal } from '@/contexts/global-context';
import { getAccessToken, getRefreshToken, setNoAuthToken } from '@/lib/auth-storage';
import { getApiUrl } from '@/lib/utils';
import type { ClusterAuth } from '@/types/services';

export default function AppInit() {
  const router = useRouter();
  const pathname = usePathname();
  const { setClusterAuth, fetchGlobalAfterAuth } = useGlobal();

  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current) return;

    initialized.current = true;
    const init = async () => {
      // Fetched client-side: the app ships as a static export, so there is no
      // request-time server rendering to resolve the cluster auth mode.
      let clusterAuth: ClusterAuth | null = null;
      try {
        const res = await fetch(getApiUrl() + '/v1/cluster/auth', {
          cache: 'no-store',
        });
        let data = null;
        try {
          data = await res.json();
        } catch {
          data = null;
        }
        if (!res.ok) {
          throw new Error(`Server error: ${res.status} - ${data?.detail || 'Unknown error'}`);
        }
        clusterAuth = data;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Cluster auth failed';
        // Fix Sonner not being mounted during initialization
        queueMicrotask(() => {
          toast.error(message);
        });
        return;
      }
      setClusterAuth(clusterAuth);
      // no_auth (login not required)
      if (clusterAuth?.auth === false) {
        setNoAuthToken();
        if (pathname === LOGIN_PATH) router.push('/');
        fetchGlobalAfterAuth();
        return;
      }
      // The following requires login logic
      const token = getAccessToken();
      const refreshToken = getRefreshToken();
      // If auth is true && already logged in -> request global APIs directly
      if ((token && token !== NO_AUTH) || refreshToken) {
        if (pathname === LOGIN_PATH) router.push('/');
        fetchGlobalAfterAuth();
        return;
      }
      // Redirect to login page, fetch token after login
      if (pathname !== LOGIN_PATH) {
        router.replace('/login');
      }
    };
    init();
  }, [pathname, router, setClusterAuth, fetchGlobalAfterAuth]);

  return null;
}
