'use client';

import { useEffect, useRef } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { toast } from 'sonner';
import { NO_AUTH, LOGIN_PATH, SETUP_PATH } from '@/constants';
import { useGlobal } from '@/contexts/global-context';
import { getAccessToken, getRefreshToken, setNoAuthToken } from '@/lib/auth-storage';
import { getApiUrl } from '@/lib/utils';
import type { ClusterAuth } from '@/types/services';

interface SetupStatus {
  needs_setup?: boolean;
}

async function fetchNeedsSetup(): Promise<boolean> {
  try {
    const res = await fetch(getApiUrl() + '/v1/admin/setup/status', {
      cache: 'no-store',
    });
    // A 404 here means this deployment isn't running advanced auth (the
    // endpoint only exists under it), so there's nothing to set up.
    if (!res.ok) return false;
    const data: SetupStatus = await res.json();
    return Boolean(data.needs_setup);
  } catch {
    return false;
  }
}

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
        if (pathname === LOGIN_PATH || pathname === SETUP_PATH) router.push('/');
        fetchGlobalAfterAuth();
        return;
      }
      // The following requires login logic
      const token = getAccessToken();
      const refreshToken = getRefreshToken();
      // If auth is true && already logged in -> request global APIs directly
      if ((token && token !== NO_AUTH) || refreshToken) {
        if (pathname === LOGIN_PATH || pathname === SETUP_PATH) router.push('/');
        fetchGlobalAfterAuth();
        return;
      }
      // Not logged in: a fresh deployment with no accounts yet needs to be
      // routed to /setup instead of /login (there's nothing to log into).
      const needsSetup = await fetchNeedsSetup();
      if (needsSetup) {
        if (pathname !== SETUP_PATH) router.replace(SETUP_PATH);
        return;
      }
      // Setup already completed (or not applicable): send /setup visitors
      // to the normal login page, and redirect anywhere else to login.
      if (pathname === SETUP_PATH) {
        router.replace(LOGIN_PATH);
        return;
      }
      if (pathname !== LOGIN_PATH) {
        router.replace(LOGIN_PATH);
      }
    };
    init();
  }, [pathname, router, setClusterAuth, fetchGlobalAfterAuth]);

  return null;
}
