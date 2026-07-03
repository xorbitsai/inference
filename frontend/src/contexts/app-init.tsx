'use client';

import { useEffect, useRef } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { toast } from 'sonner';
import { NO_AUTH, LOGIN_PATH } from '@/constants';
import { useGlobal } from '@/contexts/global-context';
import { migrateLegacyToken, setTokenValue } from '@/lib/auth-token';

interface ClusterAuthResponse {
  auth: boolean;
}

interface AppInitProps {
  clusterAuth: ClusterAuthResponse | null;
  clusterAuthError: string | null;
}

export default function AppInit({ clusterAuth, clusterAuthError }: AppInitProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { fetchGlobalAfterAuth } = useGlobal();

  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current) return;

    initialized.current = true;
    const init = async () => {
      // service error
      if (clusterAuthError) {
        // Fix Sonner not being mounted during initialization
        queueMicrotask(() => {
          toast.error(clusterAuthError);
        });
        return;
      }
      // no_auth (login not required)
      if (clusterAuth?.auth === false) {
        setTokenValue(NO_AUTH, undefined, { removeLegacy: false });
        if (pathname === LOGIN_PATH) router.push('/');
        fetchGlobalAfterAuth();
        return;
      }
      // The following requires login logic
      const token = migrateLegacyToken();
      // If auth is true && already logged in -> request global APIs directly
      if (token && token !== NO_AUTH) {
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
  }, [clusterAuth, clusterAuthError, pathname, router, fetchGlobalAfterAuth]);

  return null;
}
