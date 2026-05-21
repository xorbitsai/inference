'use client';

import { useEffect, useRef } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { toast } from 'sonner';
import Cookies from 'js-cookie';
import { NO_AUTH, LOGIN_PATH } from '@/constants';
import { useGlobal } from '@/contexts/global-context';

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
      // 接口错误
      if (clusterAuthError) {
        // 解决初始化时 sonner 未挂载
        queueMicrotask(() => {
          toast.error(clusterAuthError);
        });
        return;
      }
      // no_auth(不需要登录)
      if (clusterAuth?.auth === false) {
        Cookies.set('token', NO_AUTH, {
          path: '/',
        });
        if (pathname === LOGIN_PATH) router.push('/');
        fetchGlobalAfterAuth();
        return;
      }
      // 以下为需要登录逻辑
      const token = Cookies.get('token');
      // auth 为 true && 已经登录 -> 则直接请求全局接口
      if (token && token !== NO_AUTH) {
        if (pathname === LOGIN_PATH) router.push('/');
        fetchGlobalAfterAuth();
        return;
      }
      // 跳转登录页,登录后获取token
      if (pathname !== LOGIN_PATH) {
        router.replace('/login');
      }
    };
    init();
  }, [clusterAuth, clusterAuthError, pathname, router, fetchGlobalAfterAuth]);

  return null;
}
