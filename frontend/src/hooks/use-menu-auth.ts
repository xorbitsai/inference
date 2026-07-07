'use client';

import { useEffect, useMemo, useState } from 'react';

import { NO_AUTH } from '@/constants';
import { getAccessToken } from '@/lib/auth-storage';
import { decodeJwtScopes } from '@/lib/utils';

export function useMenuAuth() {
  const [token, setToken] = useState<string | undefined>();
  const jwtScopes = useMemo(() => decodeJwtScopes(token === NO_AUTH ? undefined : token), [token]);

  useEffect(() => {
    setToken(getAccessToken());
  }, []);

  const hasScope = (...scopes: string[]) => jwtScopes.some((item) => scopes.includes(item));

  return {
    isAdmin: hasScope('admin'),
    usersManagePage: hasScope('admin', 'users:manage'),
    keysManagePage: hasScope('admin', 'keys:manage'),
    keysManageCreate: hasScope('admin', 'keys:create'),
  };
}
