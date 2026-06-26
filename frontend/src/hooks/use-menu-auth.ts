'use client';

import Cookies from 'js-cookie';
import { NO_AUTH } from '@/constants';
import { decodeJwtScopes } from '@/lib/utils';

export function useMenuAuth() {
  const token = Cookies.get('token');
  const jwtScopes = decodeJwtScopes(token === NO_AUTH ? undefined : token);

  const hasScope = (...scopes: string[]) => jwtScopes.some((item) => scopes.includes(item));

  return {
    isAdmin: hasScope('admin'),
    usersManagePage: hasScope('admin', 'users:manage'),
    keysManagePage: hasScope('admin', 'keys:manage'),
    keysManageCreate: hasScope('admin', 'keys:create'),
  };
}
