import Cookies from 'js-cookie';
import { NO_AUTH } from '@/constants';
import { getApiUrl } from '@/lib/utils';

const LEGACY_TOKEN_COOKIE = 'token';

function hashString(value: string): string {
  let hash = 0;

  for (let index = 0; index < value.length; index += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(index);
    hash |= 0;
  }

  return Math.abs(hash).toString(36);
}

function getAuthScope(): string {
  if (typeof window === 'undefined') {
    return getApiUrl();
  }

  const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');

  return `${window.location.hostname}:${port}:${getApiUrl()}`;
}

// Browser cookies are scoped by domain/path, not by port. Multiple local
// Xinference UIs, such as localhost:3000 and localhost:3999, would overwrite
// each other's auth state if every caller used Cookies.set('token') directly.
// Include the current UI origin and API URL in the cookie name so each instance
// owns an isolated token slot.
export function getTokenCookieName(): string {
  return `xinference_token_${hashString(getAuthScope())}`;
}

// Keep reads side-effect free. Migration is handled explicitly during app init
// so rendering or request code does not unexpectedly mutate cookies.
export function getTokenValue(): string | undefined {
  const tokenName = getTokenCookieName();
  const scopedToken = Cookies.get(tokenName);

  if (scopedToken) {
    return scopedToken;
  }

  const legacyToken = Cookies.get(LEGACY_TOKEN_COOKIE);
  if (legacyToken && legacyToken !== NO_AUTH) {
    return legacyToken;
  }

  return undefined;
}

// Older versions wrote the shared "token" cookie. A real legacy token is moved
// into the scoped cookie once, while NO_AUTH is intentionally ignored so a
// no-login instance cannot poison another login-required instance.
export function migrateLegacyToken(expires = 7): string | undefined {
  const tokenName = getTokenCookieName();
  const scopedToken = Cookies.get(tokenName);

  if (scopedToken) {
    return scopedToken;
  }

  const legacyToken = Cookies.get(LEGACY_TOKEN_COOKIE);
  if (legacyToken && legacyToken !== NO_AUTH) {
    Cookies.set(tokenName, legacyToken, {
      path: '/',
      expires,
    });
    Cookies.remove(LEGACY_TOKEN_COOKIE);
    return legacyToken;
  }

  return undefined;
}

interface SetTokenOptions {
  removeLegacy?: boolean;
}

export function setTokenValue(
  token: string,
  expires?: number,
  options: SetTokenOptions = {}
): void {
  Cookies.set(getTokenCookieName(), token, {
    path: '/',
    ...(expires ? { expires } : {}),
  });

  // Only explicit login/logout flows should clean up the legacy shared cookie.
  // No-auth initialization must not remove it because another open UI instance
  // on the same hostname may still rely on that value.
  if (options.removeLegacy) {
    Cookies.remove(LEGACY_TOKEN_COOKIE);
  }
}

export function removeTokenValue(): void {
  Cookies.remove(getTokenCookieName());
  Cookies.remove(LEGACY_TOKEN_COOKIE);
}
