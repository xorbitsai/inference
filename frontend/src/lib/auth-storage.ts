import { NO_AUTH } from '@/constants';

const ACCESS_TOKEN_KEY = 'token';
const REFRESH_TOKEN_KEY = 'refresh_token';

function getSessionValue(key: string): string | undefined {
  if (typeof window === 'undefined') {
    return undefined;
  }

  return sessionStorage.getItem(key) || undefined;
}

export function getAccessToken(): string | undefined {
  return getSessionValue(ACCESS_TOKEN_KEY);
}

export function setAccessToken(token: string): void {
  if (typeof window === 'undefined') {
    return;
  }

  sessionStorage.setItem(ACCESS_TOKEN_KEY, token);
}

export function getRefreshToken(): string | undefined {
  return getSessionValue(REFRESH_TOKEN_KEY);
}

export function setRefreshToken(token: string): void {
  if (typeof window === 'undefined') {
    return;
  }

  sessionStorage.setItem(REFRESH_TOKEN_KEY, token);
}

export function setNoAuthToken(): void {
  setAccessToken(NO_AUTH);
}

export function removeAuthTokens(): void {
  if (typeof window === 'undefined') {
    return;
  }

  sessionStorage.removeItem(ACCESS_TOKEN_KEY);
  sessionStorage.removeItem(REFRESH_TOKEN_KEY);
}
