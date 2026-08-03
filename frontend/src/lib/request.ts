import axios from 'axios';
import type { AxiosRequestConfig } from 'axios';
import { RequestEvents, NO_AUTH } from '@/constants';
import { eventBus } from '@/lib/event-bus';
import { requestManager } from '@/lib/request-manager';
import { getApiUrl } from '@/lib/utils';
import {
  getAccessToken,
  getRefreshToken,
  setAccessToken,
  setRefreshToken,
} from '@/lib/auth-storage';

declare module 'axios' {
  export interface AxiosRequestConfig {
    noTimeout?: boolean;
    skipAuthRefresh?: boolean;
    _retry?: boolean;
  }

  export interface AxiosError {
    /** Parsed backend error, attached by the response interceptor so callers
     * can render the real cause without re-parsing the payload. */
    xinferenceError?: ServerErrorPayload;
  }
}

export interface ServerErrorPayload {
  detail: string;
  traceback?: string;
}

/**
 * Normalize a FastAPI error body into a readable string.
 *
 * `detail` is usually a string, but request-validation failures (422) return
 * an array of `{loc, msg, type}` objects, which would otherwise render as
 * "[object Object]".
 */
function extractDetail(data: unknown): string {
  if (typeof data === 'string') return data;
  if (Array.isArray(data)) {
    const messages = data
      .map((item) =>
        item && typeof item === 'object' && 'msg' in item
          ? String((item as { msg: unknown }).msg)
          : typeof item === 'string'
            ? item
            : safeStringify(item)
      )
      .filter(Boolean);
    if (messages.length) return messages.join('; ');
  }
  if (data && typeof data === 'object') return safeStringify(data);
  return String(data ?? '');
}

/** `JSON.stringify` throws on BigInt and circular references; this runs while
 * already handling an error, so it must never throw itself. */
function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

// Keep untyped request calls backward-compatible while typed calls can still pass <T>.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type LooseResponse = any;

interface TokenResponse {
  access_token?: string;
  refresh_token?: string;
}

const requestInstance = axios.create({
  baseURL: getApiUrl(),
  timeout: 60000,
});

let refreshTokenPromise: Promise<string> | null = null;

function shouldRefreshToken(status: number, config?: AxiosRequestConfig): boolean {
  if (status !== 401 || !config || config._retry || config.skipAuthRefresh) {
    return false;
  }

  const url = config.url || '';
  if (url === '/token' || url === '/v1/auth/refresh') {
    return false;
  }

  return Boolean(getRefreshToken());
}

async function refreshAccessToken(): Promise<string> {
  if (!refreshTokenPromise) {
    refreshTokenPromise = (async () => {
      const refreshToken = getRefreshToken();
      if (!refreshToken) {
        throw new Error('Missing refresh token');
      }

      const response = await axios.post<TokenResponse>(`${getApiUrl()}/v1/auth/refresh`, {
        refresh_token: refreshToken,
      });
      const accessToken = response.data?.access_token;

      if (!accessToken) {
        throw new Error('Missing access token');
      }

      setAccessToken(accessToken);
      if (response.data?.refresh_token) {
        setRefreshToken(response.data.refresh_token);
      }

      return accessToken;
    })().finally(() => {
      refreshTokenPromise = null;
    });
  }

  return refreshTokenPromise;
}

/** Request Interception */
requestInstance.interceptors.request.use(
  (config) => {
    if (config.noTimeout) {
      config.timeout = 0;
    }
    const token = getAccessToken();
    if (!token || token === NO_AUTH) {
      return config;
    }
    config.headers = config.headers || {};
    config.headers.Authorization = 'Bearer ' + token;
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/** Response Interception */
requestInstance.interceptors.response.use(
  (response) => {
    return response.data;
  },
  async (error) => {
    const response = error.response;
    if (!response) {
      eventBus.emit(RequestEvents.SERVER_ERROR, error.message || 'Network Error');

      return Promise.reject(error);
    }
    const status = response.status;
    const originalRequest = error.config as AxiosRequestConfig | undefined;

    if (shouldRefreshToken(status, originalRequest)) {
      try {
        const token = await refreshAccessToken();
        if (originalRequest) {
          originalRequest._retry = true;
          originalRequest.headers = originalRequest.headers || {};
          originalRequest.headers.Authorization = 'Bearer ' + token;
          return requestInstance(originalRequest);
        }
      } catch {
        // Keep the existing 401/403/default handling below when refresh fails.
      }
    }

    const rawDetail = response.data?.detail ?? response.data?.message ?? response.data?.msg ?? null;
    const errorMessage =
      (rawDetail !== null ? extractDetail(rawDetail) : '') || error.message || 'Unknown error';
    // The backend sends the full traceback of a failed launch in a separate
    // field so `detail` stays a plain string.
    const traceback =
      typeof response.data?.traceback === 'string' ? response.data.traceback : undefined;
    const payload: ServerErrorPayload = { detail: errorMessage, traceback };
    // Attach to the rejected error so per-call `.catch()` handlers can render
    // the cause inline instead of relying on the global toast.
    error.xinferenceError = payload;

    switch (status) {
      case 401: {
        /** trigger only once */
        if (requestManager.canHandle401()) {
          eventBus.emit(RequestEvents.UNAUTHORIZED, errorMessage);
        }
        break;
      }
      case 403: {
        /** trigger only once */
        if (requestManager.canHandle403()) {
          eventBus.emit(RequestEvents.FORBIDDEN, errorMessage);
        }
        break;
      }
      default: {
        eventBus.emit(RequestEvents.SERVER_ERROR, `${status}: ${errorMessage}`, payload);
      }
    }
    return Promise.reject(error);
  }
);

const request = {
  get<T = LooseResponse>(url: string, config?: AxiosRequestConfig) {
    return requestInstance.get<LooseResponse, T>(url, config);
  },

  post<T = LooseResponse>(url: string, data?: LooseResponse, config?: AxiosRequestConfig) {
    return requestInstance.post<LooseResponse, T, LooseResponse>(url, data, config);
  },

  put<T = LooseResponse>(url: string, data?: LooseResponse, config?: AxiosRequestConfig) {
    return requestInstance.put<LooseResponse, T, LooseResponse>(url, data, config);
  },

  patch<T = LooseResponse>(url: string, data?: LooseResponse, config?: AxiosRequestConfig) {
    return requestInstance.patch<LooseResponse, T, LooseResponse>(url, data, config);
  },

  delete<T = LooseResponse>(url: string, config?: AxiosRequestConfig) {
    return requestInstance.delete<LooseResponse, T>(url, config);
  },
};
export default request;
