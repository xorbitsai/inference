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
    console.log(error, error.message, error.status, 'error');
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

    const errorMessage =
      response.data?.detail ||
      response.data?.message ||
      response.data?.msg ||
      error.message ||
      'Unknown error';
    console.log(status, response, 'response');

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
          eventBus.emit(RequestEvents.FORBIDDEN);
        }
        break;
      }
      default: {
        eventBus.emit(RequestEvents.SERVER_ERROR, `Server error: ${status} - ${errorMessage}`);
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
