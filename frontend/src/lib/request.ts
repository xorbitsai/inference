import axios from 'axios';
import Cookies from 'js-cookie';
import type { AxiosRequestConfig } from 'axios';
import { RequestEvents, NO_AUTH } from '@/constants';
import { eventBus } from '@/lib/event-bus';
import { requestManager } from '@/lib/request-manager';
import { getApiUrl } from '@/lib/utils';

const requestInstance = axios.create({
  baseURL: getApiUrl(),
  timeout: 30000,
});
/** Request Interception */
requestInstance.interceptors.request.use(
  (config) => {
    const token = Cookies.get('token');
    if (token === NO_AUTH) {
      return config;
    }
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
    const errorMessage =
      response.data?.detail ||
      response.data?.message ||
      response.data?.msg ||
      error.message ||
      'Unknown error';
    console.log(response, 'response');

    switch (status) {
      case 401: {
        /** trigger only once */
        if (requestManager.canHandle401()) {
          eventBus.emit(RequestEvents.UNAUTHORIZED);
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
  get<T = any>(url: string, config?: AxiosRequestConfig) {
    return requestInstance.get<any, T>(url, config);
  },

  post<T = any>(url: string, data?: any, config?: AxiosRequestConfig) {
    return requestInstance.post<any, T>(url, data, config);
  },

  put<T = any>(url: string, data?: any, config?: AxiosRequestConfig) {
    return requestInstance.put<any, T>(url, data, config);
  },

  patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig) {
    return requestInstance.patch<any, T>(url, data, config);
  },

  delete<T = any>(url: string, config?: AxiosRequestConfig) {
    return requestInstance.delete<any, T>(url, config);
  },
  // upload<T = any>(
  //   url: string,
  //   file: File,
  //   config?: AxiosRequestConfig
  // ) {
  //   const formData = new FormData();

  //   formData.append('file', file);

  //   return requestInstance.post<any, T>(
  //     url,
  //     formData,
  //     {
  //       ...config,
  //       headers: {
  //         'Content-Type':
  //           'multipart/form-data',
  //       },
  //     }
  //   );
  // },
};
export default request;
