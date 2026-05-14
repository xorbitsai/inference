import axios from 'axios';
import type { AxiosRequestConfig } from 'axios';
import { RequestEvents } from '@/constants';
import { eventBus } from '@/lib/event-bus';
import { requestManager } from '@/lib/request-manager';
import { getApiUrl } from '@/lib/utils';
import { authStore } from "@/lib/auth-store";

const requestInstance = axios.create({
  baseURL: getApiUrl(),
  timeout: 30000,
});
/** Request Interception */
requestInstance.interceptors.request.use(
  (config) => {
    const auth = authStore.get();
    // cluster/auth 为false 时
    if (!auth) {
      return config;
    }
    // token
    if (auth.type === "token") {
      config.headers.Authorization = 'Bearer ' + auth.token;
    }
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
    const status = response.status;

    console.log(response, 'response')

    if (!response) {
      // const errorData = await response.json()
      eventBus.emit(
        RequestEvents.SERVER_ERROR,
        `Server error: ${status} - ${
          response.detail || 'Unknown error'
        }`
      );
      return Promise.reject(error);
    }
    switch (status) {
      case 401: {
        /** trigger only once */
        if (requestManager.canHandle401()) {
          eventBus.emit(
            RequestEvents.UNAUTHORIZED
          );
        }
        break;
      }
      case 403: {
        /** trigger only once */
        if (requestManager.canHandle403()) {
          eventBus.emit(
            RequestEvents.FORBIDDEN
          );
        }
        break;
      }
    }

    return Promise.reject(error);
  }
);


const request = {
  get<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ) {
    return requestInstance.get<any, T>(
      url,
      config
    );
  },

  post<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ) {
    return requestInstance.post<any, T>(
      url,
      data,
      config
    );
  },

  put<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ) {
    return requestInstance.put<any, T>(
      url,
      data,
      config
    );
  },

  patch<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ) {
    return requestInstance.patch<any, T>(
      url,
      data,
      config
    );
  },

  delete<T = any>(
    url: string,
    config?: AxiosRequestConfig
  ) {
    return requestInstance.delete<any, T>(
      url,
      config
    );
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