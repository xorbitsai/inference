import Cookies from 'js-cookie';
import { NO_AUTH } from '@/constants';
import { getApiUrl } from '@/lib/utils';

export interface PostEventStreamFetcherOptions<T> {
  onData: (data: T) => void;
  onError: (msg: string) => void;
  onEnd?: () => void;
}

export interface PostEventStreamFetcherParams<T> {
  url: string; // 请求的 URL
  data: any; // 请求的参数
  options: PostEventStreamFetcherOptions<T>; // 回调选项
  headers?: Record<string, string>; // 可选的请求头
}

export class EventStreamController {
  private abortController: AbortController;

  constructor() {
    this.abortController = new AbortController();
  }

  /**
   * 终止当前的流式请求
   */
  terminate() {
    this.abortController.abort();
  }

  /**
   * 获取当前的信号对象
   */
  getSignal() {
    return this.abortController.signal;
  }
}
const errorText = 'System error. Please try again.';

export function getErrorText(errorDetail: string) {
  // 若是因为base64报错，提取前1000字符，防止报错信息因含有base64信息，导致报错文案过长
  if (errorDetail && errorDetail.length > 1000 && /data:.+?;base64/.test(errorDetail)) {
    errorDetail = `${errorDetail.slice(0, 1000)}...`;
  }
  return errorDetail;
}
export async function postEventStreamFetcher<T = any>(
  params: PostEventStreamFetcherParams<T>,
  controller: EventStreamController
): Promise<void> {
  const { url, data, options, headers } = params;
  const { onData, onError, onEnd } = options;
  const token = Cookies.get('token');

  const fullUrl = `${getApiUrl()}${url}`;

  try {
    const response = await fetch(fullUrl, {
      method: 'POST',
      body: JSON.stringify(data),
      headers: {
        'Content-Type': 'application/json',
        ...(token !== NO_AUTH ? { Authorization: 'Bearer ' + token } : {}),
        ...(headers || {}),
      },
      signal: controller.getSignal(), // 使用 AbortController 的信号
    });

    if (!response.ok) {
      const error = await response.clone().json();
      onError(getErrorText(error?.detail || errorText));
      return;
    }
    if (!data?.stream) {
      const result = await response.clone().json();
      if (result?.id) {
        onData(result);
      } else {
        onError(errorText);
      }
      return;
    }
    const reader = response.body?.getReader();
    if (!reader) {
      onError('Failed to get reader from response body');
      return;
    }
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let pos;
        while ((pos = buffer.indexOf('\r\n\r\n')) >= 0) {
          // 检查是否有完整的数据块
          const chunk = buffer.slice(0, pos);
          buffer = buffer.slice(pos + 2);
          parseEventData<T>(chunk, onData, onError); // 处理每个完整的数据块
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('被手动终止');
      } else {
        onError(err?.message || errorText);
      }
    } finally {
      // 释放 reader
      reader.releaseLock();
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      console.log('被手动终止');
    } else {
      // 网络错误
      onError(err?.message || errorText);
    }
  } finally {
    onEnd?.();
  }
}

function parseEventData<T>(text: string, next: (data: T) => void, onError: (msg: string) => void) {
  // 切割接收到的数据段
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.startsWith('data:')) {
      const eventData = line.replace('data: ', '');
      if (eventData === '[DONE]') return;
      try {
        const json = JSON.parse(eventData);
        if (json?.error) {
          onError(json.error);
        } else {
          next(json); // 处理转换后的JSON数据
        }
      } catch (error) {
        console.error('Error parsing JSON:', error);
      }
    }
  }
}
