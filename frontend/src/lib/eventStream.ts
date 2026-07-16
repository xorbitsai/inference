import { NO_AUTH } from '@/constants';
import { getAccessToken } from '@/lib/auth-storage';
import { getApiUrl } from '@/lib/utils';

export interface PostEventStreamFetcherOptions<T> {
  onData: (data: T) => void;
  onError: (msg: string) => void;
  onEnd?: () => void;
}

export interface PostEventStreamFetcherParams<T> {
  url: string; // Request URL
  data: any; // Request payload
  options: PostEventStreamFetcherOptions<T>; // Callback options
  headers?: Record<string, string>; // Optional request headers
}

export class EventStreamController {
  private abortController: AbortController;

  constructor() {
    this.abortController = new AbortController();
  }

  /**
   * Abort the current streaming request.
   */
  terminate() {
    this.abortController.abort();
  }

  /**
   * Get the current signal object.
   */
  getSignal() {
    return this.abortController.signal;
  }
}
const errorText = 'System error. Please try again.';

export function getErrorText(errorDetail: any) {
  if (!errorDetail) {
    return '';
  }
  let detailStr = typeof errorDetail === 'string' ? errorDetail : JSON.stringify(errorDetail);
  if (detailStr.length > 1000 && /data:.+?;base64/.test(detailStr)) {
    detailStr = detailStr.slice(0, 1000) + '...';
  }
  return detailStr;
}
export async function postEventStreamFetcher<T = any>(
  params: PostEventStreamFetcherParams<T>,
  controller: EventStreamController
): Promise<void> {
  const { url, data, options, headers } = params;
  const { onData, onError, onEnd } = options;
  const token = getAccessToken();

  const fullUrl = `${getApiUrl()}${url}`;

  try {
    const response = await fetch(fullUrl, {
      method: 'POST',
      body: JSON.stringify(data),
      headers: {
        'Content-Type': 'application/json',
        ...(token && token !== NO_AUTH ? { Authorization: 'Bearer ' + token } : {}),
        ...(headers || {}),
      },
      signal: controller.getSignal(), // Use the AbortController signal.
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
          // Check whether a complete data chunk is available.
          const chunk = buffer.slice(0, pos);
          buffer = buffer.slice(pos + 4);
          parseEventData<T>(chunk, onData, onError); // Process each complete data chunk.
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('被手动终止');
      } else {
        onError(err?.message || errorText);
      }
    } finally {
      // Release the reader.
      reader.releaseLock();
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      console.log('被手动终止');
    } else {
      // Network error.
      onError(err?.message || errorText);
    }
  } finally {
    onEnd?.();
  }
}

function parseEventData<T>(text: string, next: (data: T) => void, onError: (msg: string) => void) {
  // Split the received data segment.
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.startsWith('data:')) {
      let eventData = line.slice(5);
      if (eventData.startsWith(' ')) {
        eventData = eventData.slice(1);
      }
      if (eventData === '[DONE]') return;
      try {
        const json = JSON.parse(eventData);
        if (json?.error) {
          onError(json.error);
        } else {
          next(json); // Handle the parsed JSON data.
        }
      } catch (error) {
        console.error('Error parsing JSON:', error);
      }
    }
  }
}
