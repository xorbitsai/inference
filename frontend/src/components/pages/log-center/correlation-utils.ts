import type { LogRow } from './types';

const MAX_REQUEST_ID_LENGTH = 256;
const REQUEST_MESSAGE_PATTERN = /\[request ([^\]\r\n]{1,256})\]/;

const normalizeCorrelationId = (value: unknown) => {
  if (typeof value !== 'string') return '';

  const normalized = value.trim();
  if (!normalized || normalized.length > MAX_REQUEST_ID_LENGTH) return '';
  if (
    [...normalized].some((character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      return codePoint < 32 || codePoint === 127;
    })
  ) {
    return '';
  }
  return normalized;
};

export const getCorrelationId = (row: LogRow) => {
  const requestId = normalizeCorrelationId(row.request_id);
  if (requestId) return requestId;

  const correlationId = normalizeCorrelationId(row.correlation_id);
  if (correlationId) return correlationId;

  if (typeof row.message !== 'string') return '';
  const match = row.message.match(REQUEST_MESSAGE_PATTERN);
  return normalizeCorrelationId(match?.[1]);
};
