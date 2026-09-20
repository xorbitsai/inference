export type LogLevel = 'ERROR' | 'WARNING' | 'INFO' | 'DEBUG';
export type LogType = 'worker' | 'supervisor' | 'model_request';
export type FieldFilterOp = '+' | '-';

export interface FieldFilter {
  key: string;
  value: string;
  op: FieldFilterOp;
}

export interface TimeRangeValue {
  from: string;
  to: string;
}

export type LogRow = Record<string, unknown> & {
  '@timestamp'?: string;
  level?: string;
  node?: string;
  role?: string;
  message?: string;
  log_type?: string;
  request_id?: string;
  event_type?: string;
  api_protocol?: string;
  endpoint?: string;
  model_uid?: string;
  status_code?: number;
  elapsed_ms?: number;
  success?: boolean;
  error?: { type?: string; message?: string };
};

export interface LogsResponse {
  hits?: LogRow[];
  total?: number;
}

export interface LogNodesResponse {
  nodes?: string[];
  node_field?: string;
}

export interface LogContextResponse {
  older?: LogRow[];
  newer?: LogRow[];
  has_more_older?: boolean;
  has_more_newer?: boolean;
}

export interface CorrelatedLogsResponse {
  hits?: LogRow[];
  total?: number;
  truncated?: boolean;
  request_id?: string;
}

export interface ModelRequestBodyResponse {
  request_id?: string;
  request_body?: unknown;
  request_body_raw?: string;
  request_body_omitted?: { reason?: string; size_bytes?: number; max_bytes?: number };
}
