export type LogLevel = 'ERROR' | 'WARNING' | 'INFO' | 'DEBUG';
export type LogType = 'worker' | 'supervisor';
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
  message?: string;
  log_type?: string;
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
