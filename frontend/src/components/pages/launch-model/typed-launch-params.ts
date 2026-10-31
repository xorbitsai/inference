import { transformValueType } from '@/lib/utils';
import {
  isStructuredLaunchParameter,
  parseStructuredLaunchParameter,
  StructuredLaunchParameterError,
} from './structured-launch-params';

export interface LaunchParameterRow {
  key: string;
  value: string;
  value_type?: 'json' | 'string';
  value_type_key?: string;
}

export function launchParameterRow(key: string, value: unknown): LaunchParameterRow {
  if (value === undefined) return { key, value: '' };
  if (typeof value === 'string') return { key, value, value_type: 'string', value_type_key: key };
  if (value === null) return { key, value: 'None' };
  if (typeof value !== 'object') return { key, value: String(value) };
  return { key, value: JSON.stringify(value), value_type: 'json', value_type_key: key };
}

export function parseLaunchParameterRows(rows: unknown) {
  if (!Array.isArray(rows)) return {};
  return Object.fromEntries(
    rows
      .filter((row) => row && typeof row.key === 'string' && row.key.trim())
      .map((row) => [row.key.trim(), parseLaunchParameterRow(row)])
  );
}

export function parseLaunchParameterRow(row: {
  key: string;
  value: unknown;
  value_type?: string;
  value_type_key?: string;
}) {
  const key = row.key.trim();
  if (isStructuredLaunchParameter(key)) return parseStructuredLaunchParameter(key, row.value);
  if (row.value_type_key === key) {
    if (row.value_type === 'string') return String(row.value);
    if (row.value_type === 'json') {
      if (typeof row.value !== 'string') return row.value;
      try {
        return JSON.parse(row.value);
      } catch {
        throw new StructuredLaunchParameterError(key, 'invalidJson');
      }
    }
  }
  return transformValueType(row.value);
}
