'use client';

import { useEffect, useMemo } from 'react';
import { ALL_FORM_KEYS } from '@/constants/launch';
import type { FieldName, FormInstance, FormValues } from '@/types/form';
import type { ModelEngine, ModelEngineItem } from '@/types/services';
import {
  launchParameterRow,
  parseLaunchParameterRow,
  parseLaunchParameterRows,
} from '../typed-launch-params';

export function selectEngineDefaults(engines: ModelEngine, values: FormValues) {
  const entries = engines[values.model_engine];
  if (!Array.isArray(entries)) return undefined;
  const matching = entries.filter(
    (entry) =>
      (!values.model_format || entry.model_format === values.model_format) &&
      (values.model_size_in_billions == null ||
        String(entry.model_size_in_billions) === String(values.model_size_in_billions)) &&
      (!values.quantization ||
        (!entry.quantization && !entry.quantizations) ||
        entry.quantization === values.quantization ||
        entry.quantizations?.includes(values.quantization))
  );
  if (!matching.length) return undefined;
  const signature = JSON.stringify(matching[0].default_model_config || {});
  return matching.every((entry) => JSON.stringify(entry.default_model_config || {}) === signature)
    ? matching[0]
    : undefined;
}

// These select a checkpoint or deployment rather than a model-loading option.
const SELECTION_KEYS = new Set(
  ALL_FORM_KEYS.filter(
    (key) =>
      ![
        'cpu_offload',
        'quantization_config',
        'enable_thinking',
        'enable_mtp',
        'num_speculative_tokens',
        'runtime_precision',
      ].includes(key)
  )
);
const PRECISION_KEYS = new Set([
  'quantization_config',
  'torch_dtype',
  'dtype',
  'component_quantizations',
  'component_precisions',
  'pipeline_config',
]);
type Row = ReturnType<typeof launchParameterRow> & { model_default_key?: string };
type Owned = { target: 'field' | 'kwargs'; value: unknown };
const same = (left: unknown, right: unknown) => JSON.stringify(left) === JSON.stringify(right);

export function createModelDefaultsController(form: FormInstance) {
  const owned = new Map<string, Owned>();
  const protectedKeys = new Set<string>();
  let context: string | undefined;

  const reset = () => {
    owned.clear();
    protectedKeys.clear();
    context = undefined;
  };
  const protect = (key: string) => {
    protectedKeys.add(key);
    owned.delete(key);
  };
  const read = () => structuredClone(form.getFieldsValue());
  const rowsOf = (values: FormValues): Row[] => (Array.isArray(values.kwargs) ? values.kwargs : []);
  const unchanged = (values: FormValues, key: string, item: Owned) =>
    same(
      item.target === 'field' ? values[key] : rowsOf(values).find((row) => row.key === key),
      item.value
    );
  const removeOwned = (values: FormValues) => {
    for (const [key, item] of owned) {
      if (!unchanged(values, key, item)) {
        const present =
          item.target === 'field'
            ? values[key] !== undefined
            : rowsOf(values).some((row) => row.key === key);
        if (present) protectedKeys.add(key);
        continue;
      }
      if (item.target === 'field') values[key] = undefined;
      else values.kwargs = rowsOf(values).filter((row) => row.key !== key);
    }
    owned.clear();
  };
  const write = (previous: FormValues, values: FormValues) => {
    const patch = Object.fromEntries(
      Object.entries(values).filter(([key, value]) => !same(previous[key], value))
    );
    if (Object.keys(patch).length) form.setFieldsValue(patch);
  };
  const prepareSnapshot = () => {
    const previous = read(),
      values = structuredClone(previous);
    removeOwned(values);
    write(previous, values);
  };
  const withoutAutomatic = (source: FormValues) => {
    const values = structuredClone(source);
    for (const [key, item] of owned) {
      if (!unchanged(values, key, item)) continue;
      if (item.target === 'field') delete values[key];
      else values.kwargs = rowsOf(values).filter((row) => row.key !== key);
    }
    return values;
  };
  const snapshotApplied = (values: FormValues) => {
    for (const key of Object.keys(values)) if (key !== 'kwargs') protect(key);
    for (const row of rowsOf(values)) if (row.key) protect(row.key);
  };
  const userChanged = (name?: FieldName) => {
    if (name === undefined) {
      prepareSnapshot();
      return;
    }
    const path = Array.isArray(name) ? name : String(name).split('.');
    if (path[0] !== 'kwargs') {
      protect(String(path[0]));
      return;
    }
    const values = read(),
      rows = rowsOf(values);
    if (path.length > 1) {
      const row = rows[Number(path[1])];
      if (row?.key) protect(row.key);
      if (row?.model_default_key) protect(row.model_default_key);
    }
    for (const [key, item] of owned)
      if (item.target === 'kwargs' && !unchanged(values, key, item)) protect(key);
  };
  const sync = (scope: string, spec: ModelEngineItem, fields: ReadonlySet<string>) => {
    const previous = read(),
      values = structuredClone(previous);
    const signature = JSON.stringify([
      scope,
      spec.default_model_config,
      spec.default_runtime_precision,
      [...fields],
    ]);
    if (signature !== context) {
      removeOwned(values);
      context = signature;
    }
    for (const [key, item] of owned)
      if (!unchanged(values, key, item)) {
        const present =
          item.target === 'field'
            ? values[key] !== undefined
            : rowsOf(values).some((row) => row.key === key);
        if (present) protect(key);
        else owned.delete(key);
      }

    // Migrate an existing advanced override into its dedicated visible control.
    for (const key of fields) {
      if (SELECTION_KEYS.has(key)) continue;
      const row = rowsOf(values).find((item) => item.key === key);
      if (!row || owned.has(key)) continue;
      try {
        values[key] = parseLaunchParameterRow(row);
      } catch {
        continue;
      } // Preserve incomplete JSON until the user submits.
      values.kwargs = rowsOf(values).filter((item) => item !== row);
      protect(key);
    }
    if (fields.has('runtime_precision') && spec.default_runtime_precision) {
      const legacy =
        values.quantization_config ??
        rowsOf(values).find((row) => row.key === 'quantization_config')?.value;
      if (
        !values.runtime_precision &&
        typeof legacy === 'string' &&
        spec.runtime_precisions?.includes(legacy.toLowerCase())
      ) {
        values.runtime_precision = legacy.toLowerCase();
        protect('runtime_precision');
      }
      const customPrecision = [...PRECISION_KEYS].some(
        (key) =>
          (values[key] !== undefined && !owned.has(key)) ||
          rowsOf(values).some((row) => row.key === key && !owned.has(key))
      );
      if (
        !owned.has('runtime_precision') &&
        !protectedKeys.has('runtime_precision') &&
        !values.runtime_precision &&
        !customPrecision
      ) {
        values.runtime_precision = spec.default_runtime_precision;
        owned.set('runtime_precision', { target: 'field', value: values.runtime_precision });
      }
    }
    for (const [key, defaultValue] of Object.entries(spec.default_model_config || {})) {
      if (SELECTION_KEYS.has(key) || key === 'runtime_precision') continue;
      if (owned.has(key)) continue;
      if (fields.has(key)) {
        if (protectedKeys.has(key)) continue;
        if (values[key] !== undefined && !same(values[key], form.getInitialFieldValue(key))) {
          protect(key);
          continue;
        }
        values[key] = structuredClone(defaultValue);
        owned.set(key, { target: 'field', value: values[key] });
        continue;
      }
      let rows = rowsOf(values);
      const existingRow = rows.find((row) => row.key === key);
      if (existingRow) {
        // Imported CLI arguments have no JSON type information. The model's
        // declared structured default supplies it without changing the text.
        if (defaultValue !== null && typeof defaultValue === 'object') {
          values.kwargs = rows.map((row) =>
            row === existingRow ? { ...row, value_type: 'json', value_type_key: key } : row
          );
        }
        continue;
      }
      // History stores known form fields at the top level, even if this model
      // exposes that setting only through the additional-parameters editor.
      if (values[key] !== undefined) {
        let value = values[key];
        if (key === 'quantization_config' && Array.isArray(value)) {
          try {
            value = parseLaunchParameterRows(value);
          } catch {
            continue;
          }
        }
        rows = [...rows, launchParameterRow(key, value)];
        values[key] = undefined;
        values.kwargs = rows;
        protect(key);
      } else if (!protectedKeys.has(key)) {
        const row: Row = { ...launchParameterRow(key, defaultValue), model_default_key: key };
        values.kwargs = [...rows, row];
        owned.set(key, { target: 'kwargs', value: row });
      }
    }
    write(previous, values);
  };
  return { reset, userChanged, snapshotApplied, sync, withoutAutomatic };
}

export function useModelDefaults(form: FormInstance) {
  const controller = useMemo(() => createModelDefaultsController(form), [form]);
  useEffect(() => form.registerReset(controller.reset), [form, controller]);
  return controller;
}
