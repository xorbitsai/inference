'use client';

import { useState } from 'react';
import { useI18n } from '@/contexts/i18n-context';
import { useFormValues } from '@/hooks/use-form';
import type { FormInstance, FormValues } from '@/types/form';
import type { ModelEngine, ModelEngineItem } from '@/types/services';
import { transformFormToFetch } from '../utils';

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
  // Do not guess between variants whose default configurations differ.
  const config = JSON.stringify(matching[0].default_model_config || {});
  return matching.every((entry) => JSON.stringify(entry.default_model_config || {}) === config)
    ? matching[0]
    : undefined;
}

function formatValue(value: unknown) {
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

export default function ModelDefaultConfig({
  form,
  spec,
}: {
  form: FormInstance;
  spec: ModelEngineItem;
}) {
  const { t } = useI18n();
  const values = useFormValues(form);
  const [expanded, setExpanded] = useState(false);
  const entries = Object.entries(spec.default_model_config || {});
  let payload: FormValues = {};
  try {
    payload = transformFormToFetch(values);
  } catch {
    // Additional parameters may contain incomplete JSON while being edited.
  }
  const precisionSelected = spec.runtime_precisions?.includes(payload.runtime_precision) === true;
  if (!entries.length) return null;
  return (
    <section
      aria-label={t('launchModel.modelDefaultConfig')}
      className="rounded-md border bg-muted/20 p-3 space-y-2"
    >
      <h4 className="text-sm font-medium">{t('launchModel.modelDefaultConfig')}</h4>
      <p className="text-xs text-muted-foreground">{t('launchModel.modelDefaultConfigHelp')}</p>
      <dl className="divide-y divide-border/60 text-xs">
        {(expanded ? entries : entries.slice(0, 3)).map(([key, value]) => {
          // quantization is also a checkpoint selector in the launch API. Its
          // value does not override a runtime default with the same name.
          const runtimeQuantization = key === 'quantization' && !!spec.runtime_precisions?.length;
          const overrideKey = runtimeQuantization ? 'quantization_config' : key;
          const hasOverride =
            (key !== 'quantization' || runtimeQuantization) && Object.hasOwn(payload, overrideKey);
          const precisionOverride =
            precisionSelected &&
            [
              'quantization',
              'quantization_config',
              'torch_dtype',
              'dtype',
              'component_quantizations',
              'component_precisions',
              'pipeline_config',
            ].includes(key);
          return (
            <div key={key} className="grid grid-cols-[minmax(0,1fr)_minmax(0,2fr)] gap-x-3 py-2">
              <dt className="break-words font-mono text-muted-foreground">{key}</dt>
              <dd className="min-w-0 space-y-1">
                <pre className="whitespace-pre-wrap break-words font-mono">
                  {formatValue(value)}
                </pre>
                {precisionOverride ? (
                  <p className="text-primary">{t('launchModel.modelDefaultPrecisionOverride')}</p>
                ) : hasOverride ? (
                  <div className="text-primary">
                    <span>{t('launchModel.modelDefaultCurrentValue')}</span>
                    <pre className="whitespace-pre-wrap break-words font-mono">
                      {formatValue(payload[overrideKey])}
                    </pre>
                  </div>
                ) : null}
              </dd>
            </div>
          );
        })}
      </dl>
      {entries.length > 3 && (
        <button
          type="button"
          className="text-xs text-primary hover:underline"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded
            ? t('launchModel.modelDefaultCollapse')
            : t('launchModel.modelDefaultExpand', { count: entries.length })}
        </button>
      )}
    </section>
  );
}
