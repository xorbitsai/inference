'use client';

import type { ComponentProps } from 'react';
import { FormField } from '@/components/ui/form-field';
import { Select } from '@/components/ui/select';
import { useI18n } from '@/contexts/i18n-context';
import { useFormValues, useWatch } from '@/hooks/use-form';
import type { FormInstance } from '@/types/form';
import type { ModelEngine } from '@/types/services';

const PRECISION_KEYS = new Set([
  'quantization_config',
  'dtype',
  'torch_dtype',
  'component_quantizations',
  'component_precisions',
  'pipeline_config',
]);

function PrecisionSelect(props: ComponentProps<typeof Select>) {
  return <Select {...props} value={props.value || 'auto'} allowClear={false} />;
}

export default function RuntimePrecisionField({
  form,
  engines,
}: {
  form: FormInstance;
  engines: ModelEngine;
}) {
  const { t } = useI18n();
  const values = useFormValues(form);
  const engine = useWatch('model_engine', form);
  const format = useWatch('model_format', form);
  const precision = useWatch('runtime_precision', form);
  const specs = engines[engine];
  const spec = Array.isArray(specs)
    ? specs.find((item) => !format || item.model_format === format)
    : undefined;
  const supported = spec?.runtime_precisions || [];
  const valid = !precision || precision === 'auto' || supported.includes(precision);
  const custom =
    [...PRECISION_KEYS].some((key) => values[key] != null) ||
    (Array.isArray(values.kwargs) &&
      values.kwargs.some((row: { key?: string }) => PRECISION_KEYS.has(row.key || '')));
  const defaultLabel = custom
    ? t('launchModel.runtimePrecisionCustom')
    : t('launchModel.runtimePrecisionDefault', {
        precision: spec?.default_runtime_precision?.toUpperCase() || '',
      });
  const unsupported = t('launchModel.runtimePrecisionUnsupported', {
    precision: String(precision).toUpperCase(),
    engine,
  });
  const options = [
    { value: 'auto', label: defaultLabel },
    ...supported.map((value) => ({
      value,
      label: value === 'bf16' ? t('launchModel.runtimePrecisionBf16') : value.toUpperCase(),
    })),
    ...(!valid ? [{ value: precision, label: unsupported, disabled: true }] : []),
  ];
  return (
    <FormField
      name="runtime_precision"
      label={t('launchModel.runtimePrecision')}
      extra={
        <span className={valid ? undefined : 'text-destructive'}>
          {valid ? t('launchModel.runtimePrecisionHelp') : unsupported}
        </span>
      }
      rules={[
        {
          validator: (value) => !value || value === 'auto' || supported.includes(value),
          message: unsupported,
        },
      ]}
    >
      <PrecisionSelect options={options} disabled={!spec} />
    </FormField>
  );
}
