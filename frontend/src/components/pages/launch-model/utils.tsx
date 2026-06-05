import type { ReactElement } from 'react';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { RadioGroup } from '@/components/ui/radio-group';
import { Select } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { ModelType, XINFERENCE_IO } from '@/constants';
import { ALL_FORM_KEYS, LAUNCH_MODEL_ROUTE_TABS } from '@/constants/launch';
import type { FormInstance, FormValues } from '@/types/form';
import { transformFormListToObj, transformObjToFormList } from '@/lib/utils';
import type { ModelEngine, ModelEngineItem, ReplicaItem } from '@/types/services';
import CommonFormList from './common-form-list';
import type {
  CatalogModel,
  EngineIndex,
  FormatIndex,
  FormLaunchFieldBase,
  LaunchFieldConfig,
  ModelSpec,
  RequestModelType,
  RouteModelType,
  UnknownRecord,
} from './types';

export const MODEL_ENGINE_TYPES: RequestModelType[] = [
  ModelType.LLM,
  ModelType.Embedding,
  ModelType.Rerank,
  ModelType.Image,
];

export function normalizeModelSize(value: unknown) {
  return toOptionValue(value).replace('.', '_');
}

export function createCacheKey(...parts: unknown[]) {
  return parts.map(toOptionValue).join('\u0000');
}

export function getItemQuantizations(item: ModelEngineItem) {
  const quantizations = Array.isArray(item.quantizations)
    ? item.quantizations
    : [item.quantization];

  return quantizations.map(toOptionValue).filter(Boolean);
}

export function buildEngineIndex(modelEngineMap: ModelEngine): EngineIndex {
  const index: EngineIndex = new Map();

  Object.entries(modelEngineMap).forEach(([engine, engineItems]) => {
    if (!Array.isArray(engineItems)) return;

    const formats = new Map<string, FormatIndex>();

    engineItems.forEach((item) => {
      const format = toOptionValue(item.model_format);

      if (!format) return;

      const formatIndex = formats.get(format) || {
        quantizations: new Set<string>(),
        sizes: new Map(),
      };
      const size = toOptionValue(item.model_size_in_billions);
      const quantizations = getItemQuantizations(item);

      quantizations.forEach((quantization) => formatIndex.quantizations.add(quantization));

      if (size) {
        const sizeIndex = formatIndex.sizes.get(size) || {
          multimodalProjectors: new Set<string>(),
          quantizations: new Set<string>(),
          value: item.model_size_in_billions,
        };

        quantizations.forEach((quantization) => sizeIndex.quantizations.add(quantization));
        (item.multimodal_projectors || []).forEach((projector) =>
          sizeIndex.multimodalProjectors.add(projector)
        );
        formatIndex.sizes.set(size, sizeIndex);
      }

      formats.set(format, formatIndex);
    });

    index.set(engine, formats);
  });

  return index;
}

export function syncLinkedField<T>(
  form: FormInstance,
  name: string,
  currentValue: T | undefined,
  options: Array<{ value: T }>,
  selectFirst = false
) {
  const values = options.map((option) => option.value);
  const nextValue =
    currentValue !== undefined && values.includes(currentValue)
      ? currentValue
      : values.length === 1 || (selectFirst && values.length > 0)
        ? values[0]
        : undefined;

  if (currentValue !== nextValue) {
    form.setFieldValue(name, nextValue);
  }
}

export function renderFormField(field: FormLaunchFieldBase, child: ReactElement) {
  return (
    <FormField
      name={field.name}
      label={field.label}
      extra={field.extra}
      rules={field.rules}
      placeholder={field.placeholder}
      disabled={field.disabled}
      valuePropName={field.valuePropName}
      layout={field.layout}
      tooltip={field.tooltip}
      className={field.className}
      normalize={field.normalize}
    >
      {child}
    </FormField>
  );
}

export function renderLaunchField(field: LaunchFieldConfig) {
  switch (field.type) {
    case 'input':
      return renderFormField(field, <Input {...field.fieldProps} />);
    case 'select':
      return renderFormField(field, <Select {...field.fieldProps} />);
    case 'switch':
      return renderFormField(field, <Switch {...field.fieldProps} />);
    case 'radio-group':
      return renderFormField(field, <RadioGroup {...field.fieldProps} />);
    case 'form-list':
      return <CommonFormList {...field.fieldProps} />;
    case 'custom':
      return field.content;
  }
}

export function renderLaunchFields(fields: LaunchFieldConfig[]) {
  return fields
    .filter((field) => field.show !== false)
    .map((field) => (
      <div key={field.name} className={field.colSpan === 2 ? 'col-span-2' : undefined}>
        {renderLaunchField(field)}
      </div>
    ));
}

export function isCachedSpec(spec: unknown): boolean {
  if (!isRecord(spec)) {
    return false;
  }

  const cacheStatus = spec.cache_status;

  if (Array.isArray(cacheStatus)) {
    return cacheStatus.some(Boolean);
  }

  return cacheStatus === true;
}

export function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function toStringArray(value: unknown) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).filter(Boolean);
  }

  if (typeof value === 'string' && value) {
    return [value];
  }

  return [];
}

export function getString(record: UnknownRecord, keys: string[]) {
  const value = keys.map((key) => record[key]).find((item) => typeof item === 'string');

  return typeof value === 'string' ? value : '';
}

export function normalizeModels(response: unknown): CatalogModel[] {
  if (!Array.isArray(response)) {
    return [];
  }
  return response.map((model) => {
    const modelSpecs = (Array.isArray(model.model_specs) ? model.model_specs : []) as ModelSpec[];
    return {
      ...model,
      model_description: model?.model_description || '',
      abilities: toStringArray(model.model_ability),
      languages: toStringArray(model.model_lang ?? model.language),
      detailUrl: `${XINFERENCE_IO}/models/detail/${encodeURIComponent(model.model_name)}`,
      modelSpecs,
      featured: model.featured === true,
      cached: modelSpecs.some((spec) => isCachedSpec(spec)),
    };
  });
}

export function getLaunchModelEndpointType(type: RequestModelType) {
  return type === ModelType.LLM ? 'LLM' : type;
}

export function getLaunchModelRouteType(value: string): RouteModelType {
  const normalized = value.toLowerCase();
  const matched = LAUNCH_MODEL_ROUTE_TABS.find((item) => item.path === normalized);

  return matched?.key || ModelType.LLM;
}

export function getSortedModels(models: CatalogModel[], favorites: string[]) {
  return [...models].sort((a, b) => {
    const aPriority = Number(Boolean(a.featured)) * 2 + Number(favorites.includes(a.model_name));
    const bPriority = Number(Boolean(b.featured)) * 2 + Number(favorites.includes(b.model_name));

    return bPriority - aPriority;
  });
}

export function range(start: number, end: number) {
  return new Array(end - start + 1).fill(undefined).map((_, i) => i + start);
}

export function toOptionValue(value: unknown) {
  return value === undefined || value === null ? '' : String(value);
}

function getFormListObject(value: unknown, transformValue = true) {
  if (!Array.isArray(value) || value.length === 0) {
    return undefined;
  }

  const nextValue = transformFormListToObj(value, transformValue);

  return Object.keys(nextValue).length > 0 ? nextValue : undefined;
}

function restoreFormListObject(values: FormValues, field: string) {
  if (!(field in values)) return;
  const fieldValue = values[field];

  if (!isRecord(fieldValue)) return;

  const nextValue = transformObjToFormList(fieldValue);

  values[field] = nextValue;
}

function restoreNestedFormListObject(values: FormValues, parent: string, field: string) {
  const parentValue = values[parent];

  if (!isRecord(parentValue) || !(field in parentValue)) return;

  const fieldValue = parentValue[field];

  if (!isRecord(fieldValue)) return;

  const nextParentValue = { ...parentValue };

  values[parent] = nextParentValue;
  nextParentValue[field] = transformObjToFormList(fieldValue);
}

function transformOnlyValueFormListToArray(values: FormValues, field: string) {
  if (!(field in values)) return;

  if (!Array.isArray(values[field]) || values[field].length === 0) {
    delete values[field];
    return;
  }

  const fieldValue = values[field] as unknown[];
  const nextValue = fieldValue
    .map((item: unknown) => {
      if (isRecord(item) && 'value' in item) return item.value;

      return item;
    })
    .filter((item: unknown) => item !== undefined && item !== null && item !== '');

  if (nextValue.length > 0) {
    values[field] = nextValue;
  } else {
    delete values[field];
  }
}

function transformArrayToOnlyValueFormList(values: FormValues, field: string) {
  if (!Array.isArray(values[field])) return;

  values[field] = (values[field] as unknown[]).map((value: unknown) => ({ value }));
}

function restoreKwargsFormList(values: FormValues) {
  const kwargsValue = values.kwargs;
  const kwargs = isRecord(kwargsValue) ? transformObjToFormList(kwargsValue) : [];
  const extraKwargs: Record<string, any> = {};

  Object.keys(values)
    .filter((key) => key !== 'kwargs' && !ALL_FORM_KEYS.includes(key))
    .forEach((key) => {
      extraKwargs[key] = values[key];

      delete values[key];
    });

  kwargs.push(...transformObjToFormList(extraKwargs));

  if (kwargs.length > 0) {
    values.kwargs = kwargs;
  }
}

function applyFormListObject(values: FormValues, field: string, transformValue = true) {
  if (!(field in values)) return;

  const nextValue = getFormListObject(values[field], transformValue);

  if (nextValue) {
    values[field] = nextValue;
    return;
  }

  delete values[field];
}
function mergeFormListObject(values: FormValues, field: string) {
  if (!(field in values)) return;

  const nextValue = getFormListObject(values[field]);

  delete values[field];

  if (nextValue) {
    Object.assign(values, nextValue);
  }
}

function applyNestedFormListObject(values: FormValues, parent: string, field: string) {
  const parentValue = values[parent];

  if (!parentValue || typeof parentValue !== 'object' || Array.isArray(parentValue)) return;
  if (!(field in parentValue)) return;

  const nextParentValue = { ...parentValue };
  const nextValue = getFormListObject(nextParentValue[field]);

  values[parent] = nextParentValue;

  if (nextValue) {
    nextParentValue[field] = nextValue;
  } else {
    delete nextParentValue[field];
  }

  if (Object.keys(nextParentValue).length === 0) {
    delete values[parent];
  }
}

function deleteNestedEmptyArrayField(values: FormValues, parent: string, field: string) {
  const parentValue = values[parent];

  if (!parentValue || typeof parentValue !== 'object' || Array.isArray(parentValue)) return;
  if (!Array.isArray(parentValue[field]) || parentValue[field].length > 0) return;

  const nextParentValue = { ...parentValue };

  delete nextParentValue[field];

  if (Object.keys(nextParentValue).length > 0) {
    values[parent] = nextParentValue;
  } else {
    delete values[parent];
  }
}

export function transformFormToFetch(values: FormValues) {
  const nextValues = { ...values };

  applyFormListObject(nextValues, 'quantization_config');
  applyFormListObject(nextValues, 'envs', false);
  mergeFormListObject(nextValues, 'kwargs');
  applyNestedFormListObject(nextValues, 'peft_model_config', 'image_lora_load_kwargs');
  applyNestedFormListObject(nextValues, 'peft_model_config', 'image_lora_fuse_kwargs');
  transformOnlyValueFormListToArray(nextValues, 'virtual_env_packages');
  deleteNestedEmptyArrayField(nextValues, 'peft_model_config', 'lora_list');

  if (nextValues?.enable_virtual_env === 'unset') {
    delete nextValues.enable_virtual_env;
  }

  return nextValues;
}

export function transformFetchToForm(values: FormValues) {
  const nextValues = { ...values };

  restoreFormListObject(nextValues, 'quantization_config');
  restoreFormListObject(nextValues, 'envs');
  transformArrayToOnlyValueFormList(nextValues, 'virtual_env_packages');
  restoreKwargsFormList(nextValues);
  restoreNestedFormListObject(nextValues, 'peft_model_config', 'image_lora_load_kwargs');
  restoreNestedFormListObject(nextValues, 'peft_model_config', 'image_lora_fuse_kwargs');

  return nextValues;
}

export function normalizeProgress(value: unknown) {
  const numericValue = Number(value || 0);
  if (!Number.isFinite(numericValue)) return 0;

  const percentValue = numericValue <= 1 ? numericValue * 100 : numericValue;

  return Math.max(0, Math.min(100, percentValue));
}

export function normalizeReplicaStatuses(value: unknown): ReplicaItem[] {
  const statuses = Array.isArray(value)
    ? value
    : value && typeof value === 'object' && Array.isArray((value as { replicas?: unknown }).replicas)
      ? (value as { replicas: unknown[] }).replicas
      : [];

  return statuses.filter(isRecord) as unknown as ReplicaItem[];
}
