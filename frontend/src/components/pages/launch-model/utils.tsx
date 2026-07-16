import type { ReactElement } from 'react';
import { FormField } from '@/components/ui/form-field';
import { Input } from '@/components/ui/input';
import { RadioGroup } from '@/components/ui/radio-group';
import { Select } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { ModelType, XINFERENCE_IO, CUSTOM_MODEL_OPTIONS } from '@/constants';
import { ALL_FORM_KEYS, LAUNCH_MODEL_ROUTE_TABS } from '@/constants/launch';
import type { FormInstance, FormValues } from '@/types/form';
import { transformFormListToObj, transformObjToFormList } from '@/lib/utils';
import type { ModelEngine, ModelEngineItem, ReplicaItem } from '@/types/services';
import CommonFormList from './launch-dialog/common-form-list';
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
export function getInitialCustomType(activeType?: string) {
  const matched = CUSTOM_MODEL_OPTIONS.find((item) => item.value === activeType);

  return (matched?.value || ModelType.LLM) as RequestModelType;
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
  const extraKwargs: Record<string, unknown> = {};

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

function normalizeNGPU (value?: string | number) {
  if (!value) return null

  if (value === 'CPU') return null
  if (value === 'auto' || value === 'GPU') return 'auto'

  return value === 0 ? null : value
}
export const parseGpuIndexes = (value?: string): number[] | undefined => {
  if (!value) return undefined;

  const result = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .map(Number)
    .filter((num) => !Number.isNaN(num));

  return result.length ? result : undefined;
};

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
  if('n_gpu' in values){
    nextValues.n_gpu = normalizeNGPU(values.n_gpu)
  }
  if('gpu_idx' in values) {
    nextValues.gpu_idx = parseGpuIndexes(values.gpu_idx);
  }
  if ('n_gpu_layers' in values && values.n_gpu_layers < 0) {
    delete nextValues.n_gpu_layers
  }
  if (nextValues.download_hub === 'none') {
    delete nextValues.download_hub
  }
  if (nextValues.gguf_quantization === 'none') {
    delete nextValues.gguf_quantization
  }
  return nextValues;
}
function restoreNGPU (value: null | string | number, modelType: RequestModelType) {
  if (value === null) return 'CPU'
  if (value === 'auto') {
    return [ModelType.LLM, ModelType.Image].includes(modelType) ? 'auto' : 'GPU'
  }
  if (typeof value === 'number') return value
  return value || 'CPU'
}
export function transformFetchToForm(values: FormValues) {
  const nextValues = { ...values };

  restoreFormListObject(nextValues, 'quantization_config');
  restoreFormListObject(nextValues, 'envs');
  transformArrayToOnlyValueFormList(nextValues, 'virtual_env_packages');
  restoreKwargsFormList(nextValues);
  restoreNestedFormListObject(nextValues, 'peft_model_config', 'image_lora_load_kwargs');
  restoreNestedFormListObject(nextValues, 'peft_model_config', 'image_lora_fuse_kwargs');
  if('n_gpu' in values){
    nextValues.n_gpu = restoreNGPU(values.n_gpu, values?.model_type)
  }
  if('gpu_idx' in values){
    nextValues.gpu_idx = Array.isArray(values.gpu_idx) ? values.gpu_idx.join(',') : undefined;
  }
  return nextValues;
}

const commandLineKeyMap: Record<string, string> = {
  model_size_in_billions: '--size-in-billions',
  download_hub: '--download_hub',
  enable_thinking: '--enable-thinking',
  reasoning_content: '--reasoning_content',
  lightning_version: '--lightning_version',
  lightning_model_path: '--lightning_model_path',
};

const predefinedLaunchKeySet = new Set<string>(ALL_FORM_KEYS);

function isEmptyCommandValue(value: unknown) {
  if (value === null || value === undefined) return true;
  if (typeof value === 'string') return value.trim() === '';
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value).length === 0;

  return false;
}

function stringifyCommandValue(value: unknown) {
  if (value === null) return 'none';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'object') return JSON.stringify(value);

  return String(value);
}

function quoteCommandValue(value: unknown) {
  const stringValue = stringifyCommandValue(value);

  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(stringValue)) {
    return stringValue;
  }

  return `'${stringValue.replace(/'/g, `'\\''`)}'`;
}

function getCommandKey(key: string) {
  if (predefinedLaunchKeySet.has(key)) {
    return commandLineKeyMap[key] ?? `--${key.replace(/_/g, '-')}`;
  }

  return `--${key}`;
}

const commandLeadingKeys = ['model_name', 'model_type'];

const booleanCommandKeys = new Set(['enable_thinking', 'reasoning_content', 'cpu_offload']);

const commandLineKeyMapEntries = Object.entries(commandLineKeyMap);

function tokenizeCommand(command: string) {
  const tokens: string[] = [];
  let current = '';
  let quote: '"' | "'" | null = null;
  let escaping = false;

  for (const char of command.trim()) {
    if (escaping) {
      current += char;
      escaping = false;
      continue;
    }

    if (char === '\\' && quote !== "'") {
      escaping = true;
      continue;
    }

    if (quote) {
      if (char === quote) {
        quote = null;
      } else {
        current += char;
      }
      continue;
    }

    if (char === '"' || char === "'") {
      quote = char;
      continue;
    }

    if (/\s/.test(char)) {
      if (current) {
        tokens.push(current);
        current = '';
      }
      continue;
    }

    current += char;
  }

  if (current) {
    tokens.push(current);
  }

  return tokens;
}

function normalizeCommandKey(key: string) {
  const commandKey = `--${key}`;
  const mappedEntry = commandLineKeyMapEntries.find(([, value]) => value === commandKey);

  if (mappedEntry) {
    return mappedEntry[0];
  }

  return key.replace(/-/g, '_');
}

function readBooleanCommandValue(value: string | undefined) {
  if (value === undefined || value === '') return true;

  return value === 'true';
}

function setObjectPair(target: Record<string, unknown>, values: string[]) {
  const [key, value] = values;

  if (key && value !== undefined) {
    target[key] = value;
  }
}

export function generateCommandLineStatement(params: FormValues) {
  const entries = Object.entries(params).filter(([, value]) => !isEmptyCommandValue(value));

  const args = [
    ...commandLeadingKeys.flatMap((leadingKey) =>
      entries.filter(([key]) => key === leadingKey)
    ),
    ...entries.filter(([key]) => !commandLeadingKeys.includes(key)),
  ]
    .flatMap(([key, value]) => {
      if (key === 'gpu_idx' && Array.isArray(value)) {
        return `--gpu-idx ${quoteCommandValue(value.join(','))}`;
      }

      if (key === 'peft_model_config' && typeof value === 'object' && value !== null) {
        const config = value as {
          lora_list?: Array<{ lora_name?: unknown; local_path?: unknown }>;
          image_lora_load_kwargs?: Record<string, unknown>;
          image_lora_fuse_kwargs?: Record<string, unknown>;
        };
        const peftArgs = [];

        if (Array.isArray(config.lora_list)) {
          peftArgs.push(
            ...config.lora_list
              .filter(
                (lora) =>
                  !isEmptyCommandValue(lora.lora_name) && !isEmptyCommandValue(lora.local_path)
              )
              .map(
                (lora) =>
                  `--lora-modules ${quoteCommandValue(lora.lora_name)} ${quoteCommandValue(lora.local_path)}`
              )
          );
        }

        if (config.image_lora_load_kwargs) {
          peftArgs.push(
            ...Object.entries(config.image_lora_load_kwargs)
              .filter(([, v]) => !isEmptyCommandValue(v))
              .map(
                ([k, v]) =>
                  `--image-lora-load-kwargs ${quoteCommandValue(k)} ${quoteCommandValue(v)}`
              )
          );
        }

        if (config.image_lora_fuse_kwargs) {
          peftArgs.push(
            ...Object.entries(config.image_lora_fuse_kwargs)
              .filter(([, v]) => !isEmptyCommandValue(v))
              .map(
                ([k, v]) =>
                  `--image-lora-fuse-kwargs ${quoteCommandValue(k)} ${quoteCommandValue(v)}`
              )
          );
        }

        return peftArgs;
      }

      if (key === 'quantization_config' && typeof value === 'object' && value !== null) {
        return Object.entries(value)
          .filter(([, v]) => v !== undefined && v !== '')
          .map(([k, v]) => `--quantization-config ${quoteCommandValue(k)} ${quoteCommandValue(v)}`);
      }

      if (key === 'envs' && typeof value === 'object' && value !== null) {
        return Object.entries(value)
          .filter(([, v]) => !isEmptyCommandValue(v))
          .map(([k, v]) => `--env ${quoteCommandValue(k)} ${quoteCommandValue(v)}`);
      }

      if (key === 'virtual_env_packages' && Array.isArray(value)) {
        return value
          .filter((pkg) => !isEmptyCommandValue(pkg))
          .map((pkg) => `--virtual-env-package ${quoteCommandValue(pkg)}`);
      }

      if (key === 'enable_virtual_env') {
        if (value === true) return '--enable-virtual-env';
        if (value === false) return '--disable-virtual-env';

        return [];
      }

      if (key === 'enable_thinking') {
        if (value === true) return '--enable-thinking';

        return [];
      }

      return `${getCommandKey(key)} ${quoteCommandValue(value)}`;
    })
    .join(' ');

  return args ? `xinference launch ${args}` : 'xinference launch';
}

export function parseXinferenceCommand(command: string) {
  const tokens = tokenizeCommand(command);
  const params: FormValues = {};
  const peftModelConfig: {
    lora_list: Array<{ lora_name: string; local_path: string }>;
    image_lora_load_kwargs: Record<string, unknown>;
    image_lora_fuse_kwargs: Record<string, unknown>;
  } = {
    lora_list: [],
    image_lora_load_kwargs: {},
    image_lora_fuse_kwargs: {},
  };
  const quantizationConfig: Record<string, unknown> = {};
  const virtualEnvPackages: string[] = [];
  const envs: Record<string, unknown> = {};
  const args =
    tokens[0] === 'xinference' && tokens[1] === 'launch'
      ? tokens.slice(2)
      : tokens[0] === 'launch'
        ? tokens.slice(1)
        : tokens;

  for (let index = 0; index < args.length; index += 1) {
    const flag = args[index];

    if (!flag.startsWith('--')) {
      continue;
    }

    const valueTokens: string[] = [];

    while (index + 1 < args.length && !args[index + 1].startsWith('--')) {
      valueTokens.push(args[index + 1]);
      index += 1;
    }

    const key = flag.slice(2);
    const normalizedKey = normalizeCommandKey(key);
    const value = valueTokens.join(' ');

    if (normalizedKey === 'gpu_idx') {
      params.gpu_idx = value
        .split(',')
        .map((item) => Number(item.trim()))
        .filter((item) => Number.isFinite(item));
      continue;
    }

    if (normalizedKey === 'lora_modules') {
      for (let i = 0; i < valueTokens.length; i += 2) {
        const loraName = valueTokens[i];
        const localPath = valueTokens[i + 1];

        if (loraName && localPath) {
          peftModelConfig.lora_list.push({
            lora_name: loraName,
            local_path: localPath,
          });
        }
      }
      continue;
    }

    if (normalizedKey === 'image_lora_load_kwargs') {
      setObjectPair(peftModelConfig.image_lora_load_kwargs, valueTokens);
      continue;
    }

    if (normalizedKey === 'image_lora_fuse_kwargs') {
      setObjectPair(peftModelConfig.image_lora_fuse_kwargs, valueTokens);
      continue;
    }

    if (normalizedKey === 'quantization_config') {
      setObjectPair(quantizationConfig, valueTokens);
      continue;
    }

    if (normalizedKey === 'enable_virtual_env') {
      params.enable_virtual_env = true;
      continue;
    }

    if (normalizedKey === 'disable_virtual_env') {
      params.enable_virtual_env = false;
      continue;
    }

    if (normalizedKey === 'virtual_env_package') {
      if (value) {
        virtualEnvPackages.push(value);
      }
      continue;
    }

    if (normalizedKey === 'env') {
      setObjectPair(envs, valueTokens);
      continue;
    }

    if (booleanCommandKeys.has(normalizedKey)) {
      params[normalizedKey] = readBooleanCommandValue(valueTokens[0]);
      continue;
    }

    params[normalizedKey] = valueTokens.length === 0 ? true : value;
  }

  if (
    peftModelConfig.lora_list.length > 0 ||
    Object.keys(peftModelConfig.image_lora_load_kwargs).length > 0 ||
    Object.keys(peftModelConfig.image_lora_fuse_kwargs).length > 0
  ) {
    params.peft_model_config = peftModelConfig;
  }

  if (Object.keys(quantizationConfig).length > 0) {
    params.quantization_config = quantizationConfig;
  }

  if (virtualEnvPackages.length > 0) {
    params.virtual_env_packages = virtualEnvPackages;
  }

  if (Object.keys(envs).length > 0) {
    params.envs = envs;
  }

  return params;
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
    : value &&
        typeof value === 'object' &&
        Array.isArray((value as { replicas?: unknown }).replicas)
      ? (value as { replicas: unknown[] }).replicas
      : [];

  return statuses.filter(isRecord) as unknown as ReplicaItem[];
}

export function isEmptyLaunchValue(value: unknown) {
  if (value === undefined || value === null) return true;
  if (typeof value === 'string') return value.trim() === '';
  if (Array.isArray(value)) return value.length === 0;

  return false;
}

export function isVisibleRequiredLaunchField(field: LaunchFieldConfig) {
  return field.show !== false && 'rules' in field && field.rules?.some((rule) => rule.required);
}
