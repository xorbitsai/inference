import type { FormValues } from '@/types/form';
import type { ModelEngine } from '@/types/services';
import { buildEngineIndex, GPU_IDX_PATTERN, transformFormToFetch } from '../utils';

export interface RecommendationResponse {
  status: 'recommended' | 'no_recommendation';
  config: null | {
    model_engine: string;
    model_format: string;
    model_size_in_billions: string | number;
    quantization: string;
    worker_ip: string | null;
    enable_virtual_env: boolean;
  };
  reasons: { code: string; message: string }[];
  warnings: { code: string; message: string }[];
}

export function recommendationUnavailable(values: FormValues) {
  const workers = Array.isArray(values.worker_ip)
    ? values.worker_ip
    : String(values.worker_ip || '')
        .split(',')
        .filter(Boolean);
  return Boolean(
    values.model_path ||
    values.kwargs?.some((entry: { key?: string }) => Boolean(entry?.key?.trim())) ||
    values.replica_placement_mode === 'custom' ||
    values.replica_config?.length ||
    Number(values.replica || 1) !== 1 ||
    Number(values.n_worker || 1) !== 1 ||
    workers.length > 1
  );
}

export function recommendationRequest(modelName: string, values: FormValues) {
  if (recommendationUnavailable(values)) throw new Error('unsupported');
  if (
    values.gpu_idx !== undefined &&
    values.gpu_idx !== '' &&
    !GPU_IDX_PATTERN.test(String(values.gpu_idx))
  )
    throw new Error('gpu_idx');
  // Only convert the supported constraints: kwargs must not override these fields.
  const selected = Object.fromEntries(
    ['model_size_in_billions', 'worker_ip', 'enable_virtual_env', 'n_gpu', 'gpu_idx']
      .filter((key) => values[key] !== undefined && values[key] !== '')
      .map((key) => [key, values[key]])
  );
  if (selected.n_gpu === 0 || selected.n_gpu === '0') throw new Error('n_gpu');
  return { model_name: modelName, model_type: 'LLM', constraints: transformFormToFetch(selected) };
}

export function recommendationPatch(
  response: RecommendationResponse,
  values: FormValues,
  engines: ModelEngine
) {
  const config = response.config;
  if (response.status !== 'recommended' || !config) return null;
  const size = buildEngineIndex(engines)
    .get(config.model_engine)
    ?.get(config.model_format)
    ?.sizes.get(String(config.model_size_in_billions));
  if (!size?.quantizations.has(config.quantization)) throw new Error('invalid recommendation');
  const constraints = recommendationRequest('', values).constraints;
  if (
    typeof config.enable_virtual_env !== 'boolean' ||
    (typeof constraints.enable_virtual_env === 'boolean' &&
      constraints.enable_virtual_env !== config.enable_virtual_env)
  ) {
    throw new Error('environment constraint mismatch');
  }
  if (
    constraints.model_size_in_billions != null &&
    String(constraints.model_size_in_billions) !== String(config.model_size_in_billions)
  ) {
    throw new Error('size constraint mismatch');
  }
  return {
    model_engine: config.model_engine,
    model_format: config.model_format,
    // Use the catalog's value type so linked selects retain the selection.
    model_size_in_billions: size.value,
    quantization: config.quantization,
    ...(config.worker_ip ? { worker_ip: [config.worker_ip] } : {}),
    ...(typeof values.enable_virtual_env !== 'boolean'
      ? { enable_virtual_env: config.enable_virtual_env }
      : {}),
  };
}

export function recommendationWarningKeys(response: RecommendationResponse) {
  const codes = new Set([...response.reasons, ...response.warnings].map((item) => item.code));
  const keys: string[] = [];
  if (codes.has('memory_not_verified')) keys.push('launchModel.recommendMemory');
  if (codes.has('virtual_env_candidate') || codes.has('virtual_env_not_verified'))
    keys.push('launchModel.recommendSetup');
  return keys;
}
