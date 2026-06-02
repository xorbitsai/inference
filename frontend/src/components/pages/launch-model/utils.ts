import { XINFERENCE_IO, ModelType } from '@/constants';
import { LAUNCH_MODEL_ROUTE_TABS } from '@/constants/launch';
import type {
  ModelSpec,
  CatalogModel,
  UnknownRecord,
  VirtualEnv,
  RequestModelType,
  RouteModelType,
} from './types';

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

export function getVirtualEnvPath(item: VirtualEnv) {
  return String(item.env_path || item.path || item.real_path || '');
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
