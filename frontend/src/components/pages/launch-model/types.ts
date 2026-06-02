import { ModelType } from '@/constants';
export type UnknownRecord = Record<string, unknown>;

export interface ModelSpec {
  cache_status: boolean | boolean[];
  model_format: string;
  model_uri: string;
  quantization: string;
  model_size_in_billions: string | number;
  [key: string]: unknown;
}

export interface CatalogModel {
  [key: string]: unknown;
  model_name: string;
  model_description: string;
  abilities: string[];
  languages: string[];
  detailUrl?: string;
  modelSpecs?: ModelSpec[];
  featured?: boolean;
  cached?: boolean;
}

export interface VirtualEnv {
  model_name?: string;
  model_engine?: string;
  python_version?: string;
  worker_ip?: string;
  env_path?: string;
  path?: string;
  real_path?: string;
  [key: string]: unknown;
}
export type RouteModelType =
  | ModelType.LLM
  | ModelType.Embedding
  | ModelType.Rerank
  | ModelType.Image
  | ModelType.Audio
  | ModelType.Video
  | ModelType.Custom;
  
export type RequestModelType =
  | ModelType.LLM
  | ModelType.Embedding
  | ModelType.Rerank
  | ModelType.Image
  | ModelType.Audio
  | ModelType.Video
  | ModelType.Flexible;