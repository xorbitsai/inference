import type { ComponentProps, ReactNode } from 'react';
import type { Input } from '@/components/ui/input';
import type { RadioGroup } from '@/components/ui/radio-group';
import type { Select } from '@/components/ui/select';
import type { Switch } from '@/components/ui/switch';
import { ModelType } from '@/constants';
import type { FormFieldProps } from '@/types/form';
import type { ModelEngineItem } from '@/types/services';
import type { CommonFormListProps } from './launch-dialog/common-form-list';

export type UnknownRecord = Record<string, unknown>;

export interface ModelSpec {
  cache_status?: boolean | boolean[];
  model_format: string;
  model_uri: string;
  quantization?: string;
  quantizations?: string[];
  model_size_in_billions: string | number;
  gguf_quantizations?: string[];
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
  download_hubs?: string[];
  gguf_quantizations?: string[];
  lightning_versions?: string[];
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

export type FormatIndex = {
  quantizations: Set<string>;
  sizes: Map<
    string,
    {
      multimodalProjectors: Set<string>;
      quantizations: Set<string>;
      value: ModelEngineItem['model_size_in_billions'];
    }
  >;
};

export type EngineIndex = Map<string, Map<string, FormatIndex>>;

export type LaunchFieldBase = {
  name: string;
  colSpan?: 1 | 2;
  show?: boolean;
};

export type FormLaunchFieldBase = LaunchFieldBase & Omit<FormFieldProps, 'children'>;

export type LaunchFieldConfig =
  | (FormLaunchFieldBase & {
      type: 'input';
      fieldProps?: ComponentProps<typeof Input>;
    })
  | (FormLaunchFieldBase & {
      type: 'select';
      fieldProps?: ComponentProps<typeof Select>;
    })
  | (FormLaunchFieldBase & {
      type: 'switch';
      fieldProps?: ComponentProps<typeof Switch>;
    })
  | (FormLaunchFieldBase & {
      type: 'radio-group';
      fieldProps?: ComponentProps<typeof RadioGroup>;
    })
  | (LaunchFieldBase & {
      type: 'form-list';
      fieldProps: CommonFormListProps;
    })
  | (LaunchFieldBase & {
      type: 'custom';
      content: ReactNode;
    });
