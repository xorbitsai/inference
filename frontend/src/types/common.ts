import { LANGUAGES_KEYS, ModelType } from '@/constants';
import type { ReactNode } from 'react';

export type Locale = (typeof LANGUAGES_KEYS)[number];

export type Option<T extends string | number = string | number> = {
  label: string;
  value: T;
  disabled?: boolean;
  suffix?: ReactNode;
};

export type BaseFormListValueItem = { key: string; value: string };

export type RegisterModelType = Exclude<ModelType, ModelType.Video | ModelType.Custom>;

export interface FileUploadValue {
  file: File;
  type: 'image' | 'video' | 'audio' | 'document';
  url: string;
}