import type { ComponentType, ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';

import type { FileUploadValue } from '@/types/common';
import type { ModelAbility } from '@/constants';
import type { FormInstance, FormValues } from '@/types/form';
import type { RunningModelDetail } from '@/types/services';

type Attachment = Omit<FileUploadValue, 'file'> & {
  file?: File;
};

export interface CapabilityFormProps {
  form: FormInstance;
  model: RunningModelDetail;
  modelUid: string;
}

export interface CapabilityResultProps {
  result?: unknown;
  values?: FormValues;
  loading?: boolean;
  progress?: number;
  ability: ModelAbility;
}

export interface TransformContext {
  modelUid: string;
  model: RunningModelDetail;
  values: FormValues;
  requestId?: string;
}

export interface CapabilityConfig {
  ability: ModelAbility;
  label: string;
  icon: LucideIcon;
  requestApi: string;
  initialValues?: FormValues;
  showProgress?: boolean;
  formPanel: ComponentType<CapabilityFormProps>;
  resultPanel: ComponentType<CapabilityResultProps>;
  transformValues: (context: TransformContext) => BodyInit | Record<string, unknown>;
  responseType?: 'blob';
}

export interface ChatSettings {
  max_tokens: number;
  temperature: number;
  stream: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  loading: boolean;
  reasoning?: string;
  attachment?: Attachment;
  thinkingContent?: string;
  thinkingCompleted?: boolean;
  success?: boolean;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface FieldSchema {
  name: string;
  label: ReactNode;
  type?: 'text' | 'textarea' | 'number' | 'select' | 'switch' | 'upload';
}
