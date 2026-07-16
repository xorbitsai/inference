import { MODEL_TYPE_ABILITY_MAP } from '@/constants/running';
import type { FormValues } from '@/types/form';
import type { ChatChoicesMessage } from '@/types/services';
import type { FileUploadValue } from '@/types/common';

export function createId(prefix = 'item') {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function stringValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

export function numberValue(value: unknown, fallback: number): number {
  if (value === '') {
    return fallback;
  }

  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export function booleanValue(value: unknown): boolean {
  return value === true;
}

export function positiveNumber(value: unknown): number | undefined {
  const next = numberValue(value, -1);
  return next > 0 ? next : undefined;
}

export function sizeFromValues(values: FormValues, fallback?: string): string | undefined {
  const width = numberValue(values.width, -1);
  const height = numberValue(values.height, -1);

  if (width > 0 && height > 0) {
    return `${Math.round(width)}*${Math.round(height)}`;
  }

  return fallback;
}

export function appendIfPresent(formData: FormData, key: string, value: unknown) {
  if (value === undefined || value === null || value === '') {
    return;
  }

  formData.append(key, String(value));
}

interface BuildGenerationKwargsOptions {
  excludeKeys?: string[];
}

export function buildGenerationKwargs(
  values: FormValues,
  requestId?: string,
  options: BuildGenerationKwargsOptions = {}
) {
  const kwargs: Record<string, string | number | boolean> = {};
  const excludedKeys = new Set(options.excludeKeys || []);
  const guidanceScale = positiveNumber(values.guidance_scale);
  const numInferenceSteps = positiveNumber(values.num_inference_steps);
  const paddingImageToMultiple = positiveNumber(values.padding_image_to_multiple);
  const samplerName = stringValue(values.sampler_name, 'default');
  const strength = numberValue(values.strength, -1);

  if (requestId) {
    kwargs.request_id = requestId;
  }

  if (guidanceScale !== undefined) {
    kwargs.guidance_scale = guidanceScale;
  }

  if (numInferenceSteps !== undefined) {
    kwargs.num_inference_steps = Math.round(numInferenceSteps);
  }

  if (paddingImageToMultiple !== undefined) {
    kwargs.padding_image_to_multiple = Math.round(paddingImageToMultiple);
  }

  if (strength >= 0) {
    kwargs.strength = strength;
  }

  if (samplerName && samplerName !== 'default') {
    kwargs.sampler_name = samplerName;
  }

  ['num_frames', 'fps', 'width', 'height'].forEach((key) => {
    if (excludedKeys.has(key)) return;

    const value = positiveNumber(values[key]);
    if (value !== undefined) {
      kwargs[key] = Math.round(value);
    }
  });

  return kwargs;
}

export function firstUpload(values: FormValues, key: string): FileUploadValue | undefined {
  const value = values[key];
  return Array.isArray(value) ? value[0] : undefined;
}

export function uploadList(values: FormValues, key: string): FileUploadValue[] {
  const value = values[key];
  return Array.isArray(value) ? value : [];
}

export function formatOCRPrompt(ocrType: string) {
  if (ocrType === 'markdown') {
    return '<image>\nConvert this document to clean markdown format. Extract the text content and format it properly using markdown syntax. Do not include any coordinate annotations or special formatting markers.';
  }

  if (ocrType === 'format') {
    return '<image>\n<|grounding|>Convert the document to markdown with structure annotations. Include coordinate information for text regions and maintain the document structure.';
  }

  return '<image>\nFree OCR. Extract all text content from the image.';
}

const mediaTypeMap = {
  audio: 'audio/mp3',
  image: 'image/jpeg',
  video: 'video/mp4',
} as const;

type MediaType = keyof typeof mediaTypeMap;

export function transformFileInfoForResult(message?: ChatChoicesMessage) {
  if (!message) return undefined;

  for (const [type, mimeType] of Object.entries(mediaTypeMap) as [
    MediaType,
    (typeof mediaTypeMap)[MediaType],
  ][]) {
    const media = message[type];

    if (media?.data) {
      return {
        type,
        url: `data:${mimeType};base64,${media.data}`,
      };
    }
  }

  return undefined;
}

export function transformRunningModelDetail<T extends object>(detail: T) {
  if (!isRecord(detail) || !detail) return {};
  const modelType = typeof detail.model_type === 'string' ? detail.model_type : undefined;

  return {
    ...detail,
    // fix model_ability was not returned when model_type was Rerank or Embedding.
    model_ability: Array.isArray(detail.model_ability)
      ? detail.model_ability
      : (modelType && MODEL_TYPE_ABILITY_MAP[modelType]) || [],
  };
}
