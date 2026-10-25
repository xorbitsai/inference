import { RegisterModelType } from '@/types/common';
import { CUSTOM_MODEL_OPTIONS, ModelType } from '@/constants';

export function getRigisterModelTyps(value: string): RegisterModelType {
  const matched = CUSTOM_MODEL_OPTIONS.find((item) => item.value === value);

  return (matched?.value || ModelType.LLM) as RegisterModelType;
}
