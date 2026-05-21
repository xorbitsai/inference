import { LANGUAGES_KEYS } from '@/constants';

export type Locale = (typeof LANGUAGES_KEYS)[number];

export interface BaseFormFieldProps<T = any> {
  value?: T
  onChange?: (value: T) => void

  error?: boolean

  placeholder?: string

  disabled?: boolean

  className?: string
}