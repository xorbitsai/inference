import { LANGUAGES_KEYS } from '@/constants';
import type { ReactNode } from 'react';

export type Locale = (typeof LANGUAGES_KEYS)[number];

export type Option = {
  label: string;
  value: string;
  disabled?: boolean;
  suffix?: ReactNode;
};
