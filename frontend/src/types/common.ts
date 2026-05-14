import { languages } from '@/constants';

export type Locale = (typeof languages)[number]['value'];
