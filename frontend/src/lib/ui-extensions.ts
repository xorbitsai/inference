import type { ComponentType } from 'react';
import type { Locale } from '@/types/common';

/** Build-time contributions owned by a distribution wrapper. */
export interface NavigationExtension {
  path: string;
  label: string;
  labels?: Partial<Record<Locale, string>>;
  group?: string;
  Icon?: ComponentType<{ className?: string }>;
  adminOnly?: boolean;
}

export const navigationExtensions: readonly NavigationExtension[] = [];

/** Additional sections; existing section IDs may be used without declaring them. */
export interface NavigationGroupExtension {
  id: string;
  label: string;
  labels?: Partial<Record<Locale, string>>;
  /** Insert before this section, or before resources when omitted. */
  before?: string;
}

export const navigationGroupExtensions: readonly NavigationGroupExtension[] = [];

/** Optional distribution rules shared by navigation and direct page access. */
export interface PageAccessExtension {
  path: string;
  scope: string;
  appliesWhen?: string;
}
export const pageAccessExtensions: readonly PageAccessExtension[] = [];
