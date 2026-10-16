/** Build-time contributions owned by a distribution wrapper. */
export interface NavigationExtension {
  path: string;
  label: string;
  adminOnly?: boolean;
}

export const navigationExtensions: readonly NavigationExtension[] = [];

/** Optional distribution rules shared by navigation and direct page access. */
export interface PageAccessExtension {
  path: string;
  scope: string;
  appliesWhen?: string;
}
export const pageAccessExtensions: readonly PageAccessExtension[] = [];
