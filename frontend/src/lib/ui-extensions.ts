/** Build-time contributions owned by a distribution wrapper. */
export interface NavigationExtension {
  path: string;
  label: string;
  adminOnly?: boolean;
}

export const navigationExtensions: readonly NavigationExtension[] = [];
