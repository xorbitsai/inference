import type { ReactNode } from 'react';
import type { UserItem } from '@/types/services';

/** Row information available to custom authorization renderers. UI checks do not replace backend authorization. */
export interface UserAuthorizationContext {
  /** True for the signed-in user, false for another user, or null when identity cannot be decoded. */
  isCurrentUser: boolean | null;
  /** Refresh the user list after an authorization change. */
  onSuccess: () => Promise<void>;
}

/** Optional replacement for the built-in permission fields, display and editing actions. */
export interface UserAuthorizationConfig {
  /** Heading for the authorization column. */
  columnTitle: string;
  /** Optional page subtitle; the built-in subtitle is used when omitted. */
  description?: string;
  /** Extra creation fields inside the shared Form. Use FormField so validation and reset-on-reopen work. */
  createFields: ReactNode;
  /** Disable creation until required authorization data is available. */
  createDisabled?: boolean;
  /**
   * Create the account itself; the default create request is not sent.
   * Receives username, password, confirm_password and all custom FormField values.
   * Resolve only after creation succeeds. On failure, report the error (for example
   * through the request interceptor or a toast) and reject; the dialog retains its draft.
   */
  onCreateUser: (values: Record<string, unknown>) => Promise<unknown>;
  /** Render the user's authorization summary. */
  renderUser: (user: UserItem, context: UserAuthorizationContext) => ReactNode;
  /** Render authorization actions. The caller decides visibility, including for self or unknown identity. */
  renderActions: (user: UserItem, context: UserAuthorizationContext) => ReactNode;
}

export interface UserManagementProps {
  authorization?: UserAuthorizationConfig;
}

export type UserCreationAuthorization = Pick<
  UserAuthorizationConfig,
  'createFields' | 'createDisabled' | 'onCreateUser'
>;
