import type { ReactNode } from 'react';
import type { UserItem } from '@/types/services';

export interface UserAuthorizationConfig {
  columnTitle: string;
  description?: string;
  createFields: ReactNode;
  createDisabled?: boolean;
  onCreateUser: (values: Record<string, unknown>) => Promise<unknown>;
  renderUser: (user: UserItem) => ReactNode;
  renderActions: (user: UserItem, onSuccess: () => Promise<void>) => ReactNode;
}

export type UserCreationOptions =
  | { authorizationMode?: 'permissions'; authorizationConfig?: never }
  | {
      authorizationMode: 'roles';
      authorizationConfig: Pick<
        UserAuthorizationConfig,
        'createFields' | 'createDisabled' | 'onCreateUser'
      >;
    };

export type UserManagementProps =
  | { authorizationMode?: 'permissions'; authorizationConfig?: never }
  | { authorizationMode: 'roles'; authorizationConfig: UserAuthorizationConfig };
