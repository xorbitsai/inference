export interface ModelPermissionRecord {
  id?: number;
  api_key_id?: number;
  permission_type?: PermissionMode;
  permission_value?: string | null;
}

export type ModelPermission = ModelPermissionRecord;

export interface ApiKey {
  id: number;
  user_id: number | string | null;
  owner_username?: string | null;
  key_prefix: string;
  name: string | null;
  description: string | null;
  enabled: boolean;
  expires_at: string | null;
  model_permissions: ModelPermission[];
  created_at: string | null;
  banned_count?: number;
  banned?: number;
}

export interface RunningModel {
  id?: string;
  model_uid?: string;
  model_name?: string;
  model_type?: string;
}

export interface ApiKeyUser {
  id: number | string;
  username: string;
}

export type PermissionMode = 'all' | 'model_type' | 'model_id' | 'mixed';
export type DialogMode = 'create' | 'edit';

export interface KeyFormValues {
  name?: string;
  description?: string;
  user_id?: string;
  expires_at?: string;
  permission_mode?: PermissionMode;
  model_types?: string[];
  model_ids?: string[];
}

export const MODEL_TYPE_OPTIONS = [
  { value: 'LLM', label: 'LLM' },
  { value: 'embedding', label: 'Embedding' },
  { value: 'rerank', label: 'Rerank' },
  { value: 'image', label: 'Image' },
  { value: 'video', label: 'Video' },
  { value: 'audio', label: 'Audio' },
];

export const getBannedCount = (key: ApiKey) => key.banned_count ?? key.banned ?? '-';

export const getPermissionValue = (permission: ModelPermission) => {
  if (typeof permission === 'string') {
    return permission;
  }

  return permission.permission_value || '';
};

export const getPermissionType = (permission: ModelPermission) => {
  if (typeof permission === 'string') {
    return '';
  }

  return permission.permission_type || '';
};

export const getPermissionLabel = (permission: ModelPermission) => {
  if (typeof permission === 'string') {
    return permission;
  }

  return permission.permission_value || permission.permission_type || '-';
};
