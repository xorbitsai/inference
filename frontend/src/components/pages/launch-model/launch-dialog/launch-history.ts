import { NO_AUTH } from '@/constants';
import { getAccessToken } from '@/lib/auth-storage';
import request from '@/lib/request';
import { decodeJwtPayload } from '@/lib/utils';
import type { FormValues } from '@/types/form';
import { toOptionValue } from '../utils';
import {
  getLaunchHistoryItemKey,
  getLaunchHistoryStorageKey,
  getModelLaunchHistory,
  LEGACY_LAUNCH_HISTORY_KEY,
  mergeLaunchHistories,
  migrateLegacyLaunchHistory,
  normalizeLaunchHistory,
  normalizeLaunchHistoryItem,
} from './launch-history-utils.mjs';

export interface LaunchHistoryResponseItem {
  id?: number;
  model_name: string;
  model_uid: string;
  data: FormValues;
  created_by?: string;
  updated_by?: string;
  created_at?: string | number;
  updated_at?: string | number;
  autostart_enabled?: boolean;
  is_owner?: boolean;
}

export interface LaunchConfigHistoryItem {
  data: FormValues;
  model_name: string;
  model_uid: string;
  created_by: string;
  updated_at: number;
  autostart_enabled: boolean;
  source: 'server' | 'local';
  pending_sync: boolean;
  is_owner: boolean;
}

interface LaunchHistoryContext {
  username: string;
  storageKey: string | null;
}

export interface RefreshLaunchHistoryResult {
  history: LaunchConfigHistoryItem[];
  usedLocalFallback: boolean;
  syncFailed: boolean;
}

const USERNAME_CLAIMS = ['username', 'preferred_username', 'name', 'sub', 'user_name', 'user_id'];

const resolveUsername = (token?: string): string => {
  const payload = decodeJwtPayload(token);
  if (!payload) return '';

  for (const key of USERNAME_CLAIMS) {
    const value = payload[key];
    if ((typeof value === 'string' && value.trim()) || typeof value === 'number') {
      return String(value);
    }
  }
  return '';
};

export const getLaunchHistoryContext = (
  authenticated: boolean | undefined
): LaunchHistoryContext => {
  if (authenticated === false) {
    return {
      username: '',
      storageKey: getLaunchHistoryStorageKey('anonymous'),
    };
  }
  if (authenticated !== true) {
    return { username: '', storageKey: null };
  }

  const token = getAccessToken();
  const username = token && token !== NO_AUTH ? resolveUsername(token) : '';
  return {
    username,
    storageKey: getLaunchHistoryStorageKey(username),
  };
};

const parseStorageArray = (key: string): unknown[] => {
  if (typeof window === 'undefined') return [];
  try {
    const value = JSON.parse(window.localStorage.getItem(key) || '[]');
    return Array.isArray(value) ? value : [];
  } catch (error) {
    console.warn(`Unable to read launch history cache ${key}`, error);
    return [];
  }
};

export const writeLaunchConfigHistory = (
  history: LaunchConfigHistoryItem[],
  authenticated: boolean | undefined
): void => {
  if (typeof window === 'undefined') return;
  const { storageKey } = getLaunchHistoryContext(authenticated);
  if (!storageKey) return;
  try {
    window.localStorage.setItem(storageKey, JSON.stringify(history));
  } catch (error) {
    console.warn('Unable to write launch history cache', error);
  }
};

export const readLaunchConfigHistory = (
  authenticated: boolean | undefined
): LaunchConfigHistoryItem[] => {
  if (typeof window === 'undefined') return [];
  const context = getLaunchHistoryContext(authenticated);
  if (!context.storageKey) return [];

  const stored = normalizeLaunchHistory(parseStorageArray(context.storageKey), {
    created_by: context.username,
    source: 'server',
    pending_sync: false,
    is_owner: true,
  }) as LaunchConfigHistoryItem[];
  let scopedCacheExists = false;
  try {
    scopedCacheExists = window.localStorage.getItem(context.storageKey) !== null;
  } catch (error) {
    console.warn(`Unable to inspect launch history cache ${context.storageKey}`, error);
  }
  if (stored.length || authenticated || scopedCacheExists) return stored;

  const migrated = migrateLegacyLaunchHistory(
    parseStorageArray(LEGACY_LAUNCH_HISTORY_KEY),
    authenticated
  ) as LaunchConfigHistoryItem[];
  if (migrated.length) writeLaunchConfigHistory(migrated, authenticated);
  return migrated;
};

export const getModelConfigHistory = (
  history: LaunchConfigHistoryItem[],
  modelName?: string
): LaunchConfigHistoryItem[] =>
  getModelLaunchHistory(history, modelName) as LaunchConfigHistoryItem[];

export const getLatestModelConfigHistory = (
  modelName: string | undefined,
  authenticated: boolean
) => getModelConfigHistory(readLaunchConfigHistory(authenticated), modelName)[0];

const replaceCachedItem = (
  item: LaunchConfigHistoryItem,
  authenticated: boolean | undefined
): LaunchConfigHistoryItem[] => {
  const history = readLaunchConfigHistory(authenticated);
  const key = getLaunchHistoryItemKey(item);
  const next = [item, ...history.filter((entry) => getLaunchHistoryItemKey(entry) !== key)].sort(
    (left, right) => right.updated_at - left.updated_at
  );
  writeLaunchConfigHistory(next, authenticated);
  return next;
};

export const removeCachedLaunchHistoryItem = (
  item: LaunchConfigHistoryItem,
  authenticated: boolean
): LaunchConfigHistoryItem[] => {
  const key = getLaunchHistoryItemKey(item);
  const next = readLaunchConfigHistory(authenticated).filter(
    (entry) => getLaunchHistoryItemKey(entry) !== key
  );
  writeLaunchConfigHistory(next, authenticated);
  return next;
};

const postLaunchHistory = (item: LaunchConfigHistoryItem) =>
  request.post('/v1/launch_history', {
    model_name: item.model_name,
    model_uid: item.model_uid,
    data: item.data,
  });

export const saveLaunchConfigHistory = async (
  values: FormValues,
  authenticated: boolean | undefined
): Promise<boolean> => {
  const modelName = toOptionValue(values.model_name);
  const modelUid = toOptionValue(values.model_uid);
  if (!modelName) return true;

  const context = getLaunchHistoryContext(authenticated);
  const pending = normalizeLaunchHistoryItem(
    {
      data: values,
      model_name: modelName,
      model_uid: modelUid,
      created_by: context.username,
      updated_at: Date.now(),
      autostart_enabled: false,
      source: 'local',
      pending_sync: true,
      is_owner: true,
    },
    {}
  ) as LaunchConfigHistoryItem;

  if (context.storageKey) {
    try {
      replaceCachedItem(pending, authenticated);
    } catch (error) {
      console.warn('Unable to stage launch history locally', error);
    }
  }
  try {
    await postLaunchHistory(pending);
    if (context.storageKey) {
      try {
        replaceCachedItem({ ...pending, source: 'server', pending_sync: false }, authenticated);
      } catch (error) {
        console.warn('Unable to update the synchronized launch history cache', error);
      }
    }
    return true;
  } catch (error) {
    console.warn('Unable to persist launch history', error);
    return false;
  }
};

const retryPendingHistory = async (modelName: string, authenticated: boolean): Promise<boolean> => {
  const pending = getModelConfigHistory(readLaunchConfigHistory(authenticated), modelName).filter(
    (item) => item.pending_sync
  );
  let failed = false;
  for (const item of pending) {
    try {
      await postLaunchHistory(item);
      replaceCachedItem({ ...item, source: 'server', pending_sync: false }, authenticated);
    } catch (error) {
      console.warn('Unable to retry launch history synchronization', error);
      failed = true;
    }
  }
  return failed;
};

export const refreshLaunchConfigHistory = async (
  modelName: string,
  authenticated: boolean
): Promise<RefreshLaunchHistoryResult> => {
  const syncFailed = await retryPendingHistory(modelName, authenticated);
  const local = readLaunchConfigHistory(authenticated);
  try {
    const response = await request.get<LaunchHistoryResponseItem[]>('/v1/launch_history', {
      params: { model_name: modelName },
    });
    if (!Array.isArray(response)) throw new Error('Launch history response is not an array');

    const server = normalizeLaunchHistory(response, {
      source: 'server',
      pending_sync: false,
    }) as LaunchConfigHistoryItem[];
    const otherModels = local.filter((item) => item.model_name !== modelName);
    const currentModel = local.filter((item) => item.model_name === modelName);
    const merged = [
      ...otherModels,
      ...(mergeLaunchHistories(server, currentModel) as LaunchConfigHistoryItem[]),
    ].sort((left, right) => right.updated_at - left.updated_at);
    writeLaunchConfigHistory(merged, authenticated);
    return { history: merged, usedLocalFallback: false, syncFailed };
  } catch (error) {
    console.warn('Unable to load launch history from server', error);
    return { history: local, usedLocalFallback: true, syncFailed };
  }
};

export const deleteLaunchHistory = async (item: LaunchConfigHistoryItem): Promise<void> => {
  const base = `/v1/launch_history/${encodeURIComponent(item.model_name)}`;
  const path = item.model_uid ? `${base}/${encodeURIComponent(item.model_uid)}` : base;
  await request.delete(path);
};
