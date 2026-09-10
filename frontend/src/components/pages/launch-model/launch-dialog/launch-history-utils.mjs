export const LEGACY_LAUNCH_HISTORY_KEY = 'historyArr';
export const LAUNCH_HISTORY_KEY_PREFIX = 'xinference:launch-history:v2:';

export const normalizeTimestamp = (value) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim()) {
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) ? timestamp : null;
  }
  return null;
};

export const getLaunchHistoryStorageKey = (scope) =>
  scope ? `${LAUNCH_HISTORY_KEY_PREFIX}${encodeURIComponent(scope)}` : null;

export const getLaunchHistoryItemKey = (item) =>
  `${item.model_name}::${item.model_uid || '__default__'}::${item.created_by || '__anonymous__'}`;

export const normalizeLaunchHistoryItem = (item, defaults = {}) => {
  if (!item || typeof item !== 'object' || Array.isArray(item)) return null;
  if (typeof item.model_name !== 'string' || !item.model_name) return null;
  if (!item.data || typeof item.data !== 'object' || Array.isArray(item.data)) return null;

  const updatedAt = normalizeTimestamp(item.updated_at ?? item.created_at);
  if (updatedAt === null) return null;

  return {
    data: item.data,
    model_name: item.model_name,
    model_uid: typeof item.model_uid === 'string' ? item.model_uid : '',
    created_by: typeof item.created_by === 'string' ? item.created_by : defaults.created_by || '',
    updated_at: updatedAt,
    autostart_enabled: Boolean(item.autostart_enabled),
    source: item.source === 'local' ? 'local' : defaults.source || 'server',
    pending_sync:
      typeof item.pending_sync === 'boolean' ? item.pending_sync : Boolean(defaults.pending_sync),
    is_owner: typeof item.is_owner === 'boolean' ? item.is_owner : Boolean(defaults.is_owner),
  };
};

export const normalizeLaunchHistory = (items, defaults = {}) => {
  if (!Array.isArray(items)) return [];
  return items.map((item) => normalizeLaunchHistoryItem(item, defaults)).filter(Boolean);
};

export const mergeLaunchHistories = (serverItems, localItems) => {
  const merged = new Map();
  for (const item of serverItems) {
    merged.set(getLaunchHistoryItemKey(item), item);
  }
  for (const item of localItems) {
    const key = getLaunchHistoryItemKey(item);
    if (item.pending_sync) {
      const existing = merged.get(key);
      if (!existing || item.updated_at > existing.updated_at) {
        merged.set(key, item);
      }
    }
  }
  return [...merged.values()].sort((left, right) => right.updated_at - left.updated_at);
};

export const getModelLaunchHistory = (items, modelName) => {
  if (!modelName) return [];
  return items
    .filter((item) => item.model_name === modelName)
    .sort((left, right) => right.updated_at - left.updated_at);
};

export const selectLatestModelLaunchHistory = (items, modelName, formEdited = false) => {
  if (formEdited) return null;
  return getModelLaunchHistory(items, modelName)[0] || null;
};

export const migrateLegacyLaunchHistory = (items, authenticated) => {
  if (authenticated) return [];
  return normalizeLaunchHistory(items, {
    created_by: '',
    source: 'local',
    pending_sync: true,
    is_owner: true,
  }).map((item) => ({
    ...item,
    created_by: '',
    source: 'local',
    pending_sync: true,
    is_owner: true,
  }));
};
