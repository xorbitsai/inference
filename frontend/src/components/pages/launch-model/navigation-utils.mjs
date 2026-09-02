const LAUNCH_MODEL_PATHS = new Set([
  'llm',
  'embedding',
  'rerank',
  'image',
  'audio',
  'video',
  'world',
]);
const CUSTOM_MODEL_TYPES = new Set(['llm', 'embedding', 'rerank', 'image', 'audio', 'flexible']);
let pendingLaunchModelTarget = null;

export function setPendingLaunchModelTarget(modelType, modelName) {
  pendingLaunchModelTarget = { modelType, modelName };
}

export function peekPendingLaunchModelTarget() {
  return pendingLaunchModelTarget;
}

export function clearPendingLaunchModelTarget(target) {
  if (pendingLaunchModelTarget === target) {
    pendingLaunchModelTarget = null;
  }
}

export function findModelRegistration(registrationGroups, targetModelName) {
  if (!targetModelName) return null;

  for (const { modelType, registrations } of registrationGroups) {
    const registration = registrations.find((item) => item.model_name === targetModelName);
    if (registration) {
      return {
        modelType,
        isBuiltin: registration.is_builtin,
      };
    }
  }

  return null;
}

export function getLaunchModelHref(modelType, modelName, isBuiltin) {
  const normalizedType = String(modelType ?? '')
    .trim()
    .toLowerCase();
  const normalizedName = String(modelName ?? '').trim();

  if (!normalizedName) return null;

  const useCustomRoute = isBuiltin === false || normalizedType === 'flexible';
  if (useCustomRoute) {
    if (!CUSTOM_MODEL_TYPES.has(normalizedType)) return null;
  } else if (!LAUNCH_MODEL_PATHS.has(normalizedType)) {
    return null;
  }

  return `/launch-model/${useCustomRoute ? 'custom' : normalizedType}`;
}

export function prioritizeModelByName(models, targetModelName) {
  if (!targetModelName) return models;

  const targetIndex = models.findIndex((model) => model.model_name === targetModelName);
  if (targetIndex <= 0) return models;

  return [models[targetIndex], ...models.slice(0, targetIndex), ...models.slice(targetIndex + 1)];
}
