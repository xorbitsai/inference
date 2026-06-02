import {
  FieldName,
  NamePath,
} from '@/types/form';

export function normalizeNamePath(
  name: FieldName
): NamePath {
  if (Array.isArray(name)) {
    return name;
  }

  if (typeof name === 'number') {
    return [name];
  }

  return name
    .split('.')
    .filter(Boolean)
    .map((item) =>
      /^\d+$/.test(item)
        ? Number(item)
        : item
    );
}

export function getNamePathString(
  name: FieldName
) {
  return normalizeNamePath(name)
    .join('.');
}

export function getValue(
  store: any,
  name: FieldName
) {
  const path =
    normalizeNamePath(name);

  return path.reduce(
    (current, key) =>
      current?.[key],
    store
  );
}

export function hasValue(
  store: unknown,
  name: FieldName
) {
  const path =
    normalizeNamePath(name);

  let current = store as
    | Record<string | number, unknown>
    | undefined
    | null;

  for (const key of path) {
    if (
      current === null ||
      current === undefined ||
      !Object.prototype.hasOwnProperty.call(
        current,
        key
      )
    ) {
      return false;
    }

    current = current[key] as
      | Record<string | number, unknown>
      | undefined
      | null;
  }

  return true;
}

export function setValue(
  store: any,
  name: FieldName,
  value: any
) {
  const path =
    normalizeNamePath(name);

  const cloned =
    structuredClone(store);

  let current = cloned;

  for (
    let i = 0;
    i < path.length - 1;
    i++
  ) {
    const key = path[i];

    if (
      current[key] === null ||
      typeof current[key] !==
        'object'
    ) {
      current[key] =
        typeof path[i + 1] ===
        'number'
          ? []
          : {};
    }

    current = current[key];
  }

  current[path[path.length - 1]] =
    value;

  return cloned;
}

export function deleteValue<T>(
  store: T,
  name: FieldName
) {
  const path =
    normalizeNamePath(name);

  if (!path.length || !hasValue(store, path)) {
    return store;
  }

  const cloned =
    structuredClone(store);

  let current = cloned as Record<
    string | number,
    unknown
  >;

  const parents: Array<{
    target: Record<
      string | number,
      unknown
    >;
    key: string | number;
  }> = [];

  for (
    let i = 0;
    i < path.length - 1;
    i++
  ) {
    const key = path[i];

    parents.push({
      target: current,
      key,
    });

    current = current?.[key] as Record<
      string | number,
      unknown
    >;

    if (
      current === null ||
      current === undefined
    ) {
      return cloned as T;
    }
  }

  const lastKey =
    path[path.length - 1];

  if (
    Array.isArray(current) &&
    typeof lastKey === 'number'
  ) {
    current.splice(lastKey, 1);
  } else {
    delete current[lastKey];
  }

  for (
    let i = parents.length - 1;
    i >= 0;
    i--
  ) {
    const { target, key } =
      parents[i];

    const value =
      target[key];

    if (
      Array.isArray(target) ||
      Array.isArray(value) ||
      value === null ||
      typeof value !== 'object' ||
      Object.keys(value).length > 0
    ) {
      break;
    }

    delete target[key];
  }

  return cloned as T;
}

export function deepMerge(
  target: any,
  source: any
) {
  // array:
  // direct replace
  if (Array.isArray(source)) {
    return structuredClone(source);
  }

  // primitive
  if (
    source === null ||
    typeof source !== 'object'
  ) {
    return source;
  }

  const result = structuredClone(
    target ?? {}
  );

  Object.keys(source).forEach(
    (key) => {
      result[key] = deepMerge(
        result[key],
        source[key]
      );
    }
  );

  return result;
}
