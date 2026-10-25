/**
 * A nearly identical implementation of lodash.isEmpty
 *
 * Rules:
 * - null / undefined => true
 * - boolean => true
 * - number => true
 * - string => length === 0
 * - array => length === 0
 * - arguments => length === 0
 * - map / set => size === 0
 * - typed array / buffer => length === 0
 * - plain object => no enumerable own keys
 * - prototype object => no own keys except constructor
 */
export function isEmpty(value: unknown): boolean {
  // null / undefined
  if (value == null) {
    return true;
  }

  // Array-like
  if (typeof value === 'string' || Array.isArray(value)) {
    return value.length === 0;
  }

  if (isArguments(value) || isTypedArray(value) || isBuffer(value)) {
    return (value as ArrayLike<unknown>).length === 0;
  }

  // Map / Set
  if (value instanceof Map || value instanceof Set) {
    return value.size === 0;
  }

  // Prototype objects
  if (isPrototype(value)) {
    return Object.keys(value).length === 0;
  }

  // Objects
  if (typeof value === 'object') {
    for (const key in value) {
      if (Object.prototype.hasOwnProperty.call(value, key)) {
        return false;
      }
    }

    return true;
  }

  // number / boolean / symbol / function
  return true;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isArguments(value: unknown): boolean {
  return Object.prototype.toString.call(value) === '[object Arguments]';
}

function isTypedArray(value: unknown): boolean {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function isBuffer(value: unknown): boolean {
  return (
    typeof Buffer !== 'undefined' && typeof Buffer.isBuffer === 'function' && Buffer.isBuffer(value)
  );
}

function isPrototype(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }

  const Ctor = value.constructor;
  const proto = (typeof Ctor === 'function' && Ctor.prototype) || Object.prototype;

  return value === proto;
}

export function isNumber(value: unknown): value is number {
  return typeof value === 'number' || Object.prototype.toString.call(value) === '[object Number]';
}
