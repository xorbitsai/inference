export const MAX_SEED = 2 ** 31 - 1;

export const parseScalarSeed = (value: unknown): number | undefined => {
  if (value === undefined || value === null || value === '') return undefined;

  const normalized = typeof value === 'string' ? value.trim() : value;
  if (normalized === '') return undefined;

  const seed = Number(normalized);
  if (!Number.isSafeInteger(seed) || seed < -1 || seed > MAX_SEED) {
    throw new Error(`Seed must be -1 or an integer from 0 to ${MAX_SEED}.`);
  }
  return seed;
};

export const createRandomSeed = (): number => {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    return crypto.getRandomValues(new Uint32Array(1))[0] & MAX_SEED;
  }
  return Math.floor(Math.random() * (MAX_SEED + 1));
};
