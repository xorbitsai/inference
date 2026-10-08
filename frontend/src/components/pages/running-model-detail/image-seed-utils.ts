import { createRandomSeed, MAX_SEED } from './seed-utils';

export const MAX_IMAGE_SEED = MAX_SEED;

const normalizeImageCount = (count: unknown): number => {
  const parsed = Number(count);
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : 1;
};

/** Parse only seeds supplied by the user; backend pads missing positions. */
export const parseImageSeeds = (value: unknown, count: unknown): number[] => {
  const imageCount = normalizeImageCount(count);
  const text = typeof value === 'string' ? value : value == null ? '' : String(value);
  const tokens = text.trim() === '' ? [] : text.replaceAll('，', ',').split(',');

  if (tokens.length > imageCount) {
    throw new Error(`Enter no more than ${imageCount} seeds.`);
  }

  return tokens.map((token) => {
    const normalized = token.trim();
    if (normalized === '') return -1;
    if (!/^-?\d+$/.test(normalized)) {
      throw new Error('Seeds must be integers separated by commas.');
    }

    const seed = Number(normalized);
    if (!Number.isSafeInteger(seed) || seed < -1 || seed > MAX_IMAGE_SEED) {
      throw new Error(`Each seed must be -1 or an integer from 0 to ${MAX_IMAGE_SEED}.`);
    }
    return seed;
  });
};

export const createRandomImageSeed = (): number => {
  return createRandomSeed();
};

/** Generate a fresh seed for every image when the random button is clicked. */
export const generateRandomImageSeeds = (
  count: unknown,
  randomSeed: () => number = createRandomImageSeed
): number[] => {
  const imageCount = normalizeImageCount(count);
  return Array.from({ length: imageCount }, () => randomSeed());
};

export const formatImageSeeds = (seeds: number[]): string => seeds.join(', ');
