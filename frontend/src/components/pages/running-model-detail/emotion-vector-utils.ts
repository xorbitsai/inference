export const INDEX_TTS_EMOTION_DIMENSIONS = [
  { key: 'happy', label: 'Happy' },
  { key: 'angry', label: 'Angry' },
  { key: 'sad', label: 'Sad' },
  { key: 'afraid', label: 'Afraid' },
  { key: 'disgusted', label: 'Disgusted' },
  { key: 'melancholic', label: 'Melancholic' },
  { key: 'surprised', label: 'Surprised' },
  { key: 'calm', label: 'Calm' },
] as const;

export const INDEX_TTS_EMOTION_MAX_TOTAL = 0.8;

export const EMPTY_INDEX_TTS_EMOTION_VECTOR = INDEX_TTS_EMOTION_DIMENSIONS.map(() => 0);

const EMOTION_TOTAL_EPSILON = 1e-9;

// IndexTTS-2 and IndexTTS-2.5 share the IndexTTS2 model family.
// Model-name checks keep the UI working if older runtime metadata omits the family.
export function isIndexTTSEmotionModel(
  modelFamily: string | undefined,
  modelName: string | undefined
): boolean {
  return modelFamily === 'IndexTTS2' || modelName === 'IndexTTS2' || modelName === 'IndexTTS-2.5';
}

// Validate and copy the emotion vector accepted by the IndexTTS runtime.
export function parseIndexTTSEmotionVector(value: unknown): number[] | undefined {
  if (!Array.isArray(value) || value.length !== INDEX_TTS_EMOTION_DIMENSIONS.length) {
    return undefined;
  }

  if (
    value.some((item) => typeof item !== 'number' || !Number.isFinite(item) || item < 0 || item > 1)
  ) {
    return undefined;
  }

  const vector = [...value] as number[];
  const total = vector.reduce((sum, item) => sum + item, 0);
  if (total > INDEX_TTS_EMOTION_MAX_TOTAL + EMOTION_TOTAL_EPSILON) {
    return undefined;
  }

  return vector;
}
