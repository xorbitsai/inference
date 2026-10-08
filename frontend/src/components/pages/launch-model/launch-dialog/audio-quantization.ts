import type { ModelSpec } from '../types';

export function audioSpecsForEngine(specs: ModelSpec[], engine: string): ModelSpec[] {
  if (!engine) return specs;

  const matching = specs.filter(
    (spec) =>
      typeof spec.model_engine === 'string' &&
      spec.model_engine.toLowerCase() === engine.toLowerCase()
  );
  // Legacy audio models have no engine metadata. Match the backend fallback,
  // but never expose another explicitly named engine's quantizations.
  return matching.length ? matching : specs.filter((spec) => !spec.model_engine);
}
