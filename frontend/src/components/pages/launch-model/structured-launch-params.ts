// Keep structured conversion local to model launch; other text parameters must
// retain the shared form converter's existing behavior.
export const STRUCTURED_LAUNCH_PARAMETERS = ['realtime_options'] as const;

export function isStructuredLaunchParameter(parameter: string) {
  return STRUCTURED_LAUNCH_PARAMETERS.some((name) => name === parameter);
}

export class StructuredLaunchParameterError extends Error {
  readonly messageKey: string;

  constructor(
    readonly parameter: string,
    reason: 'invalidJson' | 'objectRequired'
  ) {
    super(
      reason === 'invalidJson'
        ? `${parameter} must contain valid JSON.`
        : `${parameter} must be a JSON object.`
    );
    this.name = 'StructuredLaunchParameterError';
    this.messageKey =
      reason === 'invalidJson'
        ? 'launchModel.structuredParameterInvalidJson'
        : 'launchModel.structuredParameterObjectRequired';
  }
}

export function parseStructuredLaunchParameter(parameter: string, value: unknown) {
  let parsed = value;
  if (typeof parsed === 'string') {
    try {
      parsed = JSON.parse(parsed.trim());
    } catch {
      // Do not expose input text (or JSON.parse's source excerpt) in errors.
      throw new StructuredLaunchParameterError(parameter, 'invalidJson');
    }
  }
  if (
    parsed === null ||
    typeof parsed !== 'object' ||
    ![Object.prototype, null].includes(Object.getPrototypeOf(parsed))
  ) {
    throw new StructuredLaunchParameterError(parameter, 'objectRequired');
  }
  return { ...(parsed as Record<string, unknown>) };
}
