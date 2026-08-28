const normalizeValue = (value) =>
  value === undefined || value === null ? '' : String(value).trim().toLowerCase();

const normalizeModelSize = (value) => normalizeValue(value).replace('.', '_');

const getQuantizations = (spec) => {
  const quantizations = Array.isArray(spec.quantizations)
    ? spec.quantizations
    : [spec.quantization];

  return quantizations.map(normalizeValue).filter(Boolean);
};

/**
 * Return whether an engine-discovery entry can launch the exact specification
 * already used by a running model. Missing current values are treated as
 * dimensions that do not apply to that model family.
 *
 * @param {string | Array<{
 *   model_format?: string,
 *   model_size_in_billions?: string | number,
 *   quantization?: string,
 *   quantizations?: string[],
 * }>} engineMetadata
 * @param {{
 *   modelFormat?: string,
 *   modelSizeInBillions?: string | number,
 *   quantization?: string,
 * }} currentSpec
 */
export const hasCompatibleEngineSpec = (engineMetadata, currentSpec) => {
  if (!Array.isArray(engineMetadata)) return false;

  const currentFormat = normalizeValue(currentSpec.modelFormat);
  const currentSize = normalizeModelSize(currentSpec.modelSizeInBillions);
  const currentQuantization = normalizeValue(currentSpec.quantization);

  if (!currentFormat) return false;

  return engineMetadata.some((spec) => {
    if (normalizeValue(spec.model_format) !== currentFormat) return false;

    if (currentSize && normalizeModelSize(spec.model_size_in_billions) !== currentSize) {
      return false;
    }

    if (currentQuantization && !getQuantizations(spec).includes(currentQuantization)) {
      return false;
    }

    return true;
  });
};
