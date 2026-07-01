export const SAMPLING_METHOD_OPTIONS = [
  'default',
  'DPM++ 2M',
  'DPM++ 2M Karras',
  'DPM++ 2M SDE',
  'DPM++ 2M SDE Karras',
  'DPM++ SDE',
  'DPM++ SDE Karras',
  'DPM2',
  'DPM2 Karras',
  'DPM2 a',
  'DPM2 a Karras',
  'Euler',
  'Euler a',
  'Heun',
  'LMS',
  'LMS Karras',
].map((value) => ({
  label: value === 'default' ? 'Default' : value,
  value,
}));

export const OCR_TYPE_OPTIONS = [
  { label: 'Ocr', value: 'ocr' },
  { label: 'Markdown', value: 'markdown' },
  { label: 'Format', value: 'format' },
];

export const OCR_MODEL_SIZE_OPTIONS = [
  { label: 'Tiny', value: 'tiny' },
  { label: 'Small', value: 'small' },
  { label: 'Base', value: 'base' },
  { label: 'Large', value: 'large' },
  { label: 'Gundam', value: 'gundam' },
];