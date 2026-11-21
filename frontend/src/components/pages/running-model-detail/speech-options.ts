import type { RunningModelDetail } from '@/types/services';

import { canPlayPcmStream } from './audio-stream';
import { stringValue } from './utils';

export function getSpeechOptions(
  model: Pick<RunningModelDetail, 'model_family' | 'model_name' | 'model_engine'>,
  format: unknown
) {
  const isZonos3 = model.model_family === 'zonos3_mini' || model.model_name === 'ZONOS3-Mini';
  const engine = model.model_engine?.toLowerCase();
  const selectedFormat = stringValue(format).trim().toLowerCase() || 'mp3';
  const offerPcm =
    isZonos3 &&
    (engine === 'sglang-omni' || engine === 'mlx' || engine === 'ggml') &&
    canPlayPcmStream();
  const responseFormat = isZonos3 && selectedFormat === 'pcm' && !offerPcm ? 'mp3' : selectedFormat;

  return {
    isZonos3,
    responseFormat,
    offerPcm,
    canStream: !isZonos3 || (offerPcm && responseFormat === 'pcm'),
    acceptsLanguage: !isZonos3 || engine !== 'sglang-omni',
    pcmSampleRate: isZonos3 && responseFormat === 'pcm' ? 44100 : undefined,
  };
}
