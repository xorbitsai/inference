export interface AudioStreamResult {
  kind: 'audio-stream';
  playbackUrl?: string;
  playbackStream?: MediaStream;
  file?: File;
  streaming: boolean;
}

const MEDIA_SOURCE_OPEN_TIMEOUT_MS = 5000;
const PCM_BYTES_PER_SAMPLE = 2;
const PCM_PLAYBACK_LEAD_SECONDS = 0.05;

const AUDIO_MIME_TYPES: Record<string, string> = {
  flac: 'audio/flac',
  mp3: 'audio/mpeg',
  ogg: 'audio/ogg',
  pcm: 'audio/pcm',
  wav: 'audio/wav',
};

const MEDIA_SOURCE_MIME_TYPES: Record<string, string[]> = {
  flac: ['audio/flac'],
  mp3: ['audio/mpeg'],
  ogg: ['audio/ogg; codecs="vorbis"', 'audio/ogg'],
  wav: ['audio/wav; codecs="1"', 'audio/wav'],
};

function normalizeResponseFormat(responseFormat: string): string {
  return responseFormat.trim().toLowerCase() || 'mp3';
}

export function audioMimeType(responseFormat: string): string {
  const normalizedFormat = normalizeResponseFormat(responseFormat);
  return AUDIO_MIME_TYPES[normalizedFormat] || `audio/${normalizedFormat}`;
}

function abortError() {
  return new DOMException('The audio stream was aborted.', 'AbortError');
}

function waitForEvent(
  target: EventTarget,
  eventName: string,
  errorEventName: string | undefined,
  signal: AbortSignal,
  timeoutMs?: number
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError());
      return;
    }

    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    const cleanup = () => {
      target.removeEventListener(eventName, handleEvent);
      if (errorEventName) target.removeEventListener(errorEventName, handleError);
      signal.removeEventListener('abort', handleAbort);
      if (timeoutId !== undefined) clearTimeout(timeoutId);
    };
    const handleEvent = () => {
      cleanup();
      resolve();
    };
    const handleError = () => {
      cleanup();
      reject(new Error(`Failed while waiting for ${eventName}.`));
    };
    const handleAbort = () => {
      cleanup();
      reject(abortError());
    };

    target.addEventListener(eventName, handleEvent, { once: true });
    if (errorEventName) target.addEventListener(errorEventName, handleError, { once: true });
    signal.addEventListener('abort', handleAbort, { once: true });

    if (timeoutMs !== undefined) {
      timeoutId = setTimeout(() => {
        cleanup();
        reject(
          new DOMException(
            `Timed out after ${timeoutMs}ms while waiting for ${eventName}.`,
            'TimeoutError'
          )
        );
      }, timeoutMs);
    }
  });
}

async function waitForSourceBuffer(sourceBuffer: SourceBuffer, signal: AbortSignal) {
  if (!sourceBuffer.updating) return;
  await waitForEvent(sourceBuffer, 'updateend', 'error', signal);
}

function findMediaSourceMimeType(responseFormat: string): string | undefined {
  if (typeof window === 'undefined' || typeof window.MediaSource === 'undefined') {
    return undefined;
  }

  return MEDIA_SOURCE_MIME_TYPES[normalizeResponseFormat(responseFormat)]?.find((mimeType) =>
    window.MediaSource.isTypeSupported(mimeType)
  );
}

export function canPlayPcmStream(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.AudioContext?.prototype.createMediaStreamDestination === 'function' &&
    'srcObject' in window.HTMLMediaElement.prototype
  );
}

export function pcmAudioToWav(pcm: Blob, sampleRate: number): Blob {
  if (pcm.size % PCM_BYTES_PER_SAMPLE) throw new Error('Incomplete PCM audio sample.');

  const header = new DataView(new ArrayBuffer(44));
  header.setUint32(0, 0x52494646); // RIFF
  header.setUint32(4, 36 + pcm.size, true);
  header.setUint32(8, 0x57415645); // WAVE
  header.setUint32(12, 0x666d7420); // fmt chunk
  header.setUint32(16, 16, true);
  header.setUint16(20, 1, true); // Linear PCM
  header.setUint16(22, 1, true); // Mono
  header.setUint32(24, sampleRate, true);
  header.setUint32(28, sampleRate * PCM_BYTES_PER_SAMPLE, true);
  header.setUint16(32, PCM_BYTES_PER_SAMPLE, true);
  header.setUint16(34, 16, true);
  header.setUint32(36, 0x64617461); // data
  header.setUint32(40, pcm.size, true);
  return new Blob([header.buffer, pcm], { type: 'audio/wav' });
}

class PcmAudioPlayback {
  private readonly destination: MediaStreamAudioDestinationNode;
  private pendingByte?: number;
  private nextTime = 0;
  private lastSource?: AudioBufferSourceNode;
  private onFinished?: () => void;

  constructor(
    private readonly context: AudioContext,
    private readonly sampleRate: number
  ) {
    this.destination = context.createMediaStreamDestination();
    // Web Audio defaults to stereo, including when the supplied PCM is mono.
    this.destination.channelCount = 1;
  }

  get stream() {
    return this.destination.stream;
  }

  append(chunk: Uint8Array<ArrayBuffer>) {
    let bytes = chunk;
    if (this.pendingByte !== undefined) {
      bytes = new Uint8Array(chunk.byteLength + 1);
      bytes[0] = this.pendingByte;
      bytes.set(chunk, 1);
    }

    const sampleCount = Math.floor(bytes.byteLength / PCM_BYTES_PER_SAMPLE);
    const audioByteLength = sampleCount * PCM_BYTES_PER_SAMPLE;
    this.pendingByte = bytes[audioByteLength];
    if (!sampleCount) return;

    const view = new DataView(bytes.buffer, bytes.byteOffset, audioByteLength);
    const buffer = this.context.createBuffer(1, sampleCount, this.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let index = 0; index < sampleCount; index++) {
      channel[index] = view.getInt16(index * PCM_BYTES_PER_SAMPLE, true) / 32768;
    }

    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.destination);
    source.onended = () => {
      source.disconnect();
      if (this.lastSource === source) {
        this.lastSource = undefined;
        this.onFinished?.();
      }
    };
    const now = this.context.currentTime;
    if (this.nextTime <= now) this.nextTime = now + PCM_PLAYBACK_LEAD_SECONDS;
    source.start(this.nextTime);
    this.nextTime += buffer.duration;
    this.lastSource = source;
  }

  finish(onFinished: () => void) {
    if (this.lastSource) this.onFinished = onFinished;
    else onFinished();
  }

  dispose() {
    this.onFinished = undefined;
    for (const track of this.stream.getTracks()) track.stop();
    void this.context.close().catch(() => undefined);
  }
}

export class AudioStreamSession {
  private readonly abortController = new AbortController();
  private readonly mimeType: string;
  private readonly mediaSourceMimeType?: string;
  private mediaSource?: MediaSource;
  private sourceBufferPromise?: Promise<SourceBuffer>;
  private reader?: ReadableStreamDefaultReader<Uint8Array<ArrayBuffer>>;
  private playbackUrlValue?: string;
  private pcmPlayback?: PcmAudioPlayback;
  private onPlaybackChanged?: () => void;

  constructor(
    responseFormat: string,
    private readonly pcmSampleRate?: number
  ) {
    this.mimeType = audioMimeType(responseFormat);
    this.mediaSourceMimeType = findMediaSourceMimeType(responseFormat);

    if (pcmSampleRate && canPlayPcmStream()) {
      let context: AudioContext | undefined;
      try {
        context = new window.AudioContext();
        this.pcmPlayback = new PcmAudioPlayback(context, pcmSampleRate);
        void context.resume().catch(() => this.disablePcmPlayback());
      } catch {
        void context?.close().catch(() => undefined);
      }
    } else if (this.mediaSourceMimeType) {
      this.mediaSource = new MediaSource();
      this.playbackUrlValue = URL.createObjectURL(this.mediaSource);
    }
  }

  get signal() {
    return this.abortController.signal;
  }

  result(file?: File): AudioStreamResult {
    return {
      kind: 'audio-stream',
      playbackUrl: this.playbackUrlValue,
      playbackStream: this.pcmPlayback?.stream,
      file,
      streaming: !file,
    };
  }

  private async getSourceBuffer(): Promise<SourceBuffer | undefined> {
    const mediaSource = this.mediaSource;
    const mediaSourceMimeType = this.mediaSourceMimeType;
    if (!mediaSource || !mediaSourceMimeType) return undefined;

    if (!this.sourceBufferPromise) {
      this.sourceBufferPromise = (async () => {
        if (mediaSource.readyState !== 'open') {
          await waitForEvent(
            mediaSource,
            'sourceopen',
            'error',
            this.signal,
            MEDIA_SOURCE_OPEN_TIMEOUT_MS
          );
        }
        return mediaSource.addSourceBuffer(mediaSourceMimeType);
      })();
    }

    return this.sourceBufferPromise;
  }

  private disableStreamingPlayback() {
    const playbackUrl = this.playbackUrlValue;
    this.playbackUrlValue = undefined;
    this.mediaSource = undefined;
    this.sourceBufferPromise = undefined;
    if (playbackUrl) URL.revokeObjectURL(playbackUrl);
  }

  private disablePcmPlayback() {
    const playback = this.pcmPlayback;
    this.pcmPlayback = undefined;
    if (!playback) return;
    playback.dispose();
    if (!this.signal.aborted) this.onPlaybackChanged?.();
  }

  private async appendChunk(chunk: BufferSource): Promise<boolean> {
    try {
      const sourceBuffer = await this.getSourceBuffer();
      if (!sourceBuffer) return false;

      await waitForSourceBuffer(sourceBuffer, this.signal);
      sourceBuffer.appendBuffer(chunk);
      await waitForSourceBuffer(sourceBuffer, this.signal);
      return true;
    } catch (error) {
      if (this.signal.aborted) throw error;
      this.disableStreamingPlayback();
      return false;
    }
  }

  private async finishStreamingPlayback() {
    const mediaSource = this.mediaSource;
    if (!mediaSource || !this.sourceBufferPromise) return;

    try {
      const sourceBuffer = await this.sourceBufferPromise;
      await waitForSourceBuffer(sourceBuffer, this.signal);
      if (mediaSource.readyState === 'open') mediaSource.endOfStream();
    } catch (error) {
      if (this.signal.aborted) throw error;
      this.disableStreamingPlayback();
    }
  }

  async consume(
    stream: ReadableStream<Uint8Array<ArrayBuffer>>,
    onPlaybackChanged?: () => void
  ): Promise<Blob> {
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    const reader = stream.getReader();
    this.reader = reader;
    this.onPlaybackChanged = onPlaybackChanged;

    try {
      if (this.pcmSampleRate && !this.pcmPlayback) onPlaybackChanged?.();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
        if (this.pcmPlayback) {
          try {
            this.pcmPlayback.append(value);
          } catch {
            this.disablePcmPlayback();
          }
        } else if (this.mediaSource && !(await this.appendChunk(value))) {
          onPlaybackChanged?.();
        }
      }

      await this.finishStreamingPlayback();
      const audio = new Blob(chunks, { type: this.mimeType });
      if (!this.pcmSampleRate) return audio;

      const wav = pcmAudioToWav(audio, this.pcmSampleRate);
      this.pcmPlayback?.finish(() => this.disablePcmPlayback());
      return wav;
    } finally {
      if (this.reader === reader) this.reader = undefined;
      reader.releaseLock();
    }
  }

  dispose() {
    this.abortController.abort();
    const reader = this.reader;
    if (reader) void reader.cancel().catch(() => undefined);
    this.disableStreamingPlayback();
    this.disablePcmPlayback();
    this.onPlaybackChanged = undefined;
  }
}

export function isAudioStreamResult(value: unknown): value is AudioStreamResult {
  return (
    typeof value === 'object' && value !== null && 'kind' in value && value.kind === 'audio-stream'
  );
}
