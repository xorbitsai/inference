export interface AudioStreamResult {
  kind: 'audio-stream';
  playbackUrl?: string;
  file?: File;
  streaming: boolean;
}

const MEDIA_SOURCE_OPEN_TIMEOUT_MS = 5000;

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

export class AudioStreamSession {
  private readonly abortController = new AbortController();
  private readonly mimeType: string;
  private readonly mediaSourceMimeType?: string;
  private mediaSource?: MediaSource;
  private sourceBufferPromise?: Promise<SourceBuffer>;
  private reader?: ReadableStreamDefaultReader<Uint8Array<ArrayBuffer>>;
  private playbackUrlValue?: string;

  constructor(responseFormat: string) {
    this.mimeType = audioMimeType(responseFormat);
    this.mediaSourceMimeType = findMediaSourceMimeType(responseFormat);

    if (this.mediaSourceMimeType) {
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
    onPlaybackUnavailable?: () => void
  ): Promise<Blob> {
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    const reader = stream.getReader();
    this.reader = reader;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
        if (this.mediaSource && !(await this.appendChunk(value))) {
          onPlaybackUnavailable?.();
        }
      }

      await this.finishStreamingPlayback();
      return new Blob(chunks, { type: this.mimeType });
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
  }
}

export function isAudioStreamResult(value: unknown): value is AudioStreamResult {
  return (
    typeof value === 'object' && value !== null && 'kind' in value && value.kind === 'audio-stream'
  );
}
