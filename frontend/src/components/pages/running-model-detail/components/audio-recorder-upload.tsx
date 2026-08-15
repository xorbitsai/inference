'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, Square, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { FileUpload } from '@/components/ui/file-upload';
import { cn } from '@/lib/utils';
import type { FileUploadValue } from '@/types/common';
import type { FormInstance } from '@/types/form';

type RecorderStatus = 'idle' | 'requesting' | 'recording' | 'stopping';

interface AudioRecorderUploadProps {
  form: FormInstance;
  value?: FileUploadValue[];
  onChange?: (value: FileUploadValue[]) => void;
  error?: boolean;
  disabled?: boolean;
}

const RECORDER_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/mp4',
  'audio/webm',
  'audio/ogg;codecs=opus',
];

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function getRecorderMimeType() {
  return RECORDER_MIME_TYPES.find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
}

function getFileExtension(mimeType: string) {
  switch (mimeType.split(';', 1)[0]) {
    case 'audio/mp4':
      return 'm4a';
    case 'audio/ogg':
      return 'ogg';
    case 'audio/wav':
      return 'wav';
    default:
      return 'webm';
  }
}

function formatDuration(duration: number) {
  const minutes = Math.floor(duration / 60);
  const seconds = duration % 60;

  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function getRecordingErrorMessage(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      return 'Microphone access was denied. Allow access in your browser and try again.';
    }
    if (error.name === 'NotFoundError') {
      return 'No microphone was found.';
    }
    if (error.name === 'NotReadableError') {
      return 'The microphone is unavailable or already in use.';
    }
  }

  return 'Unable to start microphone recording.';
}

export function AudioRecorderUpload({
  form,
  value = [],
  onChange,
  error,
  disabled,
}: AudioRecorderUploadProps) {
  const [status, setStatus] = useState<RecorderStatus>('idle');
  const [duration, setDuration] = useState(0);
  const [recordingError, setRecordingError] = useState('');
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionRef = useRef(0);
  const discardRef = useRef(false);
  const mountedRef = useRef(false);
  const recordedUrlRef = useRef<string | null>(null);
  const onChangeRef = useRef(onChange);

  onChangeRef.current = onChange;

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const releaseStream = useCallback(() => {
    stopStream(streamRef.current);
    streamRef.current = null;
  }, []);

  const updateStatus = useCallback((nextStatus: RecorderStatus) => {
    if (mountedRef.current) {
      setStatus(nextStatus);
    }
  }, []);

  const showRecordingError = useCallback((message: string) => {
    if (mountedRef.current) {
      setRecordingError(message);
    }
  }, []);

  const revokeRecordedUrl = useCallback(() => {
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
    }
  }, []);

  const finishRecording = useCallback(
    (recorder: MediaRecorder) => {
      clearTimer();
      releaseStream();
      recorderRef.current = null;

      const chunks = chunksRef.current;
      chunksRef.current = [];

      if (!discardRef.current && mountedRef.current) {
        const mimeType = recorder.mimeType || chunks[0]?.type || 'audio/webm';
        const blob = new Blob(chunks, { type: mimeType });

        if (blob.size > 0) {
          const extension = getFileExtension(mimeType);
          const file = new File([blob], `recording-${Date.now()}.${extension}`, {
            type: mimeType,
          });
          const url = URL.createObjectURL(file);

          revokeRecordedUrl();
          recordedUrlRef.current = url;
          onChangeRef.current?.([{ file, type: 'audio', url }]);
          showRecordingError('');
        } else {
          showRecordingError('No audio was captured. Please try again.');
        }
      }

      discardRef.current = false;
      updateStatus('idle');
    },
    [clearTimer, releaseStream, revokeRecordedUrl, showRecordingError, updateStatus]
  );

  const cancelRecording = useCallback(() => {
    sessionRef.current += 1;
    discardRef.current = true;
    clearTimer();

    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      updateStatus('stopping');
      recorder.stop();
      releaseStream();
      return;
    }

    recorderRef.current = null;
    chunksRef.current = [];
    releaseStream();
    updateStatus('idle');
  }, [clearTimer, releaseStream, updateStatus]);

  const startRecording = async () => {
    if (disabled || status !== 'idle') return;

    if (
      typeof navigator === 'undefined' ||
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === 'undefined'
    ) {
      showRecordingError('Microphone recording is not supported in this browser or context.');
      return;
    }

    const session = sessionRef.current + 1;
    sessionRef.current = session;
    discardRef.current = false;
    setDuration(0);
    showRecordingError('');
    updateStatus('requesting');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      if (!mountedRef.current || sessionRef.current !== session) {
        stopStream(stream);
        return;
      }

      streamRef.current = stream;
      const mimeType = getRecorderMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      const startedAt = Date.now();

      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onerror = () => {
        discardRef.current = true;
        showRecordingError('Recording failed. Please try again.');
        if (recorder.state !== 'inactive') {
          recorder.stop();
        }
      };
      recorder.onstop = () => finishRecording(recorder);

      recorder.start();
      updateStatus('recording');
      timerRef.current = setInterval(() => {
        if (mountedRef.current) {
          setDuration(Math.floor((Date.now() - startedAt) / 1000));
        }
      }, 250);
    } catch (recordingException) {
      releaseStream();
      recorderRef.current = null;

      if (!mountedRef.current || sessionRef.current !== session) {
        return;
      }

      updateStatus('idle');
      showRecordingError(getRecordingErrorMessage(recordingException));
    }
  };

  const stopRecording = () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') return;

    updateStatus('stopping');
    recorder.stop();
  };

  const handleUploadChange = (nextValue: FileUploadValue[]) => {
    if (
      recordedUrlRef.current &&
      !nextValue.some((upload) => upload.url === recordedUrlRef.current)
    ) {
      revokeRecordedUrl();
    }
    onChangeRef.current?.(nextValue);
  };

  useEffect(() => {
    mountedRef.current = true;
    const unregisterReset = form.registerReset(() => {
      cancelRecording();
      revokeRecordedUrl();
      setDuration(0);
      showRecordingError('');
    });

    return () => {
      mountedRef.current = false;
      unregisterReset();
      sessionRef.current += 1;
      discardRef.current = true;
      clearTimer();

      const recorder = recorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        recorder.stop();
      }

      releaseStream();
      revokeRecordedUrl();
    };
  }, [cancelRecording, clearTimer, form, releaseStream, revokeRecordedUrl, showRecordingError]);

  const isActive = status === 'recording' || status === 'stopping';

  return (
    <div className="space-y-3">
      <FileUpload
        value={value}
        onChange={handleUploadChange}
        accept="audio/*,video/*"
        label="Upload or drop audio"
        description="MP3, WAV, M4A, WebM..."
        error={error}
        disabled={disabled || status !== 'idle'}
      />

      <div className="flex items-center gap-3 text-xs text-muted-foreground" aria-hidden="true">
        <span className="h-px flex-1 bg-border" />
        <span>or</span>
        <span className="h-px flex-1 bg-border" />
      </div>

      {isActive ? (
        <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3">
          <div className="flex items-center justify-between gap-3" aria-live="polite">
            <span className="flex items-center gap-2 text-sm font-medium">
              <span className="size-2.5 animate-pulse rounded-full bg-destructive" />
              {status === 'stopping' ? 'Finishing recording...' : 'Recording'}
            </span>
            <span className="font-mono text-sm tabular-nums">{formatDuration(duration)}</span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <Button
              type="button"
              variant="destructive"
              onClick={stopRecording}
              disabled={status === 'stopping'}
            >
              <Square className="size-3.5 fill-current" />
              Stop
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={cancelRecording}
              disabled={status === 'stopping'}
            >
              <X className="size-4" />
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Button
          type="button"
          variant="outline"
          block
          loading={status === 'requesting'}
          disabled={disabled}
          className={cn('h-10', value.length > 0 && 'text-muted-foreground')}
          onClick={startRecording}
        >
          <Mic className="size-4" />
          {status === 'requesting'
            ? 'Requesting microphone...'
            : value.length > 0
              ? 'Record again'
              : 'Record audio'}
        </Button>
      )}

      {recordingError && (
        <p className="text-xs text-destructive" role="alert">
          {recordingError}
        </p>
      )}
    </div>
  );
}
