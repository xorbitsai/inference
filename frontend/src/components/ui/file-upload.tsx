'use client';

import { useRef, useState } from 'react';
import type { RefObject, ReactNode, SyntheticEvent } from 'react';
import { Upload, X } from 'lucide-react';

import { cn, getFileMeta } from '@/lib/utils';
import type { FileUploadValue } from '@/types/common';


export interface FileUploadProps {
  value?: FileUploadValue[];
  onChange?: (value: FileUploadValue[]) => void;
  accept?: string;
  label?: ReactNode;
  description?: ReactNode;
  icon?: ReactNode;
  children?: ReactNode;
  showResult?: boolean;
  error?: boolean;
  disabled?: boolean;
  className?: string;
}

export interface FileUploadResultProps {
  fileList?: FileUploadValue[];
  onRemove?: (index: number) => void;
  className?: string;
}

function formatFileSize(size: number) {
  if (!size) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(size) / Math.log(1024)), units.length - 1);

  return `${(size / 1024 ** index).toFixed(index === 0 ? 0 : 2)} ${units[index]}`;
}

async function toUploadValue(file: File): Promise<FileUploadValue> {
  return {
    file,
    type: getFileMeta(file).kind,
    url: URL.createObjectURL(file),
  };
}



function HiddenInput({
  accept,
  disabled,
  inputRef,
  onFiles,
}: {
  accept?: string;
  disabled?: boolean;
  inputRef?: RefObject<HTMLInputElement | null>;
  onFiles: (files: File[]) => void;
}) {
  return (
    <input
      ref={inputRef}
      className="sr-only"
      type="file"
      accept={accept}
      disabled={disabled}
      onChange={(event) => {
        onFiles(Array.from(event.target.files || []));
        event.currentTarget.value = '';
      }}
    />
  );
}

export function FileUploadResult({ fileList = [], onRemove, className }: FileUploadResultProps) {
  if (!fileList.length) return null;

  return (
    <div className={cn('grid min-w-0 gap-2', className)}>
      {fileList.map((item, index) => {
        const meta = getFileMeta(item.file);
        const Icon = meta.icon;
        const isImage = meta.kind === 'image';

        return (
          <div
            key={`${item.file.name}-${index}`}
            className="flex min-w-0 max-w-full items-center gap-3 rounded-xl border bg-background p-2 text-sm shadow-sm"
          >
            <div className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-muted">
              {isImage ? (
                // eslint-disable-next-line @next/next/no-img-element -- Local object/data URLs are not suitable for next/image.
                <img src={item.url} alt={item.file.name} className="h-full w-full object-contain" />
              ) : (
                <Icon className="size-5 text-muted-foreground" />
              )}
            </div>

            <div className="min-w-0 flex-1">
              <div className="max-w-full truncate font-medium">{item.file.name}</div>
              <div className="max-w-full truncate text-xs text-muted-foreground">
                {item.file.type || 'file'} · {formatFileSize(item.file.size)}
              </div>
            </div>

            {onRemove && (
              <button
                type="button"
                className="flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
                onClick={() => onRemove(index)}
              >
                <X className="size-4" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

function stopUploadTrigger(event: SyntheticEvent) {
  event.preventDefault();
  event.stopPropagation();
}

function FileUploadFrameResult({
  item,
  onRemove,
}: {
  item: FileUploadValue;
  onRemove?: () => void;
}) {
  const meta = getFileMeta(item.file);
  const Icon = meta.icon;

  return (
    <div className="absolute inset-0 overflow-hidden rounded-md" onClick={stopUploadTrigger}>
      <div className="absolute left-0 top-0 z-10 flex items-center gap-2 rounded-br-md bg-background/95 px-2 py-1 text-sm font-medium text-muted-foreground shadow-sm">
        <Icon className="size-4" />
        {meta.label}
      </div>

      {onRemove && (
        <button
          type="button"
          className="absolute right-0 top-0 z-10 flex size-7 items-center justify-center rounded-bl-md bg-background/95 text-muted-foreground shadow-sm transition-colors hover:text-foreground"
          onClick={(event) => {
            stopUploadTrigger(event);
            onRemove();
          }}
        >
          <X className="size-5" />
        </button>
      )}

      {meta.kind === 'image' && (
        // eslint-disable-next-line @next/next/no-img-element -- Local object/data URLs are not suitable for next/image.
        <img src={item.url} alt={item.file.name} className="h-full w-full object-contain" />
      )}

      {meta.kind === 'video' && (
        <video
          src={item.url}
          controls
          className="h-full w-full bg-black object-cover"
          onClick={stopUploadTrigger}
        />
      )}

      {meta.kind === 'audio' && (
        <div className="flex h-full flex-col justify-end gap-3 bg-muted/30 p-2">
          <div className="min-w-0 text-left">
            <div className="truncate text-sm font-medium">{item.file.name}</div>
          </div>
          <audio src={item.url} controls className="w-full" onClick={stopUploadTrigger} />
        </div>
      )}

      {meta.kind === 'document' && (
        <div className="flex h-full flex-col items-center justify-center bg-muted/30 px-2 text-center">
          <span className="mb-3 flex size-12 items-center justify-center rounded-xl bg-background text-primary shadow-sm">
            <Icon className="size-6" />
          </span>
          <div className="max-w-full truncate text-sm font-medium">{item.file.name}</div>
          <div className="mt-1 text-xs text-muted-foreground">{formatFileSize(item.file.size)}</div>
        </div>
      )}
    </div>
  );
}

export function FileUpload({
  value = [],
  onChange,
  accept,
  label = 'Upload file',
  description = 'Drag a file here or click to browse',
  icon,
  children,
  showResult = true,
  error,
  disabled,
  className,
}: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const updateFiles = async (files: File[]) => {
    const file = files[0];
    if (!file || disabled) return;

    onChange?.([await toUploadValue(file)]);
  };

  const removeFile = (index: number) => {
    onChange?.(value.filter((_, currentIndex) => currentIndex !== index));
  };
  const uploadedFile = showResult ? value[0] : undefined;
  const openFileDialog = (event: SyntheticEvent) => {
    event.preventDefault();
    event.stopPropagation();

    if (disabled) return;

    inputRef.current?.click();
  };

  if (children) {
    return (
      <div className={className}>
        <HiddenInput
          accept={accept}
          disabled={disabled}
          inputRef={inputRef}
          onFiles={updateFiles}
        />
        <span
          role="button"
          tabIndex={disabled ? -1 : 0}
          className={cn('inline-flex cursor-pointer', disabled && 'cursor-not-allowed opacity-50')}
          onClick={openFileDialog}
        >
          {children}
        </span>
        {showResult && (
          <FileUploadResult fileList={value} onRemove={disabled ? undefined : removeFile} />
        )}
      </div>
    );
  }

  return (
    <div className={className}>
      <label
        className={cn(
          'group relative min-h-36 px-6 py-6 flex cursor-pointer flex-col items-center justify-center overflow-hidden rounded-md border border-dashed bg-muted/20 text-center transition-all hover:border-primary/50 hover:bg-primary/5',
          dragging && 'border-primary bg-primary/10',
          error && 'border-destructive bg-destructive/5 hover:border-destructive',
          disabled && 'cursor-not-allowed opacity-50 hover:border-border hover:bg-muted/20'
        )}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          updateFiles(Array.from(event.dataTransfer.files || []));
        }}
      >
        <HiddenInput accept={accept} disabled={disabled} onFiles={updateFiles} />
        {uploadedFile ? (
          <FileUploadFrameResult
            item={uploadedFile}
            onRemove={disabled ? undefined : () => removeFile(0)}
          />
        ) : (
          <>
            {icon || (
              <span className="mb-3 flex size-11 items-center justify-center rounded-xl bg-background text-primary shadow-sm ring-1 ring-border transition-transform group-hover:scale-105">
                <Upload className="size-5" />
              </span>
            )}
            <span className="text-sm font-medium">{label}</span>
            <span className="mt-1 text-xs text-muted-foreground">{description}</span>
          </>
        )}
      </label>
    </div>
  );
}
