'use client';

import { useState } from 'react';
import { Download, FileText, ImageIcon, Video } from 'lucide-react';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { cn } from '@/lib/utils';

export type MediaPreviewType = 'image' | 'video' | 'audio' | 'document' | string;

export interface MediaPreviewProps {
  type?: MediaPreviewType;
  url?: string;
  title?: string;
  className?: string;
}

function normalizeType(type?: MediaPreviewType) {
  if (type === 'image' || type === 'video' || type === 'audio' || type === 'document') {
    return type;
  }

  return 'document';
}

export function MediaPreview({ type, url, title, className }: MediaPreviewProps) {
  const [open, setOpen] = useState(false);
  const mediaType = normalizeType(type);

  if (!url) return null;

  if (mediaType === 'audio') {
    return (
      <audio
        src={url}
        controls
        className={cn('h-10 w-full max-w-sm', className)}
        onClick={(event) => event.stopPropagation()}
      />
    );
  }

  if (mediaType === 'document') {
    return (
      <a
        href={url}
        target="_blank"
        rel="noreferrer"
        className={cn(
          'flex w-fit max-w-sm items-center gap-2 rounded-lg border bg-background px-3 py-2 text-sm text-muted-foreground hover:text-foreground',
          className
        )}
      >
        <FileText className="size-4 shrink-0" />
        <span className="truncate">{title || 'Open file'}</span>
      </a>
    );
  }

  const Icon = mediaType === 'image' ? ImageIcon : Video;
  const label = mediaType === 'image' ? 'Image' : 'Video';

  return (
    <>
      <div
        className={cn(
          'group relative block h-32 w-48 max-w-full overflow-hidden rounded-lg border bg-background text-left shadow-sm',
          className
        )}
      >
        <button
          type="button"
          aria-label={`Open ${label}`}
          className="absolute inset-0 z-10"
          onClick={() => setOpen(true)}
        />
        <span className="pointer-events-none absolute left-0 top-0 z-20 flex items-center gap-1.5 rounded-br-md bg-background/95 px-2 py-1 text-xs font-medium text-muted-foreground shadow-sm">
          <Icon className="size-3.5" />
          {label}
        </span>
        {mediaType === 'image' ? (
          // eslint-disable-next-line @next/next/no-img-element -- Runtime media URLs can be data URLs returned by the model.
          <img src={url} alt={title || label} className="h-full w-full object-cover" />
        ) : (
          <video src={url} muted playsInline className="h-full w-full bg-black object-cover" />
        )}
        {mediaType === 'image' && (
          <a
            href={url}
            download={title || 'generated-image'}
            className="absolute right-0 top-0 z-10 flex size-7 items-center justify-center rounded-full bg-background/95 text-muted-foreground shadow-sm transition-colors hover:text-foreground"
            onClick={(event) => event.stopPropagation()}
          >
            <Download className="size-3.5" />
          </a>
        )}
        <span className="pointer-events-none absolute inset-0 bg-black/0 transition-colors group-hover:bg-black/10" />
      </div>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[calc(100vh-4rem)] max-w-[calc(100vw-4rem)] p-4">
          <DialogHeader>
            <DialogTitle>{title || label}</DialogTitle>
          </DialogHeader>
          <div className="flex min-h-0 items-center justify-center overflow-hidden rounded-lg">
            {mediaType === 'image' ? (
              <div className="relative">
                <a
                  href={url}
                  download={title || 'generated-image'}
                  className="absolute right-0 top-0 z-10 flex size-7 items-center justify-center rounded-full bg-background/95 text-muted-foreground shadow-sm transition-colors hover:text-foreground"
                >
                  <Download className="size-4" />
                </a>
                {/* eslint-disable-next-line @next/next/no-img-element -- Runtime media URLs can be data URLs returned by the model. */}
                <img
                  src={url}
                  alt={title || label}
                  className="max-h-[calc(100vh-10rem)] max-w-full object-contain"
                />
              </div>
            ) : (
              <video
                src={url}
                controls
                autoPlay
                className="max-h-[calc(100vh-10rem)] max-w-full bg-black"
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
