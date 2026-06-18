'use client';

import { type MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Brush, Eraser, Trash2, Undo2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { FileUpload } from '@/components/ui/file-upload';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Slider } from '@/components/ui/slider';
import { cn } from '@/lib/utils';
import type { FileUploadValue } from '@/types/common';

interface Point {
  x: number;
  y: number;
}

interface Stroke {
  points: Point[];
  color: string;
  width: number;
}

interface ImageEditorCreateMaskProps {
  value?: FileUploadValue[];
  onChange?: (value: FileUploadValue[]) => void;
  updateMask: (value: FileUploadValue[]) => void;
  error?: boolean;
  disabled?: boolean;
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();

    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Failed to load image'));
    image.src = src;
  });
}

function canvasToBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error('Failed to create mask image'));
      }
    }, 'image/png');
  });
}

async function canvasToUploadValue(
  canvas: HTMLCanvasElement,
  fileName = `mask_${Date.now()}.png`
): Promise<FileUploadValue> {
  const blob = await canvasToBlob(canvas);

  return {
    file: new File([blob], fileName, { type: 'image/png' }),
    type: 'image',
    url: canvas.toDataURL('image/png'),
  };
}

function imageSize(image: HTMLImageElement) {
  return {
    width: image.naturalWidth || image.width,
    height: image.naturalHeight || image.height,
  };
}

function getScaleForContainer(
  image: HTMLImageElement,
  container: HTMLDivElement | null,
  maxHeight: number
) {
  if (!container) {
    return 1;
  }

  const { width, height } = imageSize(image);
  const maxWidth = container.offsetWidth > 0 ? container.offsetWidth * 0.86 : width;

  return Math.min(1, maxWidth / width, maxHeight / height);
}

function strokePath(
  context: CanvasRenderingContext2D,
  points: Point[],
  color: string,
  width: number
) {
  if (!points.length) {
    return;
  }

  context.beginPath();
  context.strokeStyle = color;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.lineWidth = width;

  points.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(point.x, point.y);
    } else {
      context.lineTo(point.x, point.y);
    }
  });

  context.stroke();
}

export function ImageEditorCreateMask({
  value = [],
  onChange,
  updateMask,
  error,
  disabled,
}: ImageEditorCreateMaskProps) {
  const sourceValue = value[0];
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewContextRef = useRef<CanvasRenderingContext2D | null>(null);
  const previewContainerRef = useRef<HTMLDivElement | null>(null);
  const sourceValueRef = useRef<FileUploadValue | undefined>(sourceValue);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const uploadedMaskRef = useRef<FileUploadValue | undefined>(undefined);
  const strokesRef = useRef<Stroke[]>([]);
  const currentStrokeRef = useRef<Point[]>([]);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [uploadedMaskValue, setUploadedMaskValue] = useState<FileUploadValue[]>([]);
  const [drawing, setDrawing] = useState(false);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [lineWidth, setLineWidth] = useState(35);
  const [lineColor, setLineColor] = useState('#000000');

  const uploadedMask = uploadedMaskValue[0];
  const sourceKey = sourceValue
    ? `${sourceValue.file.name}:${sourceValue.file.size}:${sourceValue.file.lastModified}:${sourceValue.url}`
    : '';

  const brushStyle = useMemo(
    () => ({
      backgroundColor: lineColor,
    }),
    [lineColor]
  );

  const syncPreviewMask = useCallback(async () => {
    const canvas = previewCanvasRef.current;

    if (!canvas) {
      updateMask([]);
      return;
    }

    const maskValue = await canvasToUploadValue(canvas);

    updateMask([maskValue]);
  }, [updateMask]);

  const setupCanvasSize = useCallback(
    (
      canvas: HTMLCanvasElement,
      source: HTMLImageElement,
      container: HTMLDivElement | null,
      maxHeight: number
    ) => {
      const { width, height } = imageSize(source);
      const scale = getScaleForContainer(source, container, maxHeight);

      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${width * scale}px`;
      canvas.style.height = `${height * scale}px`;
    },
    []
  );

  const initEditorCanvas = useCallback(
    (source: HTMLImageElement) => {
      const canvas = canvasRef.current;
      const context = canvas?.getContext('2d');

      if (!canvas || !context) {
        return;
      }

      setupCanvasSize(canvas, source, containerRef.current, 280);
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(source, 0, 0);
      context.lineCap = 'round';
      context.lineJoin = 'round';
      contextRef.current = context;
    },
    [setupCanvasSize]
  );

  const redrawEditorCanvas = useCallback((nextStrokes: Stroke[]) => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    const source = imageRef.current;

    if (!canvas || !context || !source) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, 0, 0);

    nextStrokes.forEach((stroke) => strokePath(context, stroke.points, stroke.color, stroke.width));
  }, []);

  const drawPreviewFromStrokes = useCallback(
    async (nextStrokes: Stroke[], source = imageRef.current) => {
      const canvas = previewCanvasRef.current;
      const context = canvas?.getContext('2d');

      if (!canvas || !context || !source) {
        updateMask([]);
        return;
      }

      if (canvas.width !== source.width || canvas.height !== source.height) {
        setupCanvasSize(canvas, source, previewContainerRef.current, 180);
      }

      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = '#000000';
      context.fillRect(0, 0, canvas.width, canvas.height);

      nextStrokes.forEach((stroke) => strokePath(context, stroke.points, '#ffffff', stroke.width));

      previewContextRef.current = context;
      await syncPreviewMask();
    },
    [setupCanvasSize, syncPreviewMask, updateMask]
  );

  const drawUploadedMask = useCallback(
    async (maskValue: FileUploadValue) => {
      const canvas = previewCanvasRef.current;
      const context = canvas?.getContext('2d');

      if (!canvas || !context) {
        updateMask([]);
        return;
      }

      const mask = await loadImage(maskValue.url);
      const maskSize = imageSize(mask);
      const sourceCanvas = document.createElement('canvas');
      const sourceContext = sourceCanvas.getContext('2d');

      if (!sourceContext) {
        updateMask([]);
        return;
      }

      setupCanvasSize(canvas, mask, previewContainerRef.current, 180);
      sourceCanvas.width = maskSize.width;
      sourceCanvas.height = maskSize.height;
      sourceContext.drawImage(mask, 0, 0, maskSize.width, maskSize.height);

      const maskData = sourceContext.getImageData(0, 0, maskSize.width, maskSize.height);
      const pixels = maskData.data;

      for (let index = 0; index < pixels.length; index += 4) {
        const alpha = pixels[index + 3];
        const grayscale =
          alpha < 255
            ? alpha
            : Math.round(
                0.299 * pixels[index] + 0.587 * pixels[index + 1] + 0.114 * pixels[index + 2]
              );

        pixels[index] = grayscale;
        pixels[index + 1] = grayscale;
        pixels[index + 2] = grayscale;
        pixels[index + 3] = 255;
      }

      sourceContext.putImageData(maskData, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(sourceCanvas, 0, 0, maskSize.width, maskSize.height);
      previewContextRef.current = context;
      await syncPreviewMask();
    },
    [setupCanvasSize, syncPreviewMask, updateMask]
  );

  const clearPreview = useCallback(() => {
    const canvas = previewCanvasRef.current;
    const context = canvas?.getContext('2d');

    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      previewContextRef.current = context;
    }

    updateMask([]);
  }, [updateMask]);

  const getCanvasPoint = (event: MouseEvent<HTMLCanvasElement>): Point | undefined => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return undefined;
    }

    const rect = canvas.getBoundingClientRect();
    const scale = rect.width ? canvas.width / rect.width : 1;

    return {
      x: (event.clientX - rect.left) * scale,
      y: (event.clientY - rect.top) * scale,
    };
  };

  const startDrawing = (event: MouseEvent<HTMLCanvasElement>) => {
    const context = contextRef.current;
    const point = getCanvasPoint(event);

    if (!image || !context || !point || disabled) {
      return;
    }

    setDrawing(true);
    currentStrokeRef.current = [point];

    context.beginPath();
    context.strokeStyle = lineColor;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.lineWidth = lineWidth;
    context.moveTo(point.x, point.y);
  };

  const draw = (event: MouseEvent<HTMLCanvasElement>) => {
    const context = contextRef.current;
    const point = getCanvasPoint(event);

    if (!drawing || !context || !point) {
      return;
    }

    context.lineTo(point.x, point.y);
    context.stroke();
    currentStrokeRef.current = [...currentStrokeRef.current, point];
  };

  const stopDrawing = () => {
    if (!drawing) {
      return;
    }

    setDrawing(false);

    const finishedStroke = currentStrokeRef.current;

    if (!finishedStroke.length) {
      return;
    }

    const nextStrokes = [
      ...strokesRef.current,
      {
        points: finishedStroke,
        color: lineColor,
        width: lineWidth,
      },
    ];

    strokesRef.current = nextStrokes;
    currentStrokeRef.current = [];
    setStrokes(nextStrokes);
    redrawEditorCanvas(nextStrokes);
    void drawPreviewFromStrokes(nextStrokes);
  };

  const undo = () => {
    const nextStrokes = strokesRef.current.slice(0, -1);

    strokesRef.current = nextStrokes;
    setStrokes(nextStrokes);
    redrawEditorCanvas(nextStrokes);
    void drawPreviewFromStrokes(nextStrokes);
  };

  const clearStrokes = () => {
    strokesRef.current = [];
    currentStrokeRef.current = [];
    setStrokes([]);
    redrawEditorCanvas([]);
    void drawPreviewFromStrokes([]);
  };

  const resetSource = () => {
    onChange?.([]);
    imageRef.current = null;
    strokesRef.current = [];
    currentStrokeRef.current = [];
    setImage(null);
    setStrokes([]);

    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');

    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      contextRef.current = context;
    }
  };

  const handleMaskChange = (nextValue: FileUploadValue[]) => {
    setUploadedMaskValue(nextValue);
  };

  useEffect(() => {
    sourceValueRef.current = sourceValue;
  }, [sourceValue]);

  useEffect(() => {
    uploadedMaskRef.current = uploadedMask;
  }, [uploadedMask]);

  useEffect(() => {
    const resizeCanvases = () => {
      const source = imageRef.current;
      const editorCanvas = canvasRef.current;

      if (!source || !editorCanvas) {
        return;
      }

      setupCanvasSize(editorCanvas, source, containerRef.current, 280);
      redrawEditorCanvas(strokesRef.current);

      if (uploadedMaskRef.current) {
        void drawUploadedMask(uploadedMaskRef.current);
      } else {
        void drawPreviewFromStrokes(strokesRef.current, source);
      }
    };

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', resizeCanvases);
      return () => window.removeEventListener('resize', resizeCanvases);
    }

    const observer = new ResizeObserver(resizeCanvases);
    if (containerRef.current) observer.observe(containerRef.current);
    if (previewContainerRef.current) observer.observe(previewContainerRef.current);

    return () => observer.disconnect();
  }, [drawPreviewFromStrokes, drawUploadedMask, redrawEditorCanvas, setupCanvasSize, sourceKey]);

  useEffect(() => {
    let isCurrent = true;

    const syncSourceImage = async () => {
      const currentSourceValue = sourceValueRef.current;

      if (!currentSourceValue) {
        imageRef.current = null;
        strokesRef.current = [];
        currentStrokeRef.current = [];
        setImage(null);
        setStrokes([]);

        if (uploadedMaskRef.current) {
          await drawUploadedMask(uploadedMaskRef.current);
        } else {
          clearPreview();
        }

        return;
      }

      const nextImage = await loadImage(currentSourceValue.url);

      if (!isCurrent) {
        return;
      }

      setImage(nextImage);
      imageRef.current = nextImage;
      strokesRef.current = [];
      currentStrokeRef.current = [];
      setStrokes([]);
      initEditorCanvas(nextImage);

      if (uploadedMaskRef.current) {
        await drawUploadedMask(uploadedMaskRef.current);
      } else {
        await drawPreviewFromStrokes([], nextImage);
      }
    };

    void syncSourceImage();

    return () => {
      isCurrent = false;
    };
  }, [clearPreview, drawPreviewFromStrokes, drawUploadedMask, initEditorCanvas, sourceKey]);

  useEffect(() => {
    if (!uploadedMask) {
      if (imageRef.current) {
        void drawPreviewFromStrokes(strokes);
      } else {
        clearPreview();
      }

      return;
    }

    void drawUploadedMask(uploadedMask);
  }, [clearPreview, drawPreviewFromStrokes, drawUploadedMask, strokes, uploadedMask]);

  return (
    <div className="space-y-3">
      <div className="h-[300px]">
        {sourceValue ? (
          <div
            ref={containerRef}
            className={cn(
              'relative flex h-full w-full items-center justify-center overflow-hidden rounded-md border border-dashed bg-muted/20',
              error && 'border-destructive bg-destructive/5'
            )}
          >
            <canvas
              ref={canvasRef}
              className="max-h-full max-w-full"
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              style={{ cursor: disabled ? 'not-allowed' : 'crosshair' }}
            />
            <div className="absolute left-3 top-3 flex items-center gap-1 rounded-md border bg-background/95 p-1 shadow-sm">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-8"
                aria-label="Undo stroke"
                disabled={disabled || strokes.length === 0}
                onClick={undo}
              >
                <Undo2 className="size-4" />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-8"
                aria-label="Clear strokes"
                disabled={disabled || strokes.length === 0}
                onClick={clearStrokes}
              >
                <Eraser className="size-4" />
              </Button>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="size-8"
                    aria-label="Brush size"
                    disabled={disabled}
                  >
                    <Brush className="size-4" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-56">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">Brush size</span>
                      <span className="text-muted-foreground">{lineWidth}px</span>
                    </div>
                    <Slider
                      min={1}
                      max={50}
                      step={1}
                      value={[lineWidth]}
                      onValueChange={(nextValue) => setLineWidth(nextValue[0] ?? lineWidth)}
                    />
                  </div>
                </PopoverContent>
              </Popover>
              <label className="flex size-8 items-center justify-center rounded-md hover:bg-accent">
                <span className="sr-only">Brush color</span>
                <span
                  className="size-4 rounded-full border border-border shadow-sm"
                  style={brushStyle}
                />
                <input
                  type="color"
                  value={lineColor}
                  disabled={disabled}
                  className="sr-only"
                  onChange={(event) => setLineColor(event.target.value)}
                />
              </label>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-8 text-destructive hover:text-destructive"
                aria-label="Remove source image"
                disabled={disabled}
                onClick={resetSource}
              >
                <Trash2 className="size-4" />
              </Button>
            </div>
          </div>
        ) : (
          <FileUpload
            accept="image/*"
            value={[]}
            onChange={(nextValue) => onChange?.(nextValue)}
            label="Upload image"
            description="Edit Image and Create Mask"
            error={error}
            disabled={disabled}
            className="h-full [&>label]:h-full"
          />
        )}
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="space-y-1.5">
          <label className="text-sm font-medium">Or Upload Mask image</label>
          <FileUpload
            accept="image/*"
            value={uploadedMaskValue}
            onChange={handleMaskChange}
            label="Mask image"
            description="Optional mask image"
            disabled={disabled}
            className="[&>label]:h-[200px]"
          />
        </div>
        <div className="space-y-1.5">
          <label className="text-sm font-medium">Current Mask Preview</label>
          <div
            ref={previewContainerRef}
            className="relative flex h-[200px] items-center justify-center overflow-hidden rounded-md border border-dashed bg-muted/20 p-2"
          >
            <canvas ref={previewCanvasRef} className="max-h-full max-w-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
