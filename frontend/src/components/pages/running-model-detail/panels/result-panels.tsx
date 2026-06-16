'use client';

import { useEffect, useMemo } from 'react';
import { ImageIcon, Sparkles } from 'lucide-react';

import ReactMarkdown from '@/components/ui/markdown-renderer';
import { MediaPreview } from '@/components/ui/media-preview';
import { Progress } from '@/components/ui/progress';
import type { FormValues } from '@/types/form';
import { ModelAbility } from '@/constants';
import { cn } from '@/lib/utils';
import { isNumber } from '@/lib/is';

import type { CapabilityResultProps } from '../types';
import { booleanValue, isRecord, stringValue } from '../utils';

type MediaType = 'image' | 'video' | 'audio';

interface MediaResult {
  url: string;
  title?: string;
}

interface CompletionChoice {
  text?: string;
  index?: number;
  logprobs?: unknown;
  finish_reason?: string | null;
  [key: string]: unknown;
}

interface CompletionUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  [key: string]: unknown;
}

interface CompletionResult {
  id?: string;
  object?: string;
  created?: number;
  model?: string;
  choices: CompletionChoice[];
  usage?: CompletionUsage;
  [key: string]: unknown;
}

interface AudioToTextResult {
  text: string;
  [key: string]: unknown;
}

function formatProgress(progress: number) {
  return Number.isInteger(progress) ? String(progress) : progress.toFixed(1);
}

function EmptyResult({ label = 'Results will appear here.' }: { label?: string }) {
  return (
    <div className="flex min-h-80 flex-col items-center justify-center rounded-2xl border border-dashed bg-muted/20 text-center text-sm text-muted-foreground">
      <ImageIcon className="mb-3 size-8" />
      {label}
    </div>
  );
}

function Generating({ progress }: { progress?: number }) {
  const hasProgress = progress !== undefined;

  return (
    <div className="flex min-h-80 flex-col items-center justify-center gap-4 px-8">
      <div className="relative">
        <div className="flex size-20 items-center justify-center rounded-full bg-primary/10">
          <Sparkles className="size-8 animate-pulse text-primary" />
        </div>
        <div
          className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-primary/40"
          style={{ animationDuration: '2s' }}
        />
        <div
          className="absolute -inset-2 animate-spin rounded-full border-2 border-transparent border-b-primary/20"
          style={{ animationDirection: 'reverse', animationDuration: '3s' }}
        />
      </div>
      <div className="flex items-center gap-2 text-sm font-medium">
        Generating
        <div className="flex gap-1">
          <span className="size-1.5 animate-bounce rounded-full bg-primary/60" />
          <span
            className="size-1.5 animate-bounce rounded-full bg-primary/60"
            style={{ animationDelay: '150ms' }}
          />
          <span
            className="size-1.5 animate-bounce rounded-full bg-primary/60"
            style={{ animationDelay: '300ms' }}
          />
        </div>
      </div>
      {hasProgress && (
        <div className="w-full max-w-xs space-y-1">
          <Progress value={progress} />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Progress</span>
            <span className="font-mono">{formatProgress(progress)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

function isVisualAbility(ability: ModelAbility) {
  return [
    ModelAbility.Text2image,
    ModelAbility.Image2image,
    ModelAbility.Inpainting,
    ModelAbility.Text2video,
    ModelAbility.Image2video,
    ModelAbility.Firstlastframe2video,
  ].includes(ability);
}

function isAudioResultAbility(ability: ModelAbility) {
  return [ModelAbility.Text2audio, ModelAbility.Audio2audio, ModelAbility.Audio2audio].includes(
    ability
  );
}

function mediaTypeForAbility(ability: ModelAbility): MediaType {
  if (isAudioResultAbility(ability)) return 'audio';

  if (
    [ModelAbility.Text2video, ModelAbility.Image2video, ModelAbility.Firstlastframe2video].includes(
      ability
    )
  ) {
    return 'video';
  }

  return 'image';
}

function mediaMimeType(type: MediaType) {
  if (type === 'audio') return 'audio/mpeg';
  if (type === 'video') return 'video/mp4';
  return 'image/png';
}

function getResponseDataItems(result: unknown) {
  if (!isRecord(result) || !Array.isArray(result.data)) {
    return [];
  }

  return result.data;
}

function extractMediaResults(result: unknown, type: MediaType): MediaResult[] {
  if (typeof result === 'string') {
    return [{ url: result }];
  }

  if (isRecord(result) && typeof result.url === 'string') {
    return [{ url: result.url }];
  }

  return getResponseDataItems(result)
    .map((item, index): MediaResult | null => {
      if (!isRecord(item)) return null;

      if (typeof item.url === 'string') {
        return { url: item.url, title: `${type}-${index + 1}` };
      }

      if (typeof item.b64_json === 'string') {
        return {
          url: `data:${mediaMimeType(type)};base64,${item.b64_json}`,
          title: `${type}-${index + 1}`,
        };
      }

      return null;
    })
    .filter((item): item is MediaResult => item !== null);
}

function BlobMediaPreview({
  blob,
  type,
  className,
}: {
  blob: Blob;
  type: MediaType;
  className: string;
}) {
  const url = useMemo(() => URL.createObjectURL(blob), [blob]);

  useEffect(() => {
    return () => URL.revokeObjectURL(url);
  }, [url]);

  return <MediaPreview type={type} url={url} className={className} />;
}

function MediaResultPanel({ result, type }: { result: unknown; type: MediaType }) {
  const isAudio = type === 'audio';
  const mediaClassName = isAudio ? 'w-full max-w-full' : 'h-72 w-full max-w-full';
  if (result instanceof Blob) {
    return <BlobMediaPreview blob={result} type={type} className={mediaClassName} />;
  }

  const mediaResults = extractMediaResults(result, type);

  if (!mediaResults.length) {
    return null;
  }

  return (
    <div className={cn('grid gap-2', !isAudio && 'grid-cols-2')}>
      {mediaResults.map((item, index) => (
        <MediaPreview
          key={item.title || index}
          type={type}
          url={item.url}
          title={item.title}
          className={mediaClassName}
        />
      ))}
    </div>
  );
}

function JsonBlock({ value }: { value: unknown }) {
  return (
    <pre className="overflow-x-auto whitespace-pre-wrap text-xs leading-5">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function TextBlock({ children }: { children: string }) {
  return <pre className="whitespace-pre-wrap text-sm leading-6">{children}</pre>;
}

function RawResultPanel({ result }: { result: unknown }) {
  if (typeof result === 'string') {
    return <TextBlock>{result}</TextBlock>;
  }

  return <JsonBlock value={result} />;
}

function CompletionResultPanel({ result }: { result: CompletionResult }) {
  const firstChoice = result.choices[0];
  const text = stringValue(firstChoice?.text);
  const usage = result.usage;

  if (!text) return null;

  return (
    <div className="flex flex-col justify-between gap-2 relative">
      <ReactMarkdown>{text}</ReactMarkdown>
      <div className="shrink-0 flex flex-col items-end text-muted-foreground">
        {isNumber(usage?.total_tokens) && (
          <div>
            {String(usage.prompt_tokens ?? '-')} → {String(usage.completion_tokens ?? '-')} (∑
            {String(usage.total_tokens)})
          </div>
        )}
      </div>
    </div>
  );
}

function formatOcrText(result: unknown, values?: FormValues) {
  const ocrType = stringValue(values?.ocr_type, 'ocr');
  let text = '';
  let renderAs: 'plain' | 'markdown' = ocrType === 'markdown' ? 'markdown' : 'plain';

  if (isRecord(result)) {
    if (result.success === false) {
      return {
        text: `**Error**: ${stringValue(result.error, 'OCR failed')}`,
        renderAs: 'markdown' as const,
      };
    }

    text = stringValue(result.text, 'No text extracted');

    if (!text.trim()) {
      text = `**OCR Recognition Complete, No Text Detected**

**Possible Reasons:**
- Text in image is unclear or insufficient resolution
- Image format not supported
- Model unable to recognize text in image

**Suggestions:**
- Try uploading a clearer image
- Ensure text in image is clear and legible
- Handwritten text may have poor results`;
      renderAs = 'markdown';
    }

    if (booleanValue(values?.test_compress) && result.compression_ratio !== undefined) {
      text += '\n\n--- Compression Ratio Information ---\n';
      text += `Compression Ratio: ${String(result.compression_ratio ?? 'N/A')}\n`;
      text += `Valid Image Tokens: ${String(result.valid_image_tokens ?? 'N/A')}\n`;
      text += `Output Text Tokens: ${String(result.output_text_tokens ?? 'N/A')}`;
    }
  } else if (typeof result === 'string') {
    text = result;
  } else {
    text = String(result);
  }

  if (ocrType === 'format' && text.includes('<|ref|>')) {
    return {
      text: `\`\`\`text\n${text}\n\`\`\``,
      renderAs: 'markdown' as const,
    };
  }

  return { text, renderAs };
}

function OcrResultPanel({ result, values }: { result: unknown; values?: FormValues }) {
  const { text, renderAs } = formatOcrText(result, values);

  return renderAs === 'markdown' ? (
    <ReactMarkdown parseHtml>{text}</ReactMarkdown>
  ) : (
    <TextBlock>{text}</TextBlock>
  );
}

function DocumentResultPanel({ result, values }: { result: unknown; values?: FormValues }) {
  if (!isRecord(result)) {
    return <RawResultPanel result={result} />;
  }

  const content = result.markdown || result.text || result.content;

  if (typeof content !== 'string') {
    return <RawResultPanel result={result} />;
  }

  if (stringValue(values?.output_format, 'markdown') === 'markdown') {
    return <ReactMarkdown parseHtml>{content}</ReactMarkdown>;
  }

  return <TextBlock>{content}</TextBlock>;
}

function AudioToTextResultPanel({ result }: { result: AudioToTextResult }) {
  return <div className="grid gap-4 overflow-hidden">{result.text}</div>;
}

export function UniversalResultPanel({
  result,
  values,
  loading,
  progress,
  ability,
}: CapabilityResultProps) {
  if (loading) {
    return <Generating progress={progress} />;
  }

  if (result === undefined) {
    return <EmptyResult />;
  }

  if (isVisualAbility(ability) || isAudioResultAbility(ability)) {
    return <MediaResultPanel result={result} type={mediaTypeForAbility(ability)} />;
  }

  if (ability === ModelAbility.Generate) {
    return <CompletionResultPanel result={result as CompletionResult} />;
  }

  if (ability === ModelAbility.Ocr) {
    return <OcrResultPanel result={result} values={values} />;
  }

  if (ability === ModelAbility.Docanalyze) {
    return <DocumentResultPanel result={result} values={values} />;
  }

  if (ability === ModelAbility.Audio2text) {
    return <AudioToTextResultPanel result={result as AudioToTextResult} />;
  }

  return <RawResultPanel result={result} />;
}

export const ResultPanels = {
  Universal: UniversalResultPanel,
};
