import { clsx, type ClassValue } from "clsx"
import { FileAudio, FileText, FileVideo, ImageIcon } from 'lucide-react';
import { twMerge } from "tailwind-merge"
import { toast } from 'sonner'; 
import type { BaseFormListValueItem } from '@/types/common'; 

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getApiUrl(): string {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || ''
  return apiUrl
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null

  return (...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout)
    }

    timeout = setTimeout(() => {
      func(...args)
    }, wait)
  }
}
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
  }
}

export function sleep(ms: number): Promise<void>{
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
};

/** transform formValue example: 'true' to true */
export const transformValueType = (str: any) => {
  const value = String(str).trim();
  if (value === '') return '';
  if (value.toLowerCase() === 'none') return null;
  if (value.toLowerCase() === 'true') return true;
  if (value.toLowerCase() === 'false') return false;
  if (!isNaN(Number(value))) return Number(value);
  return value;
};
/** transform [{key: 'min', value: 1}] to { min: 1} */
export const transformFormListToObj = (formList: unknown = [], transformValue = true) => {
  if (!Array.isArray(formList)) {
    return {};
  }

  return formList.reduce((acc, item) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      return acc;
    }

    const { key, value } = item as Partial<BaseFormListValueItem>;
    const normalizedKey = typeof key === 'string' ? key.trim() : '';

    if (normalizedKey) {
      acc[normalizedKey] = transformValue ? transformValueType(value) : value;
    }

    return acc;
  }, {} as Record<string, any>);
};
/** transform { abc: 123 } to [{ key: 'abc', value: '123'}] */
export const transformObjToFormList = (obj: Record<string, any> = {}) => {
  return Object.entries(obj || {}).map(([key, value]) => ({
    key,
    value: String(value),
  }));
};
export const copyText = async (value: string) => {
  if (!value) return;

  if (!navigator.clipboard) {
    toast.error('Clipboard API not supported');
    return;
  }
  try {
    await navigator.clipboard.writeText(value);
    toast.success('Copy successful!');
  } catch {
    toast.error('Failed to copy');
  }
};

export function getFileMeta(file: File) {
  if (file.type.startsWith('image/')) {
    return { label: 'Image', icon: ImageIcon, kind: 'image' as const };
  }

  if (file.type.startsWith('video/')) {
    return { label: 'Video', icon: FileVideo, kind: 'video' as const };
  }

  if (file.type.startsWith('audio/')) {
    return { label: 'Audio', icon: FileAudio, kind: 'audio' as const };
  }

  return { label: 'Document', icon: FileText, kind: 'document' as const };
}