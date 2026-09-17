import type { TFunc } from '@/contexts/i18n-context';

export const DOCUMENT_UPLOAD_ACCEPT = '.pdf,.png,.jpeg,.jp2,.webp,.gif,.bmp,.jpg';

export function validateDocumentFile(file: Pick<File, 'name' | 'type' | 'size'>, t?: TFunc): void {
  const extension = file.name.split('.').pop()?.toLowerCase();
  const isPdf = extension === 'pdf';
  const isImage = ['png', 'jpeg', 'jp2', 'webp', 'gif', 'bmp', 'jpg'].includes(extension || '');
  const mimeType = file.type.toLowerCase();
  const unknownMime = !mimeType || mimeType === 'application/octet-stream';

  if (
    (!isPdf && !isImage) ||
    (!unknownMime && !(isPdf ? mimeType === 'application/pdf' : mimeType.startsWith('image/')))
  ) {
    throw new Error(
      t?.('documentParsing.invalidFile') ||
        'Only PDF and supported image files (PNG, JPEG, JP2, WebP, GIF, BMP) are allowed.'
    );
  }
  if (file.size === 0) {
    throw new Error(t?.('documentParsing.emptyFile') || 'The uploaded file cannot be empty.');
  }
}
