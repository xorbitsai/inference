import type { RequestErrorExtension } from '@/lib/request-extension-types';

/** Build-time error handlers contributed by a distribution wrapper. */
export const requestErrorExtensions: readonly RequestErrorExtension[] = [];
