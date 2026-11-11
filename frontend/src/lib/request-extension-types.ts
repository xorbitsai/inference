import type { AxiosError } from 'axios';
import type { ComponentType } from 'react';

export interface RequestErrorEvent {
  status: number;
  message: string;
  error: AxiosError;
  /** Release this event's lock after handling, or when a dialog closes. */
  release: () => void;
}

export interface RequestErrorHandlerProps {
  event: RequestErrorEvent | null;
}

/** Statuses sharing an event also share a lock and a global React handler. */
export interface RequestErrorExtension {
  statuses: readonly number[];
  event: string;
  Handler: ComponentType<RequestErrorHandlerProps>;
}

/** Reject ambiguous registrations before dispatching or mounting handlers. */
export function validateRequestErrorExtensions(extensions: readonly RequestErrorExtension[]) {
  const events = new Set<string>();
  for (const extension of extensions) {
    if (!/^extension:[a-z0-9-]+:[a-z0-9-]+$/.test(extension.event)) {
      throw new Error('Invalid request extension event: ' + extension.event);
    }
    if (events.has(extension.event)) {
      throw new Error('Duplicate request extension event: ' + extension.event);
    }
    events.add(extension.event);
  }
  return extensions;
}
