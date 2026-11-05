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
  error: RequestErrorEvent | null;
}

/** Statuses sharing an event also share a lock and a global React handler. */
export interface RequestErrorExtension {
  statuses: readonly number[];
  event: string;
  Handler: ComponentType<RequestErrorHandlerProps>;
}
