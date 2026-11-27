/** Shared terminal states for API, Supervisor and Worker replacement operations. */
export const isOperationFinished = (status: string): boolean =>
  status === 'succeeded' || status === 'failed';
