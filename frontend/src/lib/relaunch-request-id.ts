export function createRelaunchRequestId(service: 'api' | 'supervisor'): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `${service}-${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
}
