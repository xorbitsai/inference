const AUDIT_WINDOW_MS = 24 * 60 * 60 * 1000;

export const buildAuditCenterHref = (requestId: string, anchorTimestamp?: unknown) => {
  const params = new URLSearchParams({ request_id: requestId });
  const anchor = anchorTimestamp ? new Date(String(anchorTimestamp)) : null;

  if (anchor && !Number.isNaN(anchor.getTime())) {
    params.set('time_from', new Date(anchor.getTime() - AUDIT_WINDOW_MS).toISOString());
    params.set('time_to', new Date(anchor.getTime() + AUDIT_WINDOW_MS).toISOString());
  }

  return `/audit-center?${params.toString()}`;
};
