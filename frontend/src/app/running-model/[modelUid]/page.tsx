import RunningModelDetailPageClient from './page-client';
import { SHELL_ROUTE_PARAM } from '@/lib/route-params';

// Server wrapper for static export: provides a placeholder param so a shell
// HTML is emitted for this dynamic route. The backend serves the shell for any
// real model uid; the client component reads the actual value from the URL.
export function generateStaticParams() {
  return [{ modelUid: SHELL_ROUTE_PARAM }];
}

export default function RunningModelDetailPage() {
  return <RunningModelDetailPageClient />;
}
