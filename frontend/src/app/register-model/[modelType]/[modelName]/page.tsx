import RegisterModelEditPageClient from './page-client';
import { SHELL_ROUTE_PARAM } from '@/lib/route-params';

// Server wrapper for static export: provides placeholder params so a shell
// HTML is emitted for this dynamic route. The backend serves the shell for any
// real model type/name; the client component reads the actual values from the
// URL.
export function generateStaticParams() {
  return [{ modelType: SHELL_ROUTE_PARAM, modelName: SHELL_ROUTE_PARAM }];
}

export default function RegisterModelEditPage() {
  return <RegisterModelEditPageClient />;
}
